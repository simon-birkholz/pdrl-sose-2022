#include "neural_network.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace
{
  std::unique_ptr<torch::jit::script::Module>
      load_model(const std::string& path, bool use_gpu)
  {
    auto model =
        std::make_unique<torch::jit::script::Module>(torch::jit::load(path));
    model->eval();
    model->to(use_gpu ? torch::kCUDA : torch::kCPU);
    return model;
  }
} // namespace

NeuralNetwork::NeuralNetwork(
    std::string model_path, bool use_gpu, unsigned batch_size)
    : model_path_(std::move(model_path)),
      use_gpu_(use_gpu),
      batch_size_(batch_size),
      model_(load_model(model_path_, use_gpu_)),
      worker_([this] {
        while (!abort_)
        {
          run_batch(next_batch());
        }
      })
{}

NeuralNetwork::~NeuralNetwork()
{
  abort_ = true;
  worker_.join();
}

void NeuralNetwork::reload_weights()
{
  auto model = load_model(model_path_, use_gpu_);
  std::lock_guard lock(model_mutex_);
  model_ = std::move(model);
}

void NeuralNetwork::unload_model()
{
  std::lock_guard lock(model_mutex_);
  model_.reset();
}

std::future<NeuralNetwork::return_t>
    NeuralNetwork::evaluate(const KinaRow& game)
{
  const auto [width, height] = game.get_board_size();
  const auto board = torch::tensor(game.get_board(), torch::kInt32)
                         .view({1, 1, height, width}) *
                     game.get_player();
  // channel 0 holds the stones of the player to move, like to_nn_view in python
  auto state = torch::cat({board.eq(1), board.eq(-1)}, 1).to(torch::kFloat32);

  std::promise<return_t> promise;
  auto future = promise.get_future();
  task_queue_.push({std::move(state), std::move(promise)});
  return future;
}

std::vector<NeuralNetwork::task_t> NeuralNetwork::next_batch()
{
  // the first wait bounds how long shutdown takes, the second how long a
  // partial batch waits for more requests
  constexpr std::chrono::milliseconds idle_timeout(100);
  constexpr std::chrono::milliseconds gather_timeout(1);

  std::vector<task_t> batch;
  for (auto task = task_queue_.try_pop(idle_timeout); task;
       task = batch.size() < batch_size_ ? task_queue_.try_pop(gather_timeout)
                                         : std::nullopt)
  {
    batch.push_back(std::move(*task));
  }
  return batch;
}

std::vector<NeuralNetwork::return_t>
    NeuralNetwork::forward(const std::vector<torch::Tensor>& states)
{
  c10::InferenceMode guard;
  std::lock_guard lock(model_mutex_);
  if (!model_)
  {
    throw std::runtime_error("No model loaded");
  }

  const auto device = use_gpu_ ? torch::kCUDA : torch::kCPU;
  const auto output =
      model_->forward({torch::cat(states).to(device)}).toTuple();
  const auto policies = output->elements()[0]
                            .toTensor()
                            .exp()
                            .to(torch::kCPU, torch::kFloat64)
                            .contiguous();
  const auto values =
      output->elements()[1].toTensor().to(torch::kCPU, torch::kFloat64);

  std::vector<return_t> results(states.size());
  for (size_t i = 0; i < results.size(); i++)
  {
    const double* policy = policies[i].data_ptr<double>();
    results[i] = {
        std::vector<double>(policy, policy + policies.size(1)),
        values[i].item<double>()};
  }
  return results;
}

void NeuralNetwork::run_batch(std::vector<task_t> batch)
{
  if (batch.empty())
  {
    return;
  }

  std::vector<torch::Tensor> states(batch.size());
  std::transform(batch.begin(), batch.end(), states.begin(), [](const auto& t) {
    return t.first;
  });

  try
  {
    auto results = forward(states);
    for (size_t i = 0; i < batch.size(); i++)
    {
      batch[i].second.set_value(std::move(results[i]));
    }
  }
  catch (...)
  {
    for (auto& [state, promise] : batch)
    {
      promise.set_exception(std::current_exception());
    }
  }
}
