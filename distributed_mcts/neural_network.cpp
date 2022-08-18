#include "neural_network.hpp"

#include "log.hpp"

NeuralNetwork::NeuralNetwork(
    std::string model_path, bool use_gpu, unsigned batch_size)
    : use_gpu_(use_gpu),
      batch_size_(batch_size),
      model_path_(model_path),
      torch_(std::make_unique<torch::jit::script::Module>(
          torch::jit::load(model_path.c_str()))),
      task_queue_(),
      abort_(false)
{
  if (use_gpu_)
  {
#if ZERO_USE_CUDA
    // move to gpu
    torch_->to(at::kCUDA);
#endif
  }

  loop_ = std::make_unique<std::thread>([&] {
    while (!abort_)
    {
      while (!torch_)
      {
        constexpr std::chrono::milliseconds timeout(100);
        std::this_thread::sleep_for(timeout);
      }
      intern();
    }
  });
}

void NeuralNetwork::intern()
{
  std::lock_guard<std::mutex> lock(model_lock_);

  std::vector<torch::Tensor> states;
  std::vector<std::promise<return_t>> promises;

  constexpr std::chrono::milliseconds timeout(10);
  bool is_timeout = false;

  while (states.size() < batch_size_ && !is_timeout)
  {
    auto val = task_queue_.try_pop(timeout);
    if (val.has_value())
    {
      states.emplace_back(std::move(val.value().first));
      promises.emplace_back(std::move(val.value().second));
    }
    else
    {
      is_timeout = true;
    }
  }

  if (!torch_)
    return;

  if (states.size() == 0)
    return;
  log_debug("Collected batch of size " + states.size());

#if ZERO_USE_CUDA
  std::vector<torch::jit::IValue> to_network{
      use_gpu_ ? torch::cat(states, 0).to(at::kCUDA) : torch::cat(states, 0)};
#else
  std::vector<torch::jit::IValue> to_network{torch::cat(states, 0)};
#endif

  log_debug("Sending to torch " + states.size());
  auto res = torch_->forward(to_network).toTuple();
  log_debug("Recieved from torch " + states.size());

  torch::Tensor p_Batch =
      res->elements()[0].toTensor().exp().toType(torch::kFloat32).to(at::kCPU);

  torch::Tensor v_Batch =
      res->elements()[1].toTensor().toType(torch::kFloat32).to(at::kCPU);

  for (int i = 0; i < promises.size(); i++)
  {
    torch::Tensor p = p_Batch[i];
    torch::Tensor v = v_Batch[i];

    std::vector<double> probabilites(
        static_cast<float*>(p.data_ptr<float>()),
        static_cast<float*>(p.data_ptr<float>() + p.size(0)));

    double value = v.item<float>();

    return_t ret = std::make_pair(std::move(probabilites), value);

    promises[i].set_value(std::move(ret));
  }
}

void NeuralNetwork::reload_weights()
{
  std::lock_guard<std::mutex> lock(model_lock_);
  torch_ = std::make_unique<torch::jit::script::Module>(
      torch::jit::load(model_path_.c_str()));
}

void NeuralNetwork::unload_model()
{
  std::lock_guard<std::mutex> lock(model_lock_);
  torch_.reset();
}

NeuralNetwork::~NeuralNetwork()
{
  abort_ = true;
  loop_->join();
}

std::future<NeuralNetwork::return_t> NeuralNetwork::evaluate(KinaRow* game)
{
  std::vector<int> board_state = game->get_board();

  const auto [width, height] = game->get_board_size();

  assert(
      width * height == board_state.size() && "Detected invalid game wrapper");

  // assume only tic tac toe for now
  torch::Tensor tmp = torch::from_blob(
      &board_state[0], {1, 1, width, height}, torch::dtype(torch::kInt32));

  // extract feature maps
  torch::Tensor channel0 = tmp.gt(0).toType(torch::kFloat32);
  torch::Tensor channel1 = tmp.lt(0).toType(torch::kFloat32);

  int curr_player = game->get_player();

  if (curr_player == -1)
  {
    std::swap(channel0, channel1);
  }

  torch::Tensor state = torch::cat({channel0, channel1}, 1);

  std::promise<return_t> promise;
  std::future<return_t> future = promise.get_future();

  task_queue_.push(std::make_pair(std::move(state), std::move(promise)));

  return future;
}
