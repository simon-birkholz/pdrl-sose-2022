#pragma once

#include "kinarow.hpp"
#include "waiting_queue.hpp"

#include <future>
#include <string>
#include <torch/script.h>
#include <vector>

namespace zero
{
  class NeuralNetwork
  {
  public:
    using return_t = std::pair<std::vector<double>, double>;

  public:
  public:
    NeuralNetwork(NeuralNetwork const&) = delete;
    NeuralNetwork(NeuralNetwork&&) = default;
    NeuralNetwork& operator=(NeuralNetwork const&) = delete;
    NeuralNetwork& operator=(NeuralNetwork&&) = default;

    NeuralNetwork(std::string model_path, bool use_gpu, unsigned batch_size);
    ~NeuralNetwork();

  public:
    void set_batch_size(unsigned batch_size) { this->batch_size_ = batch_size; }

    std::future<return_t> evaluate(KinaRow* game);

    void intern();

  private:
    using task_t = std::pair<torch::Tensor, std::promise<return_t>>;

    common::waiting_queue<task_t> task_queue_;

    unsigned batch_size_;
    bool use_gpu_;
    std::unique_ptr<std::thread> loop_;
    bool abort_;
    std::shared_ptr<torch::jit::script::Module> torch_;
  };

} // namespace zero