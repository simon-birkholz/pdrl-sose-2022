#pragma once

#include "evaluator.hpp"
#include "waiting_queue.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>
#include <vector>

// Evaluates positions with a model package compiled by torch.export and
// AOTInductor (NetworkWrapper.export_model) that returns
// (log move probabilities, value). Requests from concurrent searches are
// collected into batches and run as one forward pass on a worker thread.
class NeuralNetwork : public Evaluator
{
public:
  NeuralNetwork(std::string model_path, bool use_gpu, unsigned batch_size);
  ~NeuralNetwork() override;

  void set_batch_size(unsigned batch_size) { batch_size_ = batch_size; }

  void reload_weights();
  void unload_model();

  std::future<return_t> evaluate(const KinaRow& game) override;

private:
  using task_t = std::pair<torch::Tensor, std::promise<return_t>>;

  std::vector<task_t> next_batch();
  std::vector<return_t> forward(const std::vector<torch::Tensor>& states);
  void run_batch(std::vector<task_t> batch);

  std::string model_path_;
  bool use_gpu_;
  std::atomic<unsigned> batch_size_;
  std::mutex model_mutex_;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> model_;
  common::waiting_queue<task_t> task_queue_;
  std::atomic<bool> abort_{false};
  std::thread worker_;
};
