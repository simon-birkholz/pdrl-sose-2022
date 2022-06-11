#pragma once

#include <future>
#include <string>
#include <vector>

#include "kinarow.hpp"

namespace zero
{
  class NeuralNetwork
  {
  public:
    using return_type = std::pair<std::vector<double>, double>;

  public:
    static NeuralNetwork& instance()
    {
      static NeuralNetwork instance(
          "test", false, 10); // Guaranteed to be destroyed.
                              // Instantiated on first use.
      return instance;
    }
    NeuralNetwork(std::string model_path, bool use_gpu, unsigned batch_size);
    ~NeuralNetwork();

  public:
    void set_batch_size(unsigned batch_size) { this->batch_size_ = batch_size; }

    std::future<return_type> evaluate(KinaRow game);

  private:
    unsigned batch_size_;
    bool use_gpu_;
  };

} // namespace zero