#include "neural_network.hpp"


zero::NeuralNetwork::NeuralNetwork(
    std::string model_path, bool use_gpu, unsigned batch_size)
    : use_gpu_(use_gpu), batch_size_(batch_size)
{}

zero::NeuralNetwork::~NeuralNetwork() {}

std::future<zero::NeuralNetwork::return_type>
    zero::NeuralNetwork::evaluate(KinaRow game)
{
  // assume only tic tac toe for now
  return std::future<return_type>();
}
