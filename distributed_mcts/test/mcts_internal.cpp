
#include "mcts.hpp"
#include "neural_network.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
int main()
{
  std::string model = "./test_script.pt";

  if (!fs::exists(fs::path(model)))
  {
    std::cout << "Model not found" << std::endl;
    return 0;
  }
  std::shared_ptr<NeuralNetwork> nn =
      std::make_shared<NeuralNetwork>(model.c_str(), true, 32);

  std::shared_ptr<MonteCarloTreeSearch> search =
      std::make_shared<MonteCarloTreeSearch>(nn.get(), 100);

  std::vector<int> board = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  search->search(std::make_pair(board, 1));
}