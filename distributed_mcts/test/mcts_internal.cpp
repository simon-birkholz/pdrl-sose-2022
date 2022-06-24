
#define ZERO_DEBUG

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
  std::string model =
      "D:\\dev\\builds\\alpha-zero\\distributed_mcts\\Release\\final.pt";

  const int board_width = 3;
  const int board_height = 3;
  const auto board_shape = std::make_pair(board_width, board_height);
  const int k = 3;

  if (!fs::exists(fs::path(model)))
  {
    std::cout << "Model not found" << std::endl;
    return 0;
  }
  try
  {
    std::shared_ptr<NeuralNetwork> nn =
        std::make_shared<NeuralNetwork>(model.c_str(), true, 32);

    std::cout << "Init: Neural Network" << std::endl;

    std::shared_ptr<MonteCarloTreeSearch> search =
        std::make_shared<MonteCarloTreeSearch>(nn.get(), board_shape, k, 100);

    std::cout << "Init: Monte Carlo Tree Search" << std::endl;
    std::vector<int> board = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    search->search(std::make_pair(board, 1));
  }
  catch (const c10::Error& e)
  {
    std::cerr << e.what() << std::endl;
  }
}