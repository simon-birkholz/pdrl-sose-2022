
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
      "D:\\dev\\builds\\alpha-zero\\distributed_mcts\\Debug\\8x8.pt";

  const int board_width = 8;
  const int board_height = 8;
  const auto board_shape = std::make_pair(board_width, board_height);
  const int k = 4;

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
        std::make_shared<MonteCarloTreeSearch>(nn.get(), board_shape, k, 100,1);

    std::cout << "Init: Monte Carlo Tree Search" << std::endl;
    std::vector<int> board(board_width * board_height, 0);

    const auto res = search->search(std::make_pair(board, 1));

    for (int i = 0; i < res.size(); i++)
    {
      std::cout << res[i] << " ";
    }
  }
  catch (const c10::Error& e)
  {
    std::cerr << e.what() << std::endl;
  }
}