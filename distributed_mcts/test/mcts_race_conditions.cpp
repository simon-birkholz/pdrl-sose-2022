#define ZERO_DEBUG

#include "kinarow.hpp"
#include "mcts.hpp"
#include "neural_network.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>


#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

namespace fs = std::filesystem;

TEST_CASE("Testing 3x3 Kinarow", "zero::kinarow")
{
  std::string model =
      "D:\\dev\\builds\\alpha-zero\\distributed_mcts\\Debug\\8x8.pt";

  const int board_width = 8;
  const int board_height = 8;
  const auto board_shape = std::make_pair(board_width, board_height);
  const int k = 4;

  REQUIRE(fs::exists(fs::path(model)));


  constexpr int SIMS = 10000;
  std::shared_ptr<NeuralNetwork> nn =
      std::make_shared<NeuralNetwork>(model.c_str(), true, 32);

  std::cout << "Init: Neural Network" << std::endl;

  std::shared_ptr<MonteCarloTreeSearch> search =
      std::make_shared<MonteCarloTreeSearch>(nn.get(), board_shape, k, SIMS, 1);

  std::cout << "Init: Monte Carlo Tree Search" << std::endl;
  std::vector<int> board(board_width * board_height, 0);

  const auto res = search->search(std::make_pair(board, 1));

  const auto node_cnt = search->get_node_count();

  REQUIRE(node_cnt == SIMS);
}