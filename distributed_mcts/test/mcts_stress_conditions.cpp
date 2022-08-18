#define ZERO_DEBUG

#include "kinarow.hpp"
#include "log.hpp"
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
  spdlog::set_level(spdlog::level::info);

  std::string model =
      "D:\\dev\\builds\\alpha-zero\\distributed_mcts\\Debug\\defekt.pt";

  const int board_width = 3;
  const int board_height = 3;
  const auto board_shape = std::make_pair(board_width, board_height);
  const int k = 3;

  REQUIRE(fs::exists(fs::path(model)));


  constexpr int SIMS = 200;

  constexpr int GAME_COUNT = 100;
  std::shared_ptr<NeuralNetwork> nn =
      std::make_shared<NeuralNetwork>(model.c_str(), true, 32);

  log_debug("Init: Neural Network");

  std::shared_ptr<MonteCarloTreeSearch> search =
      std::make_shared<MonteCarloTreeSearch>(nn.get(), board_shape, k, SIMS, 1);

  log_debug("Init: Monte Carlo Tree Search");

  for (int i = 0; i < GAME_COUNT; i++)
  {
    std::vector<int> board(board_width * board_height, 0);

    KinaRow wrapper(board_width, board_height, k, board, 1);
    search->reset();

    while (wrapper.get_winner() == -2)
    {
      const auto res = search->search(std::make_pair(wrapper.get_board(), wrapper.get_player()));

      int action = std::distance(res.begin(),std::max_element(res.begin(), res.end()));

      REQUIRE(wrapper.is_possible(action));
      log_debug(
          "Current node count: " + std::to_string(search->get_node_count()));
      log_debug("Applying action " + std::to_string(action));

      search->apply_action(action);
      wrapper.apply_action(action);
    }

    if (i % 10 == 0) 
    {
      std::cout << ".";
    }
  }
  std::cout << std::endl;
}