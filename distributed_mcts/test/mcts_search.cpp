#include "evaluator.hpp"
#include "kinarow.hpp"
#include "mcts.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

namespace
{
  // Uniform priors and a neutral value, answered asynchronously with some
  // latency so that concurrent simulations overlap like with a real network.
  class UniformEvaluator : public Evaluator
  {
  public:
    std::future<return_t> evaluate(const KinaRow& game) override
    {
      calls++;
      const auto [width, height] = game.get_board_size();
      return std::async(std::launch::async, [n = width * height] {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        return return_t(std::vector<double>(n, 1.0 / n), 0.0);
      });
    }

    std::atomic<int> calls{0};
  };

  class FailingEvaluator : public Evaluator
  {
  public:
    std::future<return_t> evaluate(const KinaRow&) override
    {
      std::promise<return_t> promise;
      promise.set_exception(
          std::make_exception_ptr(std::runtime_error("network down")));
      return promise.get_future();
    }
  };

  // all prior mass on one move and a neutral value, answered synchronously
  class PeakedEvaluator : public Evaluator
  {
  public:
    explicit PeakedEvaluator(int action) : action_(action) {}

    std::future<return_t> evaluate(const KinaRow& game) override
    {
      const auto [width, height] = game.get_board_size();
      std::vector<double> priors(width * height, 0.0);
      priors[action_] = 1;
      std::promise<return_t> promise;
      promise.set_value({priors, 0.0});
      return promise.get_future();
    }

  private:
    int action_;
  };

  int visited_moves(const std::vector<double>& distribution)
  {
    return static_cast<int>(
        std::count_if(distribution.begin(), distribution.end(), [](double p) {
          return p > 0;
        }));
  }

  int best_action(const std::vector<double>& distribution)
  {
    return static_cast<int>(std::distance(
        distribution.begin(),
        std::max_element(distribution.begin(), distribution.end())));
  }

  double total(const std::vector<double>& distribution)
  { return std::accumulate(distribution.begin(), distribution.end(), 0.0); }
} // namespace

TEST_CASE("Every simulation expands a distinct node", "[mcts]")
{
  // threads colliding on a pending leaf must wait instead of evaluating it again
  UniformEvaluator evaluator;
  constexpr int sims = 2000;
  MonteCarloTreeSearch search(&evaluator, {8, 8}, 4, sims, 16);

  search.search({std::vector<int>(64, 0), 1});

  REQUIRE(search.get_node_count() == sims);
  REQUIRE(evaluator.calls == sims);
}

TEST_CASE("Visit distribution covers exactly the legal moves", "[mcts]")
{
  UniformEvaluator evaluator;
  MonteCarloTreeSearch search(&evaluator, {3, 3}, 3, 400, 8);
  const std::vector<int> board = {1, -1, 0, 0, 1, 0, 0, -1, 0};

  const auto distribution = search.search({board, -1});

  REQUIRE(total(distribution) == Approx(1.0));
  for (size_t i = 0; i < board.size(); i++)
  {
    CAPTURE(i);
    REQUIRE((distribution[i] > 0) == (board[i] == 0));
  }
}

TEST_CASE("Finds the immediate win", "[mcts]")
{
  UniformEvaluator evaluator;
  MonteCarloTreeSearch search(&evaluator, {3, 3}, 3, 400, 8);

  const auto distribution = search.search({{1, 1, 0, -1, -1, 0, 0, 0, 0}, 1});

  REQUIRE(best_action(distribution) == 2);
}

TEST_CASE("Blocks the opponents win", "[mcts]")
{
  // only a two ply lookahead with correct value signs sees the threat at 2
  UniformEvaluator evaluator;
  MonteCarloTreeSearch search(&evaluator, {3, 3}, 3, 800, 8);

  const auto distribution = search.search({{-1, -1, 0, 0, 1, 0, 0, 0, 1}, 1});

  REQUIRE(best_action(distribution) == 2);
}

TEST_CASE("Applying a searched move keeps its subtree", "[mcts]")
{
  UniformEvaluator evaluator;
  MonteCarloTreeSearch search(&evaluator, {3, 3}, 3, 200, 4);
  std::vector<int> board(9, 0);

  const int action = best_action(search.search({board, 1}));
  search.apply_action(action);
  REQUIRE(search.get_node_count() > 0);

  board[action] = 1;
  const auto distribution = search.search({board, -1});
  REQUIRE(distribution[action] == 0);
  REQUIRE(total(distribution) == Approx(1.0));

  search.apply_action(-1);
  REQUIRE(search.get_node_count() == 0);
}

TEST_CASE("Evaluator failures propagate instead of hanging", "[mcts]")
{
  FailingEvaluator evaluator;
  MonteCarloTreeSearch search(&evaluator, {3, 3}, 3, 50, 4);

  REQUIRE_THROWS_AS(
      search.search({std::vector<int>(9, 0), 1}), std::runtime_error);
  REQUIRE_THROWS_AS(
      search.search({std::vector<int>(9, 0), 1}), std::runtime_error);
}

TEST_CASE("Invalid positions are rejected", "[mcts]")
{
  UniformEvaluator evaluator;
  MonteCarloTreeSearch search(&evaluator, {3, 3}, 3, 50, 4);

  REQUIRE_THROWS_AS(
      search.search({std::vector<int>(16, 0), 1}), std::invalid_argument);
  REQUIRE_THROWS_AS(
      search.search({{1, 1, 1, -1, -1, 0, 0, 0, 0}, -1}), std::runtime_error);
  REQUIRE_THROWS_AS(
      MonteCarloTreeSearch(&evaluator, {3, 3}, 3, 1, 4), std::invalid_argument);
}

TEST_CASE("Dirichlet noise spreads visits at the root", "[mcts]")
{
  // without noise moves with zero prior never get an exploration bonus
  PeakedEvaluator evaluator(27);
  const search_t empty_board = {std::vector<int>(64, 0), 1};

  MonteCarloTreeSearch plain(&evaluator, {8, 8}, 4, 64, 1);
  const auto without_noise = plain.search(empty_board);
  REQUIRE(visited_moves(without_noise) == 1);
  REQUIRE(without_noise[27] == 1);

  MonteCarloTreeSearch noisy(&evaluator, {8, 8}, 4, 64, 4, 1.5, 0.3, 0.25);
  const auto with_noise = noisy.search(empty_board);
  REQUIRE(visited_moves(with_noise) > 1);
  REQUIRE(total(with_noise) == Approx(1.0));
}

TEST_CASE("Self play with tree reuse only plays legal moves", "[mcts]")
{
  UniformEvaluator evaluator;
  MonteCarloTreeSearch search(&evaluator, {4, 4}, 3, 100, 8);

  for (int i = 0; i < 20; i++)
  {
    KinaRow game(4, 4, 3, std::vector<int>(16, 0), 1);
    search.reset();
    while (game.get_winner() == KinaRow::ongoing)
    {
      const auto distribution =
          search.search({game.get_board(), game.get_player()});
      const int action = best_action(distribution);

      REQUIRE(total(distribution) == Approx(1.0));
      REQUIRE(game.is_possible(action));
      search.apply_action(action);
      game.apply_action(action);
    }
  }
}
