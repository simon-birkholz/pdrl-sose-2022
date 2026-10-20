#pragma once

#include "evaluator.hpp"
#include "kinarow.hpp"
#include "threadpool.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <random>
#include <utility>
#include <vector>

struct Node;

using search_t = std::pair<std::vector<int>, int>;

// Tree parallel MCTS. A single mutex guards the whole tree: selection and
// backup are cheap compared to an evaluation, which runs without holding it.
// Virtual loss spreads concurrent simulations over different paths, and a
// node that is being evaluated is never handed to a second thread.
// dirichlet_alpha > 0 mixes Dirichlet noise into the priors of each new root,
// for exploration in self-play.
class MonteCarloTreeSearch
{
public:
  MonteCarloTreeSearch(
      Evaluator* evaluator,
      std::pair<int, int> board_size,
      unsigned k,
      unsigned n_sim,
      unsigned n_threads,
      double c_puct = 1.5,
      double dirichlet_alpha = 0,
      double noise_fraction = 0.25);
  ~MonteCarloTreeSearch();

  // Runs n_sim simulations from the given (board, player to move) and returns
  // the root visit distribution over all board cells.
  std::vector<double> search(search_t game);

  void apply_action(int action);
  void reset();

  int get_node_count() const;
  int get_max_depth() const;

private:
  void simulate(KinaRow game);
  std::vector<double> visit_distribution() const;

  Evaluator* evaluator_;
  unsigned board_width_;
  unsigned board_height_;
  unsigned k_;
  unsigned n_sim_;
  double c_puct_;
  double dirichlet_alpha_;
  double noise_fraction_;
  std::mt19937 rng_;
  std::unique_ptr<Node> root_;
  bool root_has_noise_ = false;
  mutable std::mutex search_mutex_;
  std::mutex tree_mutex_;
  std::condition_variable expansion_done_;
  // Declared last so its workers are joined before the tree is destroyed.
  common::thread_pool tpool_;
};
