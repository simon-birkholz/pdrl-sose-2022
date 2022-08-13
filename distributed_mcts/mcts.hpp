#pragma once

#include "kinarow.hpp"
#include "neural_network.hpp"
#include "threadpool.hpp"
//#include "locked_ptr.hpp"

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>


class Node;

using node_ptr = std::shared_ptr<Node>;
//using locked_node_ptr = common::locked_ptr<Node>;

using search_t = std::pair<std::vector<int>, int>;
class MonteCarloTreeSearch
{
  friend class Node;

public:
  // smart pointer too much headdache

  MonteCarloTreeSearch(
      NeuralNetwork* network,
      std::pair<int, int> board_size,
      unsigned k,
      unsigned n_sim,
      unsigned n_threads);

  std::vector<double> get_action_probabilites();

  void apply_action(int action);

  void reset();

  std::vector<double> search(search_t game);

  // Executes one Simulation
  void simulate(KinaRow game);

  int get_node_count() const;
  int get_max_depth() const;

private:
  void search_internal(search_t);
  unsigned n_sim_;
  unsigned k_;
  node_ptr root_;
  common::thread_pool tpool_;
  NeuralNetwork* network_;
  unsigned board_width_;
  unsigned board_height_;
  double c_loss_;
  std::mutex lock_search_;
};

class Node
{
public:
  friend class MonteCarloTreeSearch;

  Node();
  Node(const Node& node);
  Node(Node* parent, double prior);
  Node& operator=(const Node& n);

private:

public:
  inline bool is_leaf_node() const
  {
    return this->is_leaf_;
  }
  void back_propagate(double leaf_value);

  int select( const double c_loss);
  bool expand(const std::vector<double>& prior_probabilites);

  double upper_confidence_bound(const double c_loss) const;

  int get_node_count() const;
  int get_max_depth() const;

private:
  Node* parent_;
  std::map<unsigned, node_ptr> children_;
  bool is_leaf_;
  std::recursive_mutex lock_;
  std::atomic<unsigned> visit_count_;
  double prior_;
  double action_value_;
  std::atomic<int> virtual_loss_;
};
