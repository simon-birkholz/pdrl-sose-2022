#pragma once

#include "kinarow.hpp"
#include "neural_network.hpp"
#include "threadpool.hpp"

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

class Node;

using node_ptr = std::shared_ptr<Node>;

struct SearchState
{
  std::vector<int> board;
  int player;
};
class MonteCarloTreeSearch
{
  friend class Node;

public:
  MonteCarloTreeSearch(std::shared_ptr<NeuralNetwork> network, unsigned n_sim);

  std::vector<double> get_action_probabilites();

  void apply_action(int action);

  void search(SearchState game);

  // Executes one Simulation
  void simulate(KinaRow game);

private:
  unsigned n_sim_;
  node_ptr root_;
  common::thread_pool tpool_;
  std::shared_ptr<NeuralNetwork> network_;
};

class Node
{
public:
  friend class MonteCarloTreeSearch;

  Node();
  Node(const Node& node);
  Node(Node* parent, double prior);
  Node& operator=(const Node& n);

public:
  bool is_terminal_state() const { return this->is_terminal_state_; }
  void back_propagate(double leaf_value);

  unsigned select(double c_puct, double c_virtual_loss);
  void expand(const std::vector<double>& prior_probabilites);

  double upper_confidence_bound() const;


private:
  Node* parent_;
  std::map<unsigned, node_ptr> children_;
  bool is_terminal_state_;
  std::mutex lock_;
  std::atomic<unsigned> visit_count_;
  double prior_;
  double action_value_;
  std::atomic<int> virtual_loss_;
};
