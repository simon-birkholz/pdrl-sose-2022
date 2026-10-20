#include "mcts.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

enum class NodeState
{
  leaf,
  expanding,
  expanded
};

struct Node
{
  Node(Node* parent, double prior) : parent(parent), prior(prior) {}

  Node* parent;
  double prior;
  NodeState state = NodeState::leaf;
  std::map<int, std::unique_ptr<Node>> children;
  unsigned visit_count = 0;
  // Seen from the player who made the move leading to this node.
  double value_sum = 0;
  unsigned virtual_loss = 0;
};

namespace
{
  // Pending simulations count as lost visits until their result is backed up.
  double puct(const Node& child, double sqrt_parent_visits, double c_puct)
  {
    const double visits = child.visit_count + child.virtual_loss;
    const double q =
        visits > 0 ? (child.value_sum - child.virtual_loss) / visits : 0;
    return q + c_puct * child.prior * sqrt_parent_visits / (1 + visits);
  }

  std::pair<int, Node*> select_child(const Node& node, double c_puct)
  {
    const double sqrt_visits = std::sqrt(node.visit_count + node.virtual_loss);
    const auto best = std::max_element(
        node.children.begin(),
        node.children.end(),
        [&](const auto& a, const auto& b) {
          return puct(*a.second, sqrt_visits, c_puct) <
                 puct(*b.second, sqrt_visits, c_puct);
        });
    return {best->first, best->second.get()};
  }

  void backpropagate(Node* node, double value)
  {
    for (; node != nullptr; node = node->parent, value = -value)
    {
      node->visit_count++;
      node->value_sum += value;
      node->virtual_loss--;
    }
  }

  void revert_virtual_loss(Node* node)
  {
    for (; node != nullptr; node = node->parent)
    {
      node->virtual_loss--;
    }
  }

  Evaluator::return_t evaluate_legal(Evaluator& evaluator, const KinaRow& game)
  {
    auto [priors, value] = evaluator.evaluate(game).get();
    const auto legal = game.get_valid();
    if (priors.size() != legal.size())
    {
      throw std::runtime_error("Evaluator returned a policy of wrong size");
    }

    std::transform(
        priors.begin(),
        priors.end(),
        legal.begin(),
        priors.begin(),
        [](double p, bool is_legal) {
          return is_legal ? std::max(p, 0.0) : 0;
        });
    if (std::accumulate(priors.begin(), priors.end(), 0.0) <= 0)
    {
      // all mass on illegal moves, fall back to uniform over legal ones
      std::transform(
          legal.begin(), legal.end(), priors.begin(), [](bool is_legal) {
            return is_legal ? 1.0 : 0.0;
          });
    }
    const double total = std::accumulate(priors.begin(), priors.end(), 0.0);
    std::transform(
        priors.begin(), priors.end(), priors.begin(), [total](double p) {
          return p / total;
        });
    return {std::move(priors), value};
  }

  int count_expanded(const Node& node)
  {
    return std::accumulate(
        node.children.begin(),
        node.children.end(),
        node.children.empty() ? 0 : 1,
        [](int sum, const auto& child) {
          return sum + count_expanded(*child.second);
        });
  }

  int max_depth(const Node& node)
  {
    return 1 + std::accumulate(
                   node.children.begin(),
                   node.children.end(),
                   0,
                   [](int depth, const auto& child) {
                     return std::max(depth, max_depth(*child.second));
                   });
  }
} // namespace

MonteCarloTreeSearch::MonteCarloTreeSearch(
    Evaluator* evaluator,
    std::pair<int, int> board_size,
    unsigned k,
    unsigned n_sim,
    unsigned n_threads,
    double c_puct)
    : evaluator_(evaluator),
      board_width_(board_size.first),
      board_height_(board_size.second),
      k_(k),
      n_sim_(n_sim),
      c_puct_(c_puct),
      root_(std::make_unique<Node>(nullptr, 1)),
      tpool_(n_threads)
{}

MonteCarloTreeSearch::~MonteCarloTreeSearch() = default;

std::vector<double> MonteCarloTreeSearch::search(search_t game)
{
  if (game.first.size() != board_width_ * board_height_)
  {
    throw std::invalid_argument("Board size must match initialized values");
  }

  std::lock_guard lock(search_mutex_);
  const KinaRow root_game(
      board_height_, board_width_, k_, std::move(game.first), game.second);

  std::vector<std::future<void>> simulations(n_sim_);
  std::generate(simulations.begin(), simulations.end(), [&] {
    return tpool_.submit([this, &root_game] { simulate(root_game); });
  });

  // every simulation has to finish before the tree may change again
  std::exception_ptr failure;
  for (auto& simulation : simulations)
  {
    try
    {
      simulation.get();
    }
    catch (...)
    {
      if (!failure)
      {
        failure = std::current_exception();
      }
    }
  }
  if (failure)
  {
    std::rethrow_exception(failure);
  }
  return visit_distribution();
}

void MonteCarloTreeSearch::simulate(KinaRow game)
{
  std::unique_lock lock(tree_mutex_);

  Node* node = root_.get();
  node->virtual_loss++;
  while (true)
  {
    expansion_done_.wait(
        lock, [node] { return node->state != NodeState::expanding; });
    if (node->state == NodeState::leaf)
    {
      break;
    }
    const auto [action, child] = select_child(*node, c_puct_);
    game.apply_action(action);
    node = child;
    node->virtual_loss++;
  }

  if (const int winner = game.get_winner(); winner != KinaRow::ongoing)
  {
    backpropagate(node, -winner * game.get_player());
    return;
  }

  node->state = NodeState::expanding;
  lock.unlock();

  Evaluator::return_t evaluation;
  try
  {
    evaluation = evaluate_legal(*evaluator_, game);
  }
  catch (...)
  {
    lock.lock();
    node->state = NodeState::leaf;
    revert_virtual_loss(node);
    lock.unlock();
    expansion_done_.notify_all();
    throw;
  }

  const auto& [priors, value] = evaluation;
  lock.lock();
  for (int action = 0; action < static_cast<int>(priors.size()); action++)
  {
    if (game.is_possible(action))
    {
      node->children.emplace(
          action, std::make_unique<Node>(node, priors[action]));
    }
  }
  node->state = NodeState::expanded;
  backpropagate(node, -value);
  lock.unlock();
  expansion_done_.notify_all();
}

std::vector<double> MonteCarloTreeSearch::visit_distribution() const
{
  const auto& children = root_->children;
  const double total = std::accumulate(
      children.begin(), children.end(), 0.0, [](double sum, const auto& child) {
        return sum + child.second->visit_count;
      });
  if (total == 0)
  {
    throw std::runtime_error(
        "Search visited no moves, the position is terminal or n_sim < 2");
  }

  std::vector<double> distribution(board_width_ * board_height_, 0);
  for (const auto& [action, child] : children)
  {
    distribution[action] = child->visit_count / total;
  }
  return distribution;
}

void MonteCarloTreeSearch::apply_action(int action)
{
  std::lock_guard lock(search_mutex_);
  const auto child = root_->children.find(action);
  if (child == root_->children.end())
  {
    root_ = std::make_unique<Node>(nullptr, 1);
    return;
  }
  auto new_root = std::move(child->second);
  new_root->parent = nullptr;
  root_ = std::move(new_root);
}

void MonteCarloTreeSearch::reset()
{
  std::lock_guard lock(search_mutex_);
  root_ = std::make_unique<Node>(nullptr, 1);
}

int MonteCarloTreeSearch::get_node_count() const
{
  std::lock_guard lock(search_mutex_);
  return count_expanded(*root_);
}

int MonteCarloTreeSearch::get_max_depth() const
{
  std::lock_guard lock(search_mutex_);
  return max_depth(*root_);
}
