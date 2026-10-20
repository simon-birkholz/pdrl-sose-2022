#include "mcts.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

enum class NodeState
{
  leaf,
  expanding,
  expanded,
  terminal
};

struct Node
{
  Node(Node* parent, double prior) : parent(parent), prior(prior) {}

  Node* parent;
  // only changed while no simulation runs
  double prior;

  // Seen from the player who made the move leading to this node. Relaxed
  // atomics suffice: the tree structure is published through expansion_mutex,
  // the statistics only steer selection and may be read torn.
  std::atomic<unsigned> visit_count{0};
  std::atomic<double> value_sum{0};
  std::atomic<unsigned> virtual_loss{0};

  std::mutex expansion_mutex;
  std::condition_variable expansion_done;
  NodeState state = NodeState::leaf;
  // set once under expansion_mutex before state becomes expanded, read only
  // afterwards
  std::map<int, std::unique_ptr<Node>> children;
};

namespace
{
  constexpr auto relaxed = std::memory_order_relaxed;

  std::pair<int, Node*> select_child(const Node& node, double c_puct)
  {
    const double sqrt_visits = std::sqrt(
        node.visit_count.load(relaxed) + node.virtual_loss.load(relaxed));

    double best_score = std::numeric_limits<double>::lowest();
    std::pair<int, Node*> best = {-1, nullptr};
    for (const auto& [action, child] : node.children)
    {
      // pending simulations count as lost visits
      const unsigned virtual_loss = child->virtual_loss.load(relaxed);
      const double visits = child->visit_count.load(relaxed) + virtual_loss;
      const double q =
          visits > 0 ? (child->value_sum.load(relaxed) - virtual_loss) / visits
                     : 0;
      const double score =
          q + c_puct * child->prior * sqrt_visits / (1 + visits);
      if (score > best_score)
      {
        best_score = score;
        best = {action, child.get()};
      }
    }
    return best;
  }

  void backpropagate(Node* node, double value)
  {
    for (; node != nullptr; node = node->parent, value = -value)
    {
      node->visit_count.fetch_add(1, relaxed);
      node->value_sum.fetch_add(value, relaxed);
      node->virtual_loss.fetch_sub(1, relaxed);
    }
  }

  void revert_virtual_loss(Node* node)
  {
    for (; node != nullptr; node = node->parent)
    {
      node->virtual_loss.fetch_sub(1, relaxed);
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

  void add_dirichlet_noise(
      Node& node, double alpha, double fraction, std::mt19937& rng)
  {
    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> noise(node.children.size());
    std::generate(noise.begin(), noise.end(), [&] { return gamma(rng); });
    const double total = std::accumulate(noise.begin(), noise.end(), 0.0);

    auto sample = noise.begin();
    for (auto& [action, child] : node.children)
    {
      child->prior =
          (1 - fraction) * child->prior + fraction * *sample++ / total;
    }
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
    double c_puct,
    double dirichlet_alpha,
    double noise_fraction)
    : evaluator_(evaluator),
      board_width_(board_size.first),
      board_height_(board_size.second),
      k_(k),
      n_sim_(n_sim),
      c_puct_(c_puct),
      dirichlet_alpha_(dirichlet_alpha),
      noise_fraction_(noise_fraction),
      rng_(std::random_device{}()),
      root_(std::make_unique<Node>(nullptr, 1)),
      tpool_(n_threads)
{
  if (n_sim_ < 2)
  {
    throw std::invalid_argument("n_sim must be at least 2");
  }
}

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

  // runs alone so that the root is expanded before noise is mixed in
  simulate(root_game);
  if (dirichlet_alpha_ > 0 && !root_has_noise_)
  {
    add_dirichlet_noise(*root_, dirichlet_alpha_, noise_fraction_, rng_);
    root_has_noise_ = true;
  }

  std::vector<std::future<void>> simulations(n_sim_ - 1);
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
  Node* node = root_.get();
  node->virtual_loss.fetch_add(1, relaxed);
  bool claimed = false;
  while (true)
  {
    std::unique_lock lock(node->expansion_mutex);
    node->expansion_done.wait(
        lock, [node] { return node->state != NodeState::expanding; });
    if (node->state == NodeState::leaf)
    {
      claimed = game.get_winner() == KinaRow::ongoing;
      node->state = claimed ? NodeState::expanding : NodeState::terminal;
      break;
    }
    if (node->state == NodeState::terminal)
    {
      break;
    }
    lock.unlock();

    const auto [action, child] = select_child(*node, c_puct_);
    game.apply_action(action);
    node = child;
    node->virtual_loss.fetch_add(1, relaxed);
  }

  if (!claimed)
  {
    backpropagate(node, -game.get_winner() * game.get_player());
    return;
  }

  Evaluator::return_t evaluation;
  try
  {
    evaluation = evaluate_legal(*evaluator_, game);
  }
  catch (...)
  {
    {
      std::lock_guard lock(node->expansion_mutex);
      node->state = NodeState::leaf;
    }
    node->expansion_done.notify_all();
    revert_virtual_loss(node);
    throw;
  }

  const auto& [priors, value] = evaluation;
  std::map<int, std::unique_ptr<Node>> children;
  for (int action = 0; action < static_cast<int>(priors.size()); action++)
  {
    if (game.is_possible(action))
    {
      children.emplace(action, std::make_unique<Node>(node, priors[action]));
    }
  }
  {
    std::lock_guard lock(node->expansion_mutex);
    node->children = std::move(children);
    node->state = NodeState::expanded;
  }
  node->expansion_done.notify_all();
  backpropagate(node, -value);
}

std::vector<double> MonteCarloTreeSearch::visit_distribution() const
{
  const auto& children = root_->children;
  const double total = std::accumulate(
      children.begin(), children.end(), 0.0, [](double sum, const auto& child) {
        return sum + child.second->visit_count.load(relaxed);
      });
  if (total == 0)
  {
    throw std::runtime_error(
        "Search visited no moves, the position is terminal");
  }

  std::vector<double> distribution(board_width_ * board_height_, 0);
  for (const auto& [action, child] : children)
  {
    distribution[action] = child->visit_count.load(relaxed) / total;
  }
  return distribution;
}

void MonteCarloTreeSearch::apply_action(int action)
{
  std::lock_guard lock(search_mutex_);
  root_has_noise_ = false;
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
  root_has_noise_ = false;
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
