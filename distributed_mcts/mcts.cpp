#include "mcts.hpp"

#include "neural_network.hpp"

#define ZERO_DEBUG

#ifdef ZERO_DEBUG
  #define D_MESS(message) \
    std::cout << message << " at " << __LINE__ << ":" << __FILE__ << std::endl;


#else
  #define D_MESS(message)
#endif
Node::Node()
    : parent_(nullptr),
      is_leaf_(true),
      virtual_loss_(0),
      visit_count_(0),
      prior_(0),
      action_value_(0)
{}

Node::Node(const Node& node)
{
  this->parent_ = node.parent_;
  this->children_ = node.children_;
  this->is_leaf_ = node.is_leaf_;
  this->prior_ = node.prior_;
  this->action_value_ = node.action_value_;

  this->virtual_loss_.store(node.virtual_loss_.load());
  this->visit_count_.store(node.visit_count_.load());
}

Node::Node(Node* parent, double prior)
    : parent_(parent),
      children_(),
      is_leaf_(true),
      virtual_loss_(0),
      visit_count_(0),
      action_value_(0),
      prior_(prior)
{}

void Node::back_propagate(double leaf_value)
{
  if (parent_ != nullptr)
  {
    parent_->back_propagate(-leaf_value);
  }
  virtual_loss_--;

  unsigned visit_count = visit_count_.load();
  {
    std::lock_guard<std::mutex> lock(lock_);
    action_value_ =
        (visit_count * action_value_ + leaf_value) / (visit_count + 1);
  }
  visit_count_++;
}

int Node::select(double c_puct, double c_virtual_loss)
{
  double max_ucb = std::numeric_limits<double>::lowest();
  int best_action = -1;
  node_ptr best_node = {};

  for (const auto&[action, node] : children_)
  {
    // skip empty nodes
    if (!node) 
    {
      continue;
    }

    double ucb = node->upper_confidence_bound();
    if (ucb > max_ucb)
    {
      max_ucb = ucb;
      best_action = action;
      best_node = node;
    }
  }

  if (best_node != nullptr)
  {
    // temporarily apply virtual loss
    best_node->virtual_loss_++;
  }
  else
  {
    D_MESS("Found race condition")
  }
  return best_action;
}

void Node::expand(const std::vector<double>& prior_probabilites)
{
  std::lock_guard<std::mutex> lock(lock_);

  if (!is_leaf_)
  {
    return;
  }

  for (int i = 0; i < prior_probabilites.size(); i++)
  {
    // TODO add epsilon
    if (prior_probabilites[i] > 0)
    {
      children_[i] = std::make_shared<Node>(this, prior_probabilites[i]);
    }
  }

  is_leaf_ = false;
}

double Node::upper_confidence_bound() const
{
  const int N = visit_count_.load();
  double U = prior_ / static_cast<double>(1 + N);
  return action_value_ + U;
}

Node& Node::operator=(const Node& n)
{
  if (this == &n)
  {
    return *this;
  }

  this->parent_ = n.parent_;
  this->children_ = n.children_;
  this->is_leaf_ = n.is_leaf_;
  this->prior_ = n.prior_;
  this->action_value_ = n.action_value_;

  this->virtual_loss_.store(n.virtual_loss_.load());
  this->visit_count_.store(n.visit_count_.load());

  return *this;
}

MonteCarloTreeSearch::MonteCarloTreeSearch(
    NeuralNetwork* network, unsigned n_sim)
    : n_sim_(n_sim),
      root_(std::make_shared<Node>(nullptr, 1)),
      tpool_(1),
      network_(network)
{}

std::vector<double> MonteCarloTreeSearch::get_action_probabilites()
{
  std::vector<double> result(9, 0);
  int visit_sum = 0;
  for (const auto& child : root_->children_)
  {
    int visit_count = child.second->visit_count_.load();
    result[child.first] = visit_count;
    visit_sum += visit_count;
  }

  for (int i = 0; i < 9; i++)
  {
    result[i] /= visit_sum;
  }
  return std::move(result);
}

void MonteCarloTreeSearch::apply_action(int action)
{
  if (action >= 0 && root_->children_.find(action) != root_->children_.end())
  {
    node_ptr new_head = root_->children_[action];
    new_head->parent_ = nullptr;

    root_ = new_head;
  }
  else
  {
    // start new search
    root_ = std::make_shared<Node>(nullptr, 1);
  }
}

std::vector<double> MonteCarloTreeSearch::search(search_t game)
{
  search_internal(game);
  D_MESS("Completed Search");
  return get_action_probabilites();
}

void MonteCarloTreeSearch::search_internal(search_t game)
{
  std::vector<std::future<void>> futures;

  for (int i = 0; i < n_sim_; i++)
  {
    KinaRow gWrapper(3, 3, game.first, game.second);
    // lets hope this works
    auto f =
        std::bind(&MonteCarloTreeSearch::simulate, this, std::placeholders::_1);
    auto future = tpool_.submit(f, gWrapper);

    futures.emplace_back(std::move(future));
  }

  for (int i = 0; i < futures.size(); i++)
  {
    futures[i].wait();
    D_MESS("Collected Thread")
  }
}

void MonteCarloTreeSearch::simulate(KinaRow game)
{
  D_MESS("Stated simulation of Game")
  node_ptr node = root_;

  while (true)
  {
    if (node->is_leaf_node())
    {
      break;
    }

    auto action = node->select(0, 0);

    if (action >= 0)
    {
        // returned valid action
      game.apply_action(action);
      node = node->children_[action];
    }
    else
    {
        // we are at a terminal state for the game
      break;
    }
  }

  int status = game.get_winner();
  double value = 0;
  if (status == -2)
  {
    std::vector<double> action_priors(9, 0);

    auto future = network_->evaluate(&game);
    auto result = future.get();
    action_priors = std::move(result.first);
    value = result.second;


    // set impossible moves to zero
    std::vector<bool> possible = game.get_valid();
    for (int i = 0; i < 9; i++)
    {
      if (!possible[i])
      {
        action_priors[i] = 0;
      }
    }

    // probability normalization
    double p_sum = 0;
    for (int i = 0; i < 9; i++)
    {
      p_sum += action_priors[i];
    }
    for (int i = 0; i < 9; i++)
    {
      action_priors[i] /= p_sum;
    }

    // TODO workaround if all actions invalid

    node->expand(action_priors);
  }
  else
  {
    value = status * game.get_player();
  }
  node->back_propagate(-value);
  D_MESS("Ended simulation of Game")
}
