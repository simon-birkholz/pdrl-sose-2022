#include "mcts.hpp"

#include "neural_network.hpp"

zero::Node::Node()
    : parent_(nullptr),
      is_terminal_state_(true),
      virtual_loss_(0),
      visit_count_(0),
      prior_(0),
      action_value_(0)
{}

zero::Node::Node(const Node& node)
{
  this->parent_ = node.parent_;
  this->children_ = node.children_;
  this->is_terminal_state_ = node.is_terminal_state_;
  this->prior_ = node.prior_;
  this->action_value_ = node.action_value_;

  this->virtual_loss_.store(node.virtual_loss_.load());
  this->visit_count_.store(node.visit_count_.load());
}

zero::Node::Node(Node* parent, double prior)
    : parent_(parent),
      children_(),
      is_terminal_state_(true),
      virtual_loss_(0),
      visit_count_(0),
      action_value_(0),
      prior_(prior)
{}

void zero::Node::back_propagate(double leaf_value)
{
  if (parent_ != nullptr)
  {
    this->parent_->back_propagate(-leaf_value);
  }
  this->virtual_loss_--;

  unsigned visit_count = visit_count_.load();
  {
    std::lock_guard<std::mutex> lock(lock_);
    action_value_ =
        (visit_count * action_value_ + leaf_value) / (visit_count + 1);
  }
  visit_count_++;
}

unsigned zero::Node::select(double c_puct, double c_virtual_loss)
{
  double max_ucb = std::numeric_limits<double>::min();
  unsigned best_action;
  node_ptr best_node = {};

  for (const auto& child : children_)
  {
    double ucb = child.second->upper_confidence_bound();
    if (ucb > max_ucb)
    {
      max_ucb = ucb;
      best_action = child.first;
      best_node = child.second;
    }
  }

  // temporarily apply virtual loss
  best_node->virtual_loss_++;
  return best_action;
}

void zero::Node::expand(const std::vector<double>& prior_probabilites)
{
  std::lock_guard<std::mutex> lock(lock_);

  if (!is_terminal_state_)
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

  is_terminal_state_ = false;
}

double zero::Node::upper_confidence_bound() const
{
  double U = prior_ / (1 + visit_count_);
  return action_value_ + U;
}

zero::Node& zero::Node::operator=(const zero::Node& n)
{
  if (this == &n)
  {
    return *this;
  }

  this->parent_ = n.parent_;
  this->children_ = n.children_;
  this->is_terminal_state_ = n.is_terminal_state_;
  this->prior_ = n.prior_;
  this->action_value_ = n.action_value_;

  this->virtual_loss_.store(n.virtual_loss_.load());
  this->visit_count_.store(n.visit_count_.load());

  return *this;
}

zero::MonteCarloTreeSearch::MonteCarloTreeSearch(unsigned n_sim)
    : n_sim_(n_sim), root_(std::make_shared<Node>(nullptr, 1)), tpool_(1)
{}

std::vector<double> zero::MonteCarloTreeSearch::get_action_probabilites()
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

void zero::MonteCarloTreeSearch::apply_action(int action)
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

void zero::MonteCarloTreeSearch::search(SearchState game)
{
  std::vector<std::future<void>> futures;

  for (int i = 0; i < n_sim_; i++)
  {
    // lets hope this works
    auto f = std::bind(&simulate, this, std::placeholders::_1);
    auto future = tpool_.submit(f, game);

    futures.emplace_back(std::move(future));
  }

  for (int i = 0; i < futures.size(); i++)
  {
    futures[i].wait();
  }
}

void zero::MonteCarloTreeSearch::simulate(SearchState game)
{
  node_ptr node = root_;

  while (true)
  {
    if (node->is_terminal_state())
    {
      break;
    }

    auto action = node->select(0, 0);

    //TODO execute move in game wrapper

    node = node->children_[action];
  }

  //TODO get game status
  int status = 0;
  double value = 0;
  if (status == 0)
  {
    std::vector<double> action_priors(9, 0);

    auto future = NeuralNetwork::instance().evaluate(game.board);
    auto result = future.get();
    action_priors = std::move(result.first);
    value = result.second;

    // TODO do renormalization
    // set invalid actions to zero prob

    // TODO workaround if all actions invalid

    node->expand(action_priors);
  }
  else
  {
    bool isWinner = true;
    value = isWinner ? 1 : -1;
  }
  node->back_propagate(-value);
}
