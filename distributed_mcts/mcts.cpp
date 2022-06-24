#include "mcts.hpp"

#include "neural_network.hpp"

//#define ZERO_DEBUG

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

constexpr double eps = 1e-6;

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

int Node::select(const double c_p, const double c_loss)
{
  double max_ucb = std::numeric_limits<double>::lowest();
  int best_action = -1;
  node_ptr best_node = {};

  for (const auto& [action, node] : children_)
  {
    // skip empty nodes
    if (!node)
    {
      continue;
    }

    double ucb = node->upper_confidence_bound(c_p, c_loss);
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
    if (prior_probabilites[i] > eps)
    {
      children_[i] = std::make_shared<Node>(this, prior_probabilites[i]);
    }
  }

  is_leaf_ = false;
}

double Node::upper_confidence_bound(const double c_p, const double c_v) const
{
  const int N = visit_count_.load();
  double U = c_p * prior_ / static_cast<double>(1 + N);

  double v_loss = c_v * virtual_loss_.load();
  return action_value_ + U - v_loss;
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
    NeuralNetwork* network,
    std::pair<int, int> board_size,
    unsigned k,
    unsigned n_sim,
    unsigned n_threads)
    : n_sim_(n_sim),
      root_(std::make_shared<Node>(nullptr, 1)),
      tpool_(n_threads),
      network_(network),
      board_width_(board_size.first),
      board_height_(board_size.second),
      k_(k),
      c_p_(5),
      c_loss_(3)
{}

std::vector<double> MonteCarloTreeSearch::get_action_probabilites()
{
  const auto action_space = board_width_ * board_height_;
  std::vector<double> result(action_space, 0);
  int visit_sum = 0;
  D_MESS("Containing " + std::to_string(root_->children_.size()) + " nodes")
  for (const auto& [action, child] : root_->children_)
  {
    int visit_count = child->visit_count_.load();
    result[action] = visit_count;
    visit_sum += visit_count;
  }

  if (visit_sum == 0)
  {
    std::cerr << "Invalid state for mcts" << std::endl;
  }
  for (int i = 0; i < action_space; i++)
  {
    result[i] /= static_cast<double>(visit_sum);
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
  assert(
      board_height_ * board_width_ == game.first.size() &&
      "Board size must match initialized values");
  search_internal(game);
  D_MESS("Completed Search");
  return get_action_probabilites();
}

void MonteCarloTreeSearch::search_internal(search_t game)
{
  std::vector<std::future<void>> futures;

  for (int i = 0; i < n_sim_; i++)
  {
    D_MESS(std::string(
        "Querying Sim " + std::to_string(i) + " of total " +
        std::to_string(n_sim_)));

    KinaRow gWrapper(board_height_, board_width_, k_, game.first, game.second);
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

    auto action = node->select(c_p_, c_loss_);

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
  D_MESS("Selected Node")
  const auto action_space = board_width_ * board_height_;
  int status = game.get_winner();
  double value = 0;
  if (status == -2)
  {
    D_MESS("Sending to Neural Network")
    auto future = network_->evaluate(&game);
    auto result = future.get();
    D_MESS("Recieved from Neural Network")
    std::vector<double> action_priors = std::move(result.first);
    assert(
        action_priors.size() == action_space &&
        "Detected invalid return from neural network");
    value = result.second;

    // set impossible moves to zero
    std::vector<bool> possible = game.get_valid();

    assert(
        possible.size() == action_priors.size() &&
        "Detected invalid game wrapper");
    for (int i = 0; i < action_space; i++)
    {
      if (!possible[i])
      {
        action_priors[i] = 0;
      }
    }

    // probability normalization
    double p_sum = 0;
    for (int i = 0; i < action_space; i++)
    {
      p_sum += action_priors[i];
    }
    for (int i = 0; i < action_space; i++)
    {
      action_priors[i] /= p_sum;
    }

    assert(p_sum > eps && "All actions are invalid");
    // TODO workaround if all actions invalid

    D_MESS("Expanding Node");
    node->expand(action_priors);
  }
  else
  {
    value = status * game.get_player();
  }
  D_MESS("Beginning Backpropagate");
  node->back_propagate(-value);
  D_MESS("Ended simulation of Game")
}
