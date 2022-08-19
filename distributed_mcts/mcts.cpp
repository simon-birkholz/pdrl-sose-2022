#include "mcts.hpp"

#include "log.hpp"
#include "neural_network.hpp"

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
  visit_count_++;

  {
    std::lock_guard<std::recursive_mutex> lock(lock_);
    action_value_ =
        (visit_count * action_value_ + leaf_value) / (visit_count + 1);
  }
}

int Node::select(const double c_loss)
{
  double max_ucb = std::numeric_limits<double>::lowest();
  int best_action = -1;
  node_ptr best_node = {};

  for (const auto [action, node] : children_)
  {
    // skip empty nodes
    if (!node)
    {
      continue;
    }

    double ucb = node->upper_confidence_bound(c_loss);
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
    log_error("Found race condition in " + POS);
  }
  return best_action;
}

bool Node::expand(const std::vector<double>& prior_probabilites)
{
  std::lock_guard<std::recursive_mutex> lock(lock_);


  if (!is_leaf_node())
  {
    return false;
  }

  for (int i = 0; i < prior_probabilites.size(); i++)
  {
    if (prior_probabilites[i] > eps)
    {
      children_[i] = std::make_shared<Node>(this, prior_probabilites[i]);
    }
  }

  is_leaf_ = false;
  return true;
}

double Node::upper_confidence_bound(const double c_v) const
{
  const int N = visit_count_.load();
  double U = prior_ / static_cast<double>(1 + N);

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
      c_loss_(0.6)
{}

std::vector<double> MonteCarloTreeSearch::get_action_probabilites()
{
  std::lock_guard<std::mutex> lock(lock_search_);
  const auto action_space = board_width_ * board_height_;
  std::vector<double> result(action_space, 0);
  int visit_sum = 0;
  log_debug(
      "Search tree contains " + std::to_string(root_->children_.size()) +
      " nodes");
  for (const auto& [action, child] : root_->children_)
  {
    int visit_count = child->visit_count_.load();
    result[action] = visit_count;
    visit_sum += visit_count;
  }

  if (visit_sum == 0)
  {
    log_fatal(
        "Search tree doesn't contains child nodes in " + POS +
        ". Invalid state for MCTS");
  }
  for (int i = 0; i < action_space; i++)
  {
    result[i] /= static_cast<double>(visit_sum);
  }
  return std::move(result);
}

void MonteCarloTreeSearch::apply_action(int action)
{
  std::lock_guard<std::mutex> lock(lock_search_);
  if (action >= 0 && root_->children_.find(action) != root_->children_.end())
  {
    node_ptr new_head = root_->children_[action];
    new_head->parent_ = nullptr;

    root_ = new_head;
  }
  else
  {
    //std::cout << "Node has actions ";
    //for (const auto& [action, child] : root_->children_)
    //{
    //  std::cout << action << " ";
    //}
    //std::cout << std::endl;
    // start new search
    root_ = std::make_shared<Node>(nullptr, 1);
    log_debug("Started new Search on action " + std::to_string(action));
  }
}

void MonteCarloTreeSearch::reset()
{
  std::lock_guard<std::mutex> lock(lock_search_);
  root_ = std::make_shared<Node>(nullptr, 1);
}

std::vector<double> MonteCarloTreeSearch::search(search_t game)
{
  assert(
      board_height_ * board_width_ == game.first.size() &&
      "Board size must match initialized values");
  search_internal(game);
  log_debug("Completed search");

  KinaRow gWrapper(board_height_, board_width_, k_, game.first, game.second);
  //if (root_->children_.size() == 0)
  //{
  //  gWrapper.print_board();
  //}
  return get_action_probabilites();
}

void MonteCarloTreeSearch::search_internal(search_t game)
{
  std::lock_guard<std::mutex> lock(lock_search_);
  std::vector<std::future<void>> futures;

  for (int i = 0; i < n_sim_; i++)
  {
    log_debug(std::string(
        "Querying Sim " + std::to_string(i) + " of total " +
        std::to_string(n_sim_)));

    KinaRow gWrapper(board_height_, board_width_, k_, game.first, game.second);
    // lets hope this works
    auto f =
        std::bind(&MonteCarloTreeSearch::simulate, this, std::placeholders::_1);
    auto future = tpool_.submit(f, std::move(gWrapper));

    futures.emplace_back(std::move(future));
  }

  int collect_count = 0;
  for (int i = 0; i < futures.size(); i++)
  {
    try
    {
      futures[i].get();
      collect_count++;
    }
    catch (const std::exception& e)
    {
      log_error(
          "Exception in threads after " + std::to_string(collect_count) +
          " simulations");
    }
    log_debug("Collected Thread");
  }
}

void MonteCarloTreeSearch::simulate(KinaRow game)
{
  log_debug("Stated simulation of Game");
  node_ptr node = root_;

  while (true)
  {
    if (node->is_leaf_node())
    {
      break;
    }

    auto action = node->select(c_loss_);

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

  log_debug("Selected Node");
  const auto action_space = board_width_ * board_height_;
  int status = game.get_winner();
  double value = 0;
  if (status == -2)
  {
    log_debug("Sending to Neural Network");
    auto future = network_->evaluate(&game);
    auto result = future.get();
    log_debug("Recieved from Neural Network");
    std::vector<double> action_priors = std::move(result.first);
    assert(
        action_priors.size() == action_space &&
        "Detected invalid return from neural network");
    value = result.second;

    // set impossible moves to zero
    std::vector<bool> possible = game.get_valid();

    int action_count = action_space;
    assert(
        possible.size() == action_priors.size() &&
        "Detected invalid game wrapper");
    for (int i = 0; i < action_space; i++)
    {
      if (!possible[i])
      {
        action_priors[i] = 0;
        action_count--;
      }
    }

    // probability normalization
    double p_sum = 0;
    for (int i = 0; i < action_space; i++)
    {
      if (action_priors[i] < 0)
      {
        log_warning("detected negative probability in" + POS);
      }
      p_sum += action_priors[i];
    }
    if (p_sum < 1e-3)
    {
      log_warning("detected zero probabilty in" + POS);
    }
    for (int i = 0; i < action_space; i++)
    {
      action_priors[i] /= p_sum;
    }

    assert(p_sum > eps && "All actions are invalid");
    // TODO workaround if all actions invalid

    log_debug("Expanding Node");
    bool was_expanded = node->expand(action_priors);
    if (!was_expanded)
    {
      log_error("Node was selected but already expanded");
    }

    if (action_count != node->children_.size())
    {
      log_error("Invalid node state: child count mismatch");
    }
  }
  else
  {
    value = status * game.get_player();
  }
  log_debug("Beginning Backpropagate");
  node->back_propagate(-value);
  log_debug("Ended simulation of Game");
}

int MonteCarloTreeSearch::get_node_count() const
{
  return root_->get_node_count();
}
int MonteCarloTreeSearch::get_max_depth() const
{
  return root_->get_max_depth();
}

int Node::get_node_count() const
{
  if (is_leaf_node())
    return 0;
  int node_count = 1;
  for (const auto& [action, child] : children_)
  {
    node_count += child->get_node_count();
  }
  return node_count;
}
int Node::get_max_depth() const
{
  if (is_leaf_node())
    return 1;

  int max_depth = 0;
  for (const auto& [action, child] : children_)
  {
    max_depth = std::max({1 + child->get_max_depth(), max_depth});
  }
  return max_depth;
}