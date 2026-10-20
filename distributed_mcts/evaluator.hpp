#pragma once

#include "kinarow.hpp"

#include <future>
#include <utility>
#include <vector>

// Maps a position to (move probabilities, value), both seen from the player
// to move. Must be safe to call from several search threads at once.
class Evaluator
{
public:
  using return_t = std::pair<std::vector<double>, double>;

  virtual ~Evaluator() = default;
  virtual std::future<return_t> evaluate(const KinaRow& game) = 0;
};
