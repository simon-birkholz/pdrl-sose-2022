#include "kinarow.hpp"

#include <algorithm>
#include <numeric>

KinaRow::KinaRow(
    int boardHeight,
    int boardWidth,
    int k,
    std::vector<int> board,
    int curr_player)
    : height_(boardHeight),
      width_(boardWidth),
      board_(board),
      player_(curr_player),
      k_(k),
      action_space_(boardHeight * boardWidth)
{}

bool KinaRow::check_won_player(const int player) const
{
  for (int y = 0; y < height_; y++)
  {
    for (int x = 0; x < width_; x++)
    {
      const std::vector<std::pair<int, int>> directions = {
          {1, -1}, {1, 0}, {1, 1}, {0, 1}};

      for (const auto &[dx, dy] : directions)
      {
        int xx = x;
        int yy = y;
        int count = 0;
        while (count < k_)
        {
          const int idx = y * width_ + x;
          if (idx >= 0 && idx < board_.size() && board_[idx] == player)
          {
            count++;
            xx += dx;
            yy += dy;
          }
          else
          {
            break;
          }
        }
        if (count >= k_)
        {
          return true;
        }
      }
    }
  }
  return false;
}


std::vector<bool> KinaRow::get_valid() const
{
  std::vector<bool> res(action_space_, false);
  for (int i = 0; i < action_space_; i++)
  {
    res[i] = board_[i] == 0;
  }
  return std::move(res);
}

int KinaRow::get_winner() const
{
  if (check_won_player(1))
  {
    return 1;
  }
  if (check_won_player(-1))
  {
    return -1;
  }
  const auto valid_moves = get_valid();
  int sum = std::accumulate(valid_moves.begin(), valid_moves.end(), 0);
  if (sum == 0)
  {
    // no moves available, therefore draw
    return 0;
  }
  // return -2 if game not ended
  return -2;
}

void KinaRow::apply_action(int action)
{
  board_[action] = player_;
  player_ = -player_;
}
