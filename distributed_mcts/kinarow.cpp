#include "kinarow.hpp"

#include <algorithm>
#include <numeric>

KinaRow::KinaRow(
    int boardHeight, int boardWidth, std::vector<int> board, int curr_player)
    : height_(boardHeight),
      width_(boardWidth),
      board_(board),
      player_(curr_player),
      action_space_(boardHeight * boardWidth)
{}

bool check_won_player(const std::vector<int> board, const int player)
{
  for (int i = 0; i < 3; i++)
  {
    // horizontal and vertically
    if (board[i * 3] == player && board[i * 3 + 1] == player &&
        board[i * 3 + 2] == player)
    {
      return true;
    }
    if (board[i] == player && board[3 + i] == player && board[6 + i] == player)
    {
      return true;
    }
  }

  if (board[0] == player && board[4] == player && board[8] == player)
  {
    return true;
  }
  if (board[6] == player && board[4] == player && board[2] == player)
  {
    return true;
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
  if (check_won_player(board_, 1))
  {
    return 1;
  }
  if (check_won_player(board_, -1))
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
  return -1;
}

void KinaRow::apply_action(int action)
{
  board_[action] = player_;
  player_ = -player_;
}
