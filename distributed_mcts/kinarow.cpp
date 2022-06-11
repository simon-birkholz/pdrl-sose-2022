#include "kinarow.hpp"


KinaRow::KinaRow(
    int boardHeight, int boardWidth, std::vector<int> board, int curr_player)
    : height_(boardHeight),
      width_(boardWidth),
      board_(board),
      player_(curr_player),
      action_space_(boardHeight * boardWidth)
{}

std::vector<bool> KinaRow::get_valid() const
{
  std::vector<bool> res(action_space_, false);
  for (int i = 0; i < action_space_; i++)
  {
    res[i] = board_[i] == 0;
  }
  return std::move(res);
}

int KinaRow::get_winner() const { return -1; }

void KinaRow::apply_action(int action)
{
  board_[action] = player_;
  player_ = -player_;
}
