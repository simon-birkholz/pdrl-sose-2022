#pragma once

#include <vector>

class KinaRow
{
public:
  KinaRow(
      int boardHeight,
      int boardWidth,
      int k,
      std::vector<int> board,
      int curr_player);

public:
  std::vector<bool> get_valid() const;

  int get_winner() const;
  int get_player() const { return player_; }

  void apply_action(int action);

  std::vector<int> get_board() const { return board_; }

private:
  bool check_won_player(const int player) const;

private:
  const int height_;
  const int width_;
  const int k_;
  const int action_space_;
  std::vector<int> board_;
  int player_;
};