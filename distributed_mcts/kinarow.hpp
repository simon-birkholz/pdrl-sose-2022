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
  bool is_possible(int action) const;

  int get_winner() const;
  int get_player() const { return player_; }

  std::pair<int, int> get_board_size() const
  {
    return std::make_pair(width_, height_);
  }

  void apply_action(int action);

  std::vector<int> get_board() const { return board_; }

  void print_board() const;

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