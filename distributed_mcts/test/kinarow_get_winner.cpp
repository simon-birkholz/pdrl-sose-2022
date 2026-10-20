#include "kinarow.hpp"

#include <vector>


#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

TEST_CASE("Testing 3x3 Kinarow", "zero::kinarow")
{
  // 3x3 test cases
  std::vector<int> test1_board = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  KinaRow test1(3, 3, 3, test1_board, 1);

  REQUIRE(test1.get_winner() == 1);

  std::vector<int> test2_board = {0, 0, -1, 0, -1, 0, -1, 0, 1};

  KinaRow test2(3, 3, 3, test2_board, 1);

  REQUIRE(test2.get_winner() == -1);

  std::vector<int> test3_board = {0, 0, -1, 0, -1, 0, 0, 0, 1};

  KinaRow test3(3, 3, 3, test3_board, 1);

  REQUIRE(test3.get_winner() == -2);

  std::vector<int> test4_board = {1, 1, 1, 0, -1, 0, 0, 0, 1};

  KinaRow test4(3, 3, 3, test4_board, 1);

  REQUIRE(test4.get_winner() == 1);
}

TEST_CASE("Testing 4x4 Kinarow", "zero::kinarow")
{
  // 4x4 test cases
  std::vector<int> test1_board = {
      1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};

  KinaRow test1(4, 4, 4, test1_board, 1);

  REQUIRE(test1.get_winner() == 1);

  std::vector<int> test2_board = {
      1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  KinaRow test2(4, 4, 4, test2_board, 1);

  REQUIRE(test2.get_winner() == 1);

  std::vector<int> test3_board = {
      1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

  KinaRow test3(4, 4, 4, test3_board, 1);

  REQUIRE(test3.get_winner() == 1);

  std::vector<int> test4_board = {
      1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};

  KinaRow test4(4, 4, 4, test4_board, 1);

  REQUIRE(test4.get_winner() == -2);
}

TEST_CASE("Testing 4x4 Kinarow with k=3", "zero::kinarow")
{
  // 4x4x3 test cases
  std::vector<int> test1_board = {
      1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};

  KinaRow test1(4, 4, 3, test1_board, 1);

  REQUIRE(test1.get_winner() == 1);

    std::vector<int> test1a_board = {
      1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  KinaRow test1a(4, 4, 3, test1a_board, 1);

  REQUIRE(test1a.get_winner() == -2);

  std::vector<int> test2_board = {
      1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  KinaRow test2(4, 4, 3, test2_board, 1);

  REQUIRE(test2.get_winner() == 1);

   std::vector<int> test2a_board = {
      1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  KinaRow test2a(4, 4, 3, test2a_board, 1);

  REQUIRE(test2a.get_winner() == -2);

  std::vector<int> test3_board = {
      1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

  KinaRow test3(4, 4, 3, test3_board, 1);

  REQUIRE(test3.get_winner() == -2);

  std::vector<int> test4_board = {
      1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};

  KinaRow test4(4, 4, 3, test4_board, 1);

  REQUIRE(test4.get_winner() == 1);
}

TEST_CASE("Lines do not wrap around the board edge", "zero::kinarow")
{
  // (y, x) = (0,2) -> (0,3) wraps to (1,0)
  std::vector<int> horizontal = {
      0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  REQUIRE(KinaRow(4, 4, 3, horizontal, 1).get_winner() == KinaRow::ongoing);

  // (2,2) -> (1,3) wraps to (1,0)
  std::vector<int> anti_diagonal = {
      0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0};
  REQUIRE(KinaRow(4, 4, 3, anti_diagonal, 1).get_winner() == KinaRow::ongoing);

  // (0,3) wraps to (2,0) -> (3,1)
  std::vector<int> diagonal = {
      0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0};
  REQUIRE(KinaRow(4, 4, 3, diagonal, 1).get_winner() == KinaRow::ongoing);
}

TEST_CASE("Full board without line is a draw", "zero::kinarow")
{
  std::vector<int> board = {1, -1, 1, 1, -1, -1, -1, 1, 1};
  REQUIRE(KinaRow(3, 3, 3, board, 1).get_winner() == 0);
}
