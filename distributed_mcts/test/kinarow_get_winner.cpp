#define ZERO_DEBUG

#include "kinarow.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
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