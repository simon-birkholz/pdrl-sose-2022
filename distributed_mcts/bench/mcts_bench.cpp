// Simulation throughput of the search for several evaluator latencies, board
// sizes and thread counts.

#include "evaluator.hpp"
#include "kinarow.hpp"
#include "mcts.hpp"
#include "waiting_queue.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace
{
  using clock_type = std::chrono::steady_clock;

  // uniform priors and a deterministic pseudo random value per position, so
  // that the tree grows unevenly like with a real network
  Evaluator::return_t fake_evaluation(const KinaRow& game)
  {
    const auto board = game.get_board();
    const size_t hash = std::accumulate(
        board.begin(), board.end(), size_t{0}, [](size_t h, int cell) {
          return h * 31 + static_cast<size_t>(cell + 2);
        });
    const double value = static_cast<double>(hash % 1000) / 1000 - 0.5;
    return {std::vector<double>(board.size(), 1.0 / board.size()), value};
  }

  class InstantEvaluator : public Evaluator
  {
  public:
    std::future<return_t> evaluate(const KinaRow& game) override
    {
      std::promise<return_t> promise;
      promise.set_value(fake_evaluation(game));
      return promise.get_future();
    }
  };

  // answers all pending requests at once after a fixed latency, like a
  // batched forward pass on a GPU
  class BatchedEvaluator : public Evaluator
  {
  public:
    explicit BatchedEvaluator(std::chrono::microseconds latency)
        : latency_(latency), worker_([this] { run(); })
    {}

    ~BatchedEvaluator() override
    {
      abort_ = true;
      worker_.join();
    }

    std::future<return_t> evaluate(const KinaRow& game) override
    {
      std::promise<return_t> promise;
      auto future = promise.get_future();
      queue_.push({fake_evaluation(game), std::move(promise)});
      return future;
    }

  private:
    using task_t = std::pair<return_t, std::promise<return_t>>;

    void run()
    {
      while (!abort_)
      {
        std::vector<task_t> batch;
        for (auto task = queue_.try_pop(std::chrono::milliseconds(10)); task;
             task = queue_.try_pop(std::chrono::milliseconds(0)))
        {
          batch.push_back(std::move(*task));
        }
        if (!batch.empty())
        {
          std::this_thread::sleep_for(latency_);
        }
        for (auto& [result, promise] : batch)
        {
          promise.set_value(std::move(result));
        }
      }
    }

    std::chrono::microseconds latency_;
    common::waiting_queue<task_t> queue_;
    std::atomic<bool> abort_{false};
    std::thread worker_;
  };

  double sims_per_second(
      Evaluator& evaluator, int n, int k, unsigned sims, unsigned threads)
  {
    constexpr int repetitions = 3;
    MonteCarloTreeSearch search(&evaluator, {n, n}, k, sims, threads);
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < repetitions; i++)
    {
      search.reset();
      const auto start = clock_type::now();
      search.search({std::vector<int>(n * n, 0), 1});
      best = std::min(
          best,
          std::chrono::duration<double>(clock_type::now() - start).count());
    }
    return sims / best;
  }

  struct Scenario
  {
    std::string name;
    std::function<std::unique_ptr<Evaluator>()> make_evaluator;
    unsigned sims;
  };
} // namespace

int main()
{
  using std::chrono::microseconds;
  const std::vector<Scenario> scenarios = {
      {"instant", [] { return std::make_unique<InstantEvaluator>(); }, 40000},
      {"batched 100us",
       [] { return std::make_unique<BatchedEvaluator>(microseconds(100)); },
       8000},
      {"batched 1ms",
       [] { return std::make_unique<BatchedEvaluator>(microseconds(1000)); },
       1600},
  };
  const std::vector<std::pair<int, int>> boards = {{6, 4}, {15, 5}};
  const std::vector<unsigned> thread_counts = {1, 2, 4, 8, 16};

  std::printf(
      "%-14s %-6s %7s %14s\n", "evaluator", "board", "threads", "simulations");
  for (const auto& scenario : scenarios)
  {
    for (const auto& [n, k] : boards)
    {
      for (const unsigned threads : thread_counts)
      {
        const auto evaluator = scenario.make_evaluator();
        std::printf(
            "%-14s %2dx%-3d %7u %12.0f/s\n",
            scenario.name.c_str(),
            n,
            n,
            threads,
            sims_per_second(*evaluator, n, k, scenario.sims, threads));
      }
    }
  }
}
