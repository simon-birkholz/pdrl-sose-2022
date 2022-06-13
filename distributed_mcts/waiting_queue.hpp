#pragma once


#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <type_traits>
namespace common
{
  template <class T, class Container = std::deque<T>>
  class waiting_queue
  {
  private:
    Container container;
    mutable std::mutex mutex;
    std::condition_variable cv;

  public:
    void clear()
    {
      std::lock_guard lock(mutex);
      container.clear();
    }

    T pop()
    {
      std::unique_lock lock(mutex);
      while (container.empty())
      {
        cv.wait(lock);
      }
      auto object = std::move(container.front());
      container.pop_front();
      return std::move(object);
    }

    void push(T object)
    {
      std::lock_guard lock(mutex);
      container.emplace_back(std::move(object));
      cv.notify_one();
    }

    std::optional<T> try_pop(std::chrono::milliseconds timeout)
    {
      std::unique_lock<std::mutex> lock(mutex);

      if (!cv.wait_for(lock, timeout, [this] { return !container.empty(); }))
        return {};

      auto object = std::move(container.front());
      container.pop_front();
      return std::move(object);
    }

    bool wait_for_update(std::chrono::milliseconds timeout)
    {
      std::unique_lock<std::mutex> lock(mutex);

      auto current_size = container.size();

      if (!cv.wait_for(lock, timeout, [this, current_size] {
            return current_size != container.size();
          }))
        return false;
      return true;
    }

    size_t size() const
    {
      std::unique_lock<std::mutex> lock(mutex);
      return container.size();
    }

    bool empty() const
    {
      std::unique_lock<std::mutex> lock(mutex);
      return container.empty();
    }

  public:
    waiting_queue() = default;
    waiting_queue(waiting_queue const&) = default;
    waiting_queue(waiting_queue&&) = default;
    waiting_queue& operator=(waiting_queue const&) = default;
    waiting_queue& operator=(waiting_queue&&) = default;
    ~waiting_queue(void) = default;
  };


} // namespace common