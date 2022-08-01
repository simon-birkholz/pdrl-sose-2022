#pragma once

#include <memory>
#include <mutex>

namespace common
{
  template <typename wrapped_t, typename mutex_t>
  class locked_ptr
  {
  public:
    using wrapped_ptr = std::shared_ptr<wrapped_t>;

    wrapped_ptr &operator->(void) { return wrapped_; }

    locked_ptr(locked_ptr const &) = delete;
    locked_ptr &operator=(locked_ptr const &) = delete;

    locked_ptr(std::mutex &m, wrapped_ptr w)
        : this(std::unique_lock<mutex_t>(m), w)
    {}
    locked_ptr(std::unique_lock<mutex_t> &&lock, wrapped_ptr w)
        : lock_(std::move(lock)), wrapped_(w)
    {}

  private:
    wrapped_ptr wrapped_;
    std::unique_lock<mutex_t> lock_;
  };
} // namespace common