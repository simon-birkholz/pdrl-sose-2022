#pragma once

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>


class log_impl
{
public:
  using sink_t = std::shared_ptr<spdlog::logger>;
  static log_impl& get_instance()
  {
    static log_impl instance;
    return instance;
  }

public:
  std::vector<sink_t> sinks_;

private:
  log_impl()
  {
    sink_t console = spdlog::stdout_color_mt("console");
    sinks_.push_back(console);
  }

public:
  log_impl(log_impl const&) = delete;
  void operator=(log_impl const&) = delete;
};

template <typename T>
void log_debug(const T& m)
{
  auto& logger = log_impl::get_instance();
  for (const auto& s : logger.sinks_)
  {
    s->debug(m);
  }
}

template <typename T>
void log_info(const T& m)
{
  auto& logger = log_impl::get_instance();
  for (const auto& s : logger.sinks_)
  {
    s->info(m);
  }
}

template <typename T>
void log_warning(const T& m)
{
  auto& logger = log_impl::get_instance();
  for (const auto& s : logger.sinks_)
  {
    s->warn(m);
  }
}

template <typename T>
void log_error(const T& m)
{
  auto& logger = log_impl::get_instance();
  for (const auto& s : logger.sinks_)
  {
    s->error(m);
  }
}

template <typename T>
void log_fatal(const T& m)
{
  auto& logger = log_impl::get_instance();
  for (const auto& s : logger.sinks_)
  {
    s->critical(m);
  }
}

#define POS \
  (std::string(__FUNCTION__) + std::string(":") + std::to_string(__LINE__))