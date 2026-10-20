cmake_minimum_required(VERSION 3.20.0)

project(alphaZero)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OFF builds only the search and its tests, without libtorch, SWIG and python
option(ZERO_BUILD_BINDINGS "Build the libtorch network and the python module" ON)

set(SWIG_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/swig)

if (ZERO_BUILD_BINDINGS)
  # point CMAKE_PREFIX_PATH at libtorch, e.g. torch.utils.cmake_prefix_path
  find_package(Torch REQUIRED)

  find_package(SWIG REQUIRED)
  include(${SWIG_USE_FILE})

  find_package(PythonLibs 3.9 REQUIRED)
  message("found python at ${PYTHON_LIBRARIES}")
  include_directories(${PYTHON_INCLUDE_PATH})
endif()

enable_testing()

add_subdirectory(extern)

include(test)

add_subdirectory(distributed_mcts)
