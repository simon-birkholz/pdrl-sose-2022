cmake_minimum_required(VERSION 3.20.0)

project(alphaZero)

#find_package(Torch REQUIRED)

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})




add_subdirectory(distributed_mcts)

