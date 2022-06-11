cmake_minimum_required(VERSION 3.20.0)

project(alphaZero)


set(CMAKE_PREFIX_PATH "C:/Program Files/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)



find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})




add_subdirectory(distributed_mcts)

