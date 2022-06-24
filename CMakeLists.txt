cmake_minimum_required(VERSION 3.20.0)

project(alphaZero)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_PREFIX_PATH "C:/Program Files/libtorch/share/cmake/Torch")

set(SWIG_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/swig)
find_package(Torch REQUIRED)

get_filename_component(TORCH_LIBS_PATH ${TORCH_LIBRARY} DIRECTORY CACHE)


find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs 3.9 REQUIRED)

message("found python at ${PYTHON_LIBRARIES}")

include_directories(${PYTHON_INCLUDE_PATH})

add_subdirectory(extern)


include(test)

add_subdirectory(distributed_mcts)

