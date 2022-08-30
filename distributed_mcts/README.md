# Distributed MCTS

This folder contains the code for our multithreaded implementation of the MCTS in C++/SWIG.
The name is a little bit misleading because it is only multithreaded and not distributed.

## Getting started

For a successfull compilation of the wrapper [Libtorch](https://pytorch.org/cppdocs/installing.html) is needed.
We used Libtorch 1.11.0. It is important that the PyTorch Version and Libtorch version match.
Also both need to be using the same [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version.
This means the version installed manually on your system must match the version supplied in the `environment.yml`.

The wrapper is build using CMake. For Windows we recommend to use the CMake GUI to generate the Projects files for a Visual Studio Project.
Next compile the so generated Project in Release Configuration.
The Generated files are automatically copied to correct location so that they can be used by the Python Code.

The SWIG wrapper uses the following third party dependencies:
- [Catch2](https://github.com/catchorg/Catch2) licensed under the Boost Software License
- [SpdLog](https://github.com/gabime/spdlog) licensed under the MIT License
- [Threadpool Header-Only Library by Andreas Franz Borchert](https://github.com/afborchert/tpool) licensed under the MIT License

**The build process has only been tested on windows**

## Files

- `waiting_queue.hpp`: A queue which support threadsafe push/pop as well as conditional variables to avoid active waiting times.
- `threadpool.hpp`: Header-Only Threadpool implementation by Andreas Franz Borchert
- `neural_network.hpp` and `neural_network.cpp`: The wrapper around the libtorch library. Allows threadsafe and async querying of the neural network
- `mcts.i`: The definition of the swig wrapper
- `mcts.hpp` and `mcts.cpp`: The implementation of the monte carlo tree search
- `log.hpp`: wrapper around spdlog library
- `locked_ptr.hpp`: a smart pointer which also wraps a lock to keep the object locked (not used in the implementation)
- `kinarow.hpp` and `kinarow.cpp`: Implementation of the game wrapper
- `test` : Some unit test to ensure functionality and find bugs



## It can't be that hard? or?

Some thread on the topic

- https://stackoverflow.com/questions/52584142/mcts-tree-parallelization-in-python-possible
- https://github.com/lightvector/GoNN/issues/7