# Distributed MCTS

This folder contains the code for our multithreaded implementation of the MCTS in C++/SWIG.
The name is a little bit misleading because it is only multithreaded and not distributed.

## Getting started

The wrapper needs a C++20 compiler, [SWIG](https://www.swig.org/) and [Libtorch](https://pytorch.org/cppdocs/installing.html).
The libtorch version must match the PyTorch version that exports the models.
The simplest way to guarantee that is to build against the libtorch bundled with the installed PyTorch, from the repository root:

```
cmake -S . -B build -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build --config Release
```

The generated module is copied to `swig/` so that it can be used by the Python code.
On Windows the module finds the libtorch DLLs as long as `torch` is imported before it.

The search itself does not depend on libtorch.
`-DZERO_BUILD_BINDINGS=OFF` builds only the search and its tests, which is the configuration to run under ThreadSanitizer:

```
cmake -S . -B build-tsan -DZERO_BUILD_BINDINGS=OFF -DCMAKE_CXX_FLAGS=-fsanitize=thread -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread
cmake --build build-tsan && ctest --test-dir build-tsan
```

The wrapper uses the following third party dependencies:
- [Catch2](https://github.com/catchorg/Catch2) licensed under the Boost Software License
- [Threadpool Header-Only Library by Andreas Franz Borchert](https://github.com/afborchert/tpool) licensed under the MIT License

## Files

- `evaluator.hpp`: Interface the search uses to evaluate positions
- `mcts.hpp` and `mcts.cpp`: The implementation of the monte carlo tree search
- `neural_network.hpp` and `neural_network.cpp`: Evaluator backed by a TorchScript model, batches concurrent requests into one forward pass
- `kinarow.hpp` and `kinarow.cpp`: Implementation of the game wrapper
- `mcts.i`: The definition of the swig wrapper
- `waiting_queue.hpp`: A queue which support threadsafe push/pop as well as conditional variables to avoid active waiting times.
- `threadpool.hpp`: Header-Only Threadpool implementation by Andreas Franz Borchert
- `test`: Unit tests, they use a fake evaluator and need no model

## Threading model

All simulations share one tree guarded by a single mutex.
Selection and backup hold it, the network evaluation does not.
Virtual loss makes concurrent simulations pick different paths, and a leaf that is being evaluated is marked as such: other simulations reaching it wait for its expansion instead of evaluating it a second time.
This guarantees one expanded node per simulation.

## It can't be that hard? or?

Some thread on the topic

- https://stackoverflow.com/questions/52584142/mcts-tree-parallelization-in-python-possible
- https://github.com/lightvector/GoNN/issues/7
