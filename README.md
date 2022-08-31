# Project Deep Reinforcement Learning

**Project Deep Reinforcement Learning Summer Term 2022 at University of Ulm**

AlphaZero is an Reinforcement Learning algorithm that leverages Neural Networks, a Monte Carlo Tree Search and self-play to perform with superhuman performance in the classic games of Go, Chess and Shogi.
In this replication study, we focus on the different network architectures which can be employed the algorithm and compare their playing and learning performance against one another.

## ToDo

- [x] working monte-carlo tree search
- [x] different neural network architectures
- [x] functional training loop for self play
- [x] working implementation for tictactoe and k-in-a-row


## NiceToHave (not necessary for the completion of the project)

- [ ] A (web) visualization which shows the current state of the game, as well as current policy and value estimation
- [ ] A distributed version of the monte-carlo as described in the papers

## Known Issues

- ~~Ai agents often returns invalid move, and a random baseline has to be used as fallback~~
- Cpp wrapper loses nodes when executed multithreaded
- Based on the adaptation of the weights, sometimes the cpp wrapper can fail multiple times

## Getting Started

We use conda for managing our python dependencies. To create a new environment with the needed dependencies:

```
conda env create -n alpha-zero --file environment.yml
```

All needed parameters for training a model are supplied in a json configuration file. A example for this can be seen in `config.json`.
To start a new run of self-play learning or resume another run use `barracks.py`.`--out` to supply the name of the directory where the model weights are saved to.

```
python barracks.py config.json --out name_of_run
```

If you want to play against a network agent use `playing_demo.py`

## Compiling the SWIG-Wrapper

The whole self-learn pipeline is able to function without using the swig wrapper. The so performed training tends to be slower by a not so small margin.
For a better experience we suggest to use the C++/[SWIG](https://www.swig.org/) implementation of the Monte Carlo Tree Search.

For a successfull compilation of the wrapper [Libtorch](https://pytorch.org/cppdocs/installing.html) is needed.
We used Libtorch 1.11.0. It is important that the PyTorch Version and Libtorch version match.
Also both need to be using the same [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version.
This means the version installed manually on your system must match the version supplied in the `environment.yml`.

The wrapper is build using CMake. For Windows we recommend to use the CMake GUI to generate the Projects files for a Visual Studio Project.
Next compile the so generated Project in Release Configuration.
The Generated files are automatically copied to correct location so that they can be used by the Python Code.

The code can be found in `distributed_mcts`

The SWIG wrapper uses the following third party dependencies:
- [Catch2](https://github.com/catchorg/Catch2) licensed under the Boost Software License
- [SpdLog](https://github.com/gabime/spdlog) licensed under the MIT License
- [Threadpool Header-Only Library by Andreas Franz Borchert](https://github.com/afborchert/tpool) licensed under the MIT License


## Other files

- `neural_network.py`: Contains the code for the neural networks in Python
- `ai_player.py`: wrapper around the Neural Network and the MCTS. Can differentiate between the python model and the swig wrapper
- `baselines.py`: Contains the baselines for the project
- `colosseum.py`: Can duell two agents against each other and report how many games are won/lost
- `mcts`: Contains the python implementation for the mcts

Files used by use to generate the plots for the report
- `elo_generator.py`
- `generate_perf_diagram.py`
- `render_elo_diagram.py`
- `render_timing_diagram.py`
- `render_perf_diagram.py`
