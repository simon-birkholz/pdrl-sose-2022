# Project Deep Reinforcement Learning

An implementation for the AlphaZero Algorihm.

## ToDo

- [ ] working monte-carlo tree search
- [ ] different neural network architectures
- [ ] functional training loop for self play
- [ ] working implementation for tictactoe and k-in-a-row

## NiceToHave (not necessary for the completion of the project

- [ ] A (web) visualization which shows the current state of the game, as well as current policy and value estimation
- [ ] A distributed version of the monte-carlo as described in the papers

## Known Issues

- Ai agents often returns invalid move, and a random baseline has to be used as fallback

## Literature

```bibtex
@article{silver2016alphago,
  title={Mastering the game of Go with deep neural networks and tree search},
  author={Silver, David and Huang, Aja and Maddison, Chris J and Guez, Arthur and Sifre, Laurent and Van Den Driessche, George and Schrittwieser, Julian and Antonoglou, Ioannis and Panneershelvam, Veda and Lanctot, Marc and others},
  journal={nature},
  volume={529},
  number={7587},
  pages={484--489},
  year={2016},
  publisher={Nature Publishing Group}
}

@article{silver2017alphagozero,
  title={Mastering the game of go without human knowledge},
  author={Silver, David and Schrittwieser, Julian and Simonyan, Karen and Antonoglou, Ioannis and Huang, Aja and Guez, Arthur and Hubert, Thomas and Baker, Lucas and Lai, Matthew and Bolton, Adrian and others},
  journal={nature},
  volume={550},
  number={7676},
  pages={354--359},
  year={2017},
  publisher={Nature Publishing Group}
}

@article{silver2018alphazero,
  title={A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play},
  author={Silver, David and Hubert, Thomas and Schrittwieser, Julian and Antonoglou, Ioannis and Lai, Matthew and Guez, Arthur and Lanctot, Marc and Sifre, Laurent and Kumaran, Dharshan and Graepel, Thore and others},
  journal={Science},
  volume={362},
  number={6419},
  pages={1140--1144},
  year={2018},
  publisher={American Association for the Advancement of Science}
}

```