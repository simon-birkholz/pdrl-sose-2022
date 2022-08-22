import random

import gym
import boardgame2
import numpy as np

import os

os.add_dll_directory('C:\\Program Files\\libtorch\\lib')

from models.neural_network import NetworkWrapper

from baselines import GreedyPlayer

from swig.distributed_mcts import NeuralNetwork

from tqdm import tqdm
from games.kinarow import TicTacToe, KInARow
from ai_player import AiPlayer

HYPER_PARAMS = {
    'lr': 0.001,
    'l2': 0.0001,
    'n_layers': 4,
    'n_channels': 256,
    'n' : 3,
    'arch' : 'conv'
}

model = NetworkWrapper(**HYPER_PARAMS)
model.load_model('3x3_long_run_conv','final')
#swig_wrapper = NeuralNetwork('./training/checkpoint.pt', True, 10)

board_size = (3, 3)
k = 3

env = KInARow(board_size, k)

agent = AiPlayer(env, model, board_size, k, False, 100, 1)


# play against a simple ai agent
observation = env.reset()
step_count = 0
start_player = random.choice([0,1])
while True:
    env.render()
    print("\n")
    step_count += 1

    selected_action = -1
    if step_count % 2 == start_player:
        selected_action = int(input("Enter your move: ")) - 1
    else:
        # selected_action = get_ai_move(m, observation)
        selected_action = agent.get_action(observation)

    selected = np.unravel_index(selected_action, observation[0].shape)

    observation, reward, done, info = env.step(selected)
    if done:
        env.render()
        if reward == 1:
            print("You won")
        elif reward == -1:
            print("You lost")
        else:
            print("A Draw")
        break
