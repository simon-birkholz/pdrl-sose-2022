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
from games.kinarow import KInARow
from ai_player import AiPlayer

HYPER_PARAMS = {
    'lr': 0.001,
    'l2': 0.0001,
    'n_layers': 6,
    'n_channels': 256,
    'n': 6,
    'arch': 'conv'
}

# model = NetworkWrapper(**HYPER_PARAMS)
#model.load_model('6x4_co', 'final')
swig_wrapper = NeuralNetwork('./6x4_resnet_lr0.001/checkpoint-ep5000.pt', True, 10)

board_size = (6, 6)
k = 4

env = KInARow(board_size, k)

agent = AiPlayer(env, swig_wrapper, board_size, k, True, 100, 1)

# play against a simple ai agent
observation = env.reset()
step_count = 0
start_player = random.choice([0, 1])
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

    agent.inform(selected_action)
    observation, reward, done, info = env.step(selected)
    if done:
        env.render()
        if reward == 1 and start_player == 1 or reward == -1 and start_player == 0:
            print("You won")
        elif reward == -1 and start_player == 1 or reward == 1 and start_player == 0:
            print("You lost")
        else:
            print("A Draw")
        break
