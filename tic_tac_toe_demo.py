
import gym
import boardgame2
import numpy as np

import os
os.add_dll_directory('C:\\Program Files\\libtorch\\lib')

from models.neural_network import NetworkWrapper

from tqdm import tqdm
from games.kinarow import TicTacToe
from ai_player import AiPlayer


HYPER_PARAMS = {
    'lr' : 0.001,
    'l2' : 0.0001,
    'n_layers' : 4,
    'n_channels' : 64
}

model = NetworkWrapper(**HYPER_PARAMS)
model.load_model('training','checkpoint')

env = TicTacToe()
agent = AiPlayer(env,model,False, 10)

# play against a simple ai agent
observation = env.reset()
step_count = 0
while True:
    env.render()
    print("\n")
    step_count += 1

    selected_action = -1
    if step_count % 2 == 1:
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
