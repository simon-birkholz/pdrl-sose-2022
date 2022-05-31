from tensorflow import keras
import tensorflow as tf
import gym
import boardgame2
import numpy as np

from mcts_demo import MonteCarloTreeSearch
from tqdm import tqdm

tf.config.set_visible_devices([], 'GPU')

def transform_state(observation):
    board, player = observation
    res = np.zeros((3, 3, 2))
    nBoard = np.array(board * player)
    f1 = nBoard == 1
    f2 = nBoard == -1
    res[:, :, 0] = f1
    res[:, :, 1] = f2
    return np.asarray(res)


def get_board(observation):
    board, _ = observation
    res = np.zeros((3, 3, 2))
    f1 = board == 1
    f2 = board == -1
    res[:, :, 0] = f1
    res[:, :, 1] = f2
    return np.asarray(res)


def flip_colors(board):
    res = np.zeros((3, 3, 2))
    res[:, :, 0] = board[:, :, 1]
    res[:, :, 1] = board[:, :, 0]
    return res


def expand_action(action):
    v = np.zeros(9)
    v[action] = 1
    return v


m = keras.models.load_model('first-tic-tac-toe-agent')
env = gym.make('TicTacToe-v0')
mcts = MonteCarloTreeSearch(env, m)

def get_ai_move(model, obs):
    tObs = transform_state(obs)
    policy, value = model.predict(np.array([tObs]), verbose=0)

    valid = env.get_valid(obs)
    policy = policy[0].reshape((3, 3))

    # give invalid move a probability of zero
    policy[valid == 0] = 0

    policy = policy.flatten()

    return policy.argmax()


# play against a simple ai agent
observation = env.reset()
step_count = 0
while True:
    env.render()
    print("\n")
    step_count += 1

    selected_action = -1
    if step_count % 2 == 1:
        selected_action = int(input("Enter your move: "))-1
    else:
        #selected_action = get_ai_move(m, observation)
        selected_action = mcts.search(observation,50)

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

