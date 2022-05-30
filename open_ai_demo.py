import gym
import boardgame2

import time

import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, ReLU, Add, BatchNormalization, GlobalAveragePooling2D, Dense, MaxPooling2D

from mcts_demo import MonteCarloTreeSearch
from tqdm import tqdm

tf.config.set_visible_devices([], 'GPU')


def simpleCNN(n_filters=9):
    inputs = Input(shape=(3, 3, 2))

    x = Conv2D(filters=n_filters, kernel_size=3, padding='same', strides=(1, 1))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=n_filters, kernel_size=3, padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=n_filters, kernel_size=3, padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = GlobalAveragePooling2D()(x)
    policy = Dense(9, activation='softmax')(x)
    value = Dense(1, activation='tanh')(x)

    outputs = policy, value
    model = Model(inputs=inputs, outputs=outputs, name='Oscar', )
    model.summary()

    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'])

    return model


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


t = transform_state(([[1, -1, 1], [1, -1, 1], [1, -1, 1]], 1))
print(t)

env = gym.make('TicTacToe-v0')

print('observation space = {}'.format(env.observation_space))
print('action space = {}'.format(env.action_space))

m = simpleCNN(9)


def train_episode():
    """
    Executes one episode of the self-play algorithm
    :return:
    """
    observations = []
    observation = env.reset()
    mcts = MonteCarloTreeSearch(env, m)

    step_count = 0
    while True:
        #env.render()
        #print("\n")
        step_count += 1

        # todo return policy vector for learning
        selected_action = mcts.search(observation, 10)

        # todo hide away because game specific
        selected = np.unravel_index(selected_action, observation[0].shape)

        observations.append([observation, selected_action, None])

        observation, reward, done, info = env.step(selected)
        if done:
            #print("Done with game. Outcome: ", reward)

            return [(transform_state(x[0]), x[1], (-1) ** (x[2] != reward)) for x in observations]


def learn(n_iter=100):
    for i in tqdm(range(0,n_iter)):
        examples = train_episode()

        train_board = np.array([x[0] for x in examples])
        train_actions = np.array([expand_action(x[1]) for x in examples])
        train_values = np.array([x[2] for x in examples])

        m.fit(x=train_board, y=[train_actions, train_values], verbose=0)


learn(50)

env.render()
env.close()
m.save('first-tic-tac-toe-agent')
