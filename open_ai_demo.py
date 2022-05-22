import gym
import boardgame2

import time

import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, ReLU, Add, BatchNormalization, GlobalAveragePooling2D, Dense, MaxPooling2D

from mcts_demo import Node, mcts

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
    res = np.zeros((3,3,2))
    nBoard = np.array(board * player)
    f1 = nBoard == 1
    f2 = nBoard == -1
    res[:,:,0] = f1
    res[:,:,1] = f2
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
    res[:,:,0] = board[:,:,1]
    res[:,:,1] = board[:,:,0]
    return res

def expand_action(action):
    v = np.zeros(9)
    v[action] = 1
    return v

t = transform_state(([[1,-1,1],[1,-1,1],[1,-1,1]],1))
print(t)

env = gym.make('TicTacToe-v0')

print('observation space = {}'.format(env.observation_space))
print('action space = {}'.format(env.action_space))

m = simpleCNN(9)

observation = env.reset()

obs_list = []
action_list = []

game_count = 0

while True:
    env.render()
    #valid = env.get_valid(observation)
    print("\n")
    #action = env.action_space.sample()

    tObs = transform_state(observation)

    action = mcts(observation,m,env)


    #action, value = m.predict(np.array([tObs]))
    #valid = env.get_valid(observation)

    #action = action.reshape((3,3))
    #action[valid == 0] = 0

    selected = np.unravel_index(action,observation[0].shape)
    action_list.append(action)

    observation, reward, done, info = env.step(selected)
    obs_list.append(observation)

    if done:
        print("Done with game. Reward: ",reward)
        states = np.array([transform_state(o) for o in obs_list])
        #train_boards = [board for board in states] + [flip_colors(board) for board in states]
        train_actions = np.array([expand_action(a) for a in action_list])
        train_values = np.array([o[1] * reward for o in obs_list])

        m.fit(x=states, y = [train_actions,train_values])

        observation = env.reset()
        game_count += 1
        if game_count > 100:
            break
        continue
env.render()
env.close()
m.save('first-tic-tac-toe-agent')