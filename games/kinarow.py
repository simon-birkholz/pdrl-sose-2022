"""
This file is used to wrap the game k-in-a-row from the boardgame2 environment.
"""

import boardgame2
import gym
import numpy as np
from games.game import Game

"""
Wrapper classes around the k-in-a-row class for the boardgame2 environment
"""


def transform_state(observation, n):
    board, player = observation
    res = np.zeros((2, n, n))
    nBoard = np.array(board * player)
    f1 = nBoard == 1
    f2 = nBoard == -1
    res[0, :, :] = f1
    res[1, :, :] = f2
    return np.asarray(res)


def get_board(observation, n):
    board, _ = observation
    res = np.zeros((2, n, n))
    f1 = board == 1
    f2 = board == -1
    res[0, :, :] = f1
    res[1, :, :] = f2
    return np.asarray(res)


def flip_colors(board, n):
    res = np.zeros((2, n, n))
    res[0, :, :] = board[1, :, :]
    res[1, :, :] = board[0, :, :]
    return res


def rotate_action(action, n):
    # rotates the action id clockwise by 90
    o_y, o_x = np.unravel_index(action, (n, n))
    n_x = (n - 1) - o_y
    n_y = o_x
    return n_y * n + n_x


def mirror_action(action, n):
    # mirror action
    o_y, o_x = np.unravel_index(action, (n, n))
    n_x = (n - 1) - o_x
    n_y = o_y
    return n_y * n + n_x


def rotate_all(board, action, n):
    boards = [board]
    actions = [action]

    for i in range(0, 3):
        board = np.rot90(board)
        action = rotate_action(action, n)
        boards.append(board)
        actions.append(action)

    return boards, actions


def generate_symmetries(obs, action):
    board, player = obs
    n = board.shape[0]
    # flipped
    flipped = np.fliplr(board)
    flipped_a = mirror_action(action, n)

    b1, a1 = rotate_all(board, action, n)
    b2, a2 = rotate_all(flipped, flipped_a, n)
    return list(zip(b1 + b2, [player for _ in range(0, 8)])), a1 + a2


class TicTacToe(Game):

    def __init__(self, silent=True):
        self.env = gym.make('TicTacToe-v0')

        if not silent:
            print('observation space = {}'.format(self.env.observation_space))
            print('action space = {}'.format(self.env.action_space))

    def reset(self):
        return self.env.reset()

    def get_valid(self, state):
        return self.env.get_valid(state)

    def get_winner(self, state):
        return self.env.get_winner(state)

    def get_next_state(self, state, action):
        return self.env.get_next_state(state, action)

    def next_step(self, state, action):
        return self.env.next_step(state, action)

    def step(self, action):
        return self.env.step(action)

    def to_nn_view(self, state):
        # transforms the state of the game if the player happens to be -1
        return transform_state(state, 3)

    def correct_policy(self, policy, state):
        # sets the policy values of non valid moves to zero
        valid = self.get_valid(state)

        policy = policy.reshape((3, 3))

        # give invalid move a probability of zero
        policy[valid == 0] = 0
        policy = policy.flatten()

        return policy

    def render(self):
        self.env.render()


class KInARow(Game):

    def __init__(self, board_shape, k, silent=True):
        self.env = gym.make('KInARow-v0', board_shape=board_shape, target_length=k)
        self.board_shape = board_shape
        self.k = k

        if not silent:
            print('observation space = {}'.format(self.env.observation_space))
            print('action space = {}'.format(self.env.action_space))

    def reset(self):
        return self.env.reset()

    def get_valid(self, state):
        return self.env.get_valid(state)

    def get_winner(self, state):
        return self.env.get_winner(state)

    def get_next_state(self, state, action):
        return self.env.get_next_state(state, action)

    def next_step(self, state, action):
        return self.env.next_step(state, action)

    def step(self, action):
        return self.env.step(action)

    def to_nn_view(self, state):
        # transforms the state of the game if the player happens to be -1
        return transform_state(state, self.board_shape[0])

    def correct_policy(self, policy, state):
        # sets the policy values of non valid moves to zero
        valid = self.get_valid(state)

        policy = policy.reshape(self.board_shape)

        # give invalid move a probability of zero
        policy[valid == 0] = 0
        policy = policy.flatten()

        return policy

    def render(self):
        self.env.render()
