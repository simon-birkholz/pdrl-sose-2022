import boardgame2
import gym
import numpy as np
from games.game import Game


# wrapper classes arond the kinarow class for the boardgame2 enviroment

def transform_state(observation):
    board, player = observation
    res = np.zeros((2, 3, 3))
    nBoard = np.array(board * player)
    f1 = nBoard == 1
    f2 = nBoard == -1
    res[0, :, :] = f1
    res[1, :, :] = f2
    return np.asarray(res)


def get_board(observation):
    board, _ = observation
    res = np.zeros((2, 3, 3))
    f1 = board == 1
    f2 = board == -1
    res[0, :, :] = f1
    res[1, :, :] = f2
    return np.asarray(res)


def flip_colors(board):
    res = np.zeros((2, 3, 3))
    res[0, :, :] = board[1, :, :]
    res[1, :, :] = board[0, :, :]
    return res


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
        return transform_state(state)

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
        return transform_state(state)


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
