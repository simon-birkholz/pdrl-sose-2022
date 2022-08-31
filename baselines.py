"""
This file is used to define baseline players that are used to evaluate the performance of trained models.
"""

import numpy as np


class RandomPlayer:
    """
    An AI player that plays random (valid) actions.
    """
    def __init__(self, game):
        self.game = game

    def get_action(self, state):
        """
        Chooses a random action regarding the current board state.

        :param state: current game state
        :return: the chosen random action
        """
        valid = np.array(self.game.get_valid(state))
        valid = valid.flatten()
        poss_actions = np.where(valid > 0)[0]
        if len(poss_actions) > 0:
            return np.random.choice(poss_actions)
        raise ValueError("No possible moves. Game should be finished long ago")

    def inform(self, action):
        pass

    def reset(self):
        pass


class GreedyPlayer:
    """
    An AI player that greedily tries to build up its consecutive playing stones.
    """
    def __init__(self, game):
        self.game = game

    def get_action(self, state):
        """
        Finds the first owned playing stone, starting the search from left to right, top to bottom.
        Tries to connect to this stone by placing the next stone in the following positions, in the following order:
        left diagonally, vertically, right diagonally, horizontally.
        Falls back to the first free position if no suitable position is found.

        :param state: the current board state.
        :return: the action determined by greedy behavior.
        """
        valid = np.array(self.game.get_valid(state))
        valid = valid.flatten()

        print(state)
        s, p = state
        action = -1
        # top left and bottom right
        if action == -1:
            for iy, ix in np.ndindex(s.shape):
                if s[iy][ix] == p:
                    if 0 <= iy - 1 < s.shape[0] and 0 <= ix - 1 < s.shape[1] and s[iy - 1][ix - 1] == 0:
                        action = (iy - 1) * s.shape[0] + ix - 1
                        break
                    if 0 <= iy + 1 < s.shape[0] and 0 <= ix + 1 < s.shape[1] and s[iy + 1][ix + 1] == 0:
                        action = (iy + 1) * s.shape[0] + ix + 1
                        break

        # top mid and bottom mid
        if action == -1:
            for iy, ix in np.ndindex(s.shape):
                if s[iy][ix] == p:
                    if 0 <= iy - 1 < s.shape[0] and s[iy - 1][ix] == 0:
                        action = (iy - 1) * s.shape[0] + ix
                        break
                    if 0 <= iy + 1 < s.shape[0] and s[iy + 1][ix] == 0:
                        action = (iy + 1) * s.shape[0] + ix
                        break

        # top right and bottom left
        if action == -1:
            for iy, ix in np.ndindex(s.shape):
                if s[iy][ix] == p:
                    if 0 <= iy - 1 < s.shape[0] and 0 <= ix + 1 < s.shape[1] and s[iy - 1][ix + 1] == 0:
                        action = (iy - 1) * s.shape[0] + ix + 1
                        break
                    if 0 <= iy + 1 < s.shape[0] and 0 <= ix - 1 < s.shape[1] and s[iy + 1][ix - 1] == 0:
                        action = (iy + 1) * s.shape[0] + ix - 1
                        break

        # left and right
        if action == -1:
            for iy, ix in np.ndindex(s.shape):
                if s[iy][ix] == p:
                    if 0 <= ix - 1 < s.shape[1] and s[iy][ix - 1] == 0:
                        action = iy * s.shape[0] + ix - 1
                        break
                    if 0 <= ix + 1 < s.shape[1] and s[iy][ix + 1] == 0:
                        action = iy * s.shape[0] + ix + 1
                        break

        # take first empty field
        if action == -1:
            #   for iy, ix in np.ndindex(state[0].shape):
            #    if s[iy][ix] == 0:
            #          action = iy * s.shape[0] + ix
            poss_actions = np.where(valid > 0)[0]
            if len(poss_actions) > 0:
                action = np.random.choice(poss_actions)

        if action != -1:
            return action

        raise ValueError("No possible moves. Game should be finished long ago")

    def inform(self, action):
        pass

    def reset(self):
        pass
