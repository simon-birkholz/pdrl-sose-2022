import numpy as np


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def get_action(self, state):
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
    def __init__(self, game):
        self.game = game

    def get_action(self, state):
        valid = np.array(self.game.get_valid(state))
        valid = valid.flatten()

        s, p = state
        action = -1
        for iy, ix in np.ndindex(state[0].shape):
            if s[iy][ix] == p:
                # top left and bottom right
                if 0 <= iy - 1 < s.shape[0] and 0 <= ix - 1 < s.shape[1] and s[iy - 1][ix - 1] == 0:
                    action = iy * s.shape[0] + ix
                    break
                if 0 <= iy + 1 < s.shape[0] and 0 <= ix + 1 < s.shape[1] and s[iy + 1][ix + 1] == 0:
                    action = iy * s.shape[0] + ix
                    break

                # top mid and bottom mid
                if 0 <= iy - 1 < s.shape[0] and s[iy - 1][ix] == 0:
                    action = iy * s.shape[0] + ix
                    break
                if 0 <= iy + 1 < s.shape[0] and s[iy + 1][ix] == 0:
                    action = iy * s.shape[0] + ix
                    break

                # top right and bottom left
                if 0 <= iy - 1 < s.shape[0] and 0 <= ix + 1 < s.shape[1] and s[iy - 1][ix + 1] == 0:
                    action = iy * s.shape[0] + ix
                    break
                if 0 <= iy + 1 < s.shape[0] and 0 <= ix - 1 < s.shape[1] and s[iy + 1][ix - 1] == 0:
                    action = iy * s.shape[0] + ix
                    break

                # left and right
                if 0 <= ix - 1 < s.shape[1] and s[iy][ix - 1] == 0:
                    action = iy * s.shape[0] + ix
                    break
                if 0 <= ix + 1 < s.shape[1] and s[iy][ix + 1] == 0:
                    action = iy * s.shape[0] + ix
                    break

        # take first empty field
        if action == -1:
            for iy, ix in np.ndindex(state[0].shape):
                if s[iy][ix] == 0:
                    action = iy * s.shape[0] + ix

        if action != -1:
            return action
        raise ValueError("No possible moves. Game should be finished long ago")

    def inform(self, action):
        pass

    def reset(self):
        pass
