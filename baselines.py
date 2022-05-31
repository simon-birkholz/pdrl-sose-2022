
import numpy as np

class RandomPlayer:
    def __init__(self, game):
        self.game = game


    def get_action(self,state):
        valid = np.array(self.game.get_valid(state))
        valid = valid.flatten()
        poss_actions = np.where(valid > 0)[0]
        return np.random.choice(poss_actions)




