
from random import randrange
from tensorflow import keras
import tensorflow as tf
import gym
import boardgame2
import numpy as np

from mcts_demo import MonteCarloTreeSearch
from tqdm import tqdm
from baselines import RandomPlayer

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


class AiPlayer:
    def __init__(self,game):
        self.game = game
        self.model = keras.models.load_model('first-tic-tac-toe-agent')
        self.mcts = MonteCarloTreeSearch(self.game,self.model)
        self.sim_count = 50

    def get_action(self,obs):
        return self.mcts.search(obs,n_sim=self.sim_count)


class Colosseum:
    def __init__(self, game, player1, player2):
        self.game = game
        self.player1 = player1
        self.player2 = player2


    def fight(self,game_count = 100):
        p1_winCount = 0
        p2_winCount = 0
        drawCount = 0

        for i in tqdm(range(0,game_count)):
            observation = self.game.reset()
            step_count = 0
            sPlayer = randrange(2)
            while True:
                step_count += 1
                selected_action = -1
                if step_count % 2 == sPlayer:
                    selected_action = self.player1.get_action(observation)
                else:
                    selected_action = self.player2.get_action(observation)

                selected = np.unravel_index(selected_action, observation[0].shape)

                observation, reward, done, info = self.game.step(selected)
                if done:
                    if reward == 1:
                        if sPlayer == 1:
                            # player 1 is startPlayer
                            p1_winCount += 1
                        else:
                            p2_winCount += 1
                    elif reward == -1:
                        if sPlayer == 1:
                            # player 1 is startPlayer
                            p2_winCount += 1
                        else:
                            p1_winCount += 1
                    else:
                        drawCount += 1
                    break
        return p1_winCount, p2_winCount, drawCount


if __name__ == "__main__":
    env = gym.make('TicTacToe-v0')
    randomPlayer = RandomPlayer(env)
    ai_player = AiPlayer(env)

    colosseum = Colosseum(env,randomPlayer,ai_player)

    rWins, aiWins, draws = colosseum.fight(25)
    print(f" Ai won {aiWins} games with {draws} draws")
