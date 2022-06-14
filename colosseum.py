from random import randrange

import gym
import boardgame2
import numpy as np

from tqdm import tqdm
from baselines import RandomPlayer
from ai_player import AiPlayer
from games.kinarow import  TicTacToe


class Colosseum:
    def __init__(self, game, player1, player2):
        self.game = game
        self.player1 = player1
        self.player2 = player2

    def fight(self, game_count=100):
        p1_winCount = 0
        p2_winCount = 0
        drawCount = 0

        for i in tqdm(range(0, game_count)):
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
    env = TicTacToe()
    randomPlayer = RandomPlayer(env)
    ai_player = AiPlayer(env, 'longer_test')

    colosseum = Colosseum(env, randomPlayer, ai_player)

    rWins, aiWins, draws = colosseum.fight(50)
    print(f" Ai won {aiWins} games with {draws} draws")
