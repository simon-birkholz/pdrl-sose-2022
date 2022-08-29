"""
This file is used to match up different models and baselines against each other to evaluate their performance.
"""

from random import randrange

import gym
import boardgame2
import numpy as np

from tqdm import tqdm
from baselines import RandomPlayer
from ai_player import AiPlayer
from games.kinarow import TicTacToe, KInARow

from swig.distributed_mcts import NeuralNetwork

HYPER_PARAMS = {
    'lr': 0.001,
    'l2': 0.0001,
    'n_layers': 4,
    'n_channels': 64
}


class Colosseum:
    """
    This class is used to match up two different players.
    Should a player not be able to make a move, a fallback agent (by default a random player) is used to substitute this move.
    """
    def __init__(self, game, player1, player2):
        self.game = game
        self.player1 = player1
        self.player2 = player2
        self.fallback_agent = RandomPlayer(self.game)

    def fight(self, game_count=100):
        """
        Method used to start the match between the two players.

        :param game_count: how many games the players play against each other
        :return: number of player 1 wins, number of player 2 wins, number of draws
        """
        p1_winCount = 0
        p2_winCount = 0
        drawCount = 0

        for i in tqdm(range(0, game_count)) if game_count > 1 else range(0, game_count):
            observation = self.game.reset()
            self.player1.reset()
            self.player2.reset()
            self.fallback_agent.reset()
            step_count = 0
            sPlayer = randrange(2)
            while True:
                step_count += 1
                selected_action = -1
                if step_count % 2 == sPlayer:
                    selected_action = self.player1.get_action(observation)
                else:
                    selected_action = self.player2.get_action(observation)

                # include fallback
                valid = self.game.get_valid(observation)
                valid = np.array(valid).flatten()
                if not selected_action >= 0 or not valid[selected_action]:
                    selected_action = self.fallback_agent.get_action(observation)
                    print(
                        f"Warning: Using fallback agent to sample random move | critical move: {selected_action} (step no. {step_count})")

                # first move random for better exploration:
                if step_count == 1:
                    selected_action = self.fallback_agent.get_action(observation)

                selected = np.unravel_index(selected_action, observation[0].shape)

                self.player1.inform(selected_action)
                self.player2.inform(selected_action)

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
    env = KInARow((8, 8), 4)
    randomPlayer = RandomPlayer(env)
    arena_wrapper = NeuralNetwork('./training/arena-new.pt', False, 10)
    ai_player = AiPlayer(env, arena_wrapper, (8, 8), 4, True, 100, 1)

    colosseum = Colosseum(env, randomPlayer, ai_player)

    rWins, aiWins, draws = colosseum.fight(50)
    print(f" Ai won {aiWins} games with {draws} draws")
