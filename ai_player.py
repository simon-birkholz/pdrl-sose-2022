"""
This file is used to wrap up a functioning AI player for a specific board game.
"""

from models.neural_network import NetworkWrapper

from swig.distributed_mcts import MonteCarloTreeSearch, SearchState, PII
from swig.distributed_mcts import NeuralNetwork

import mcts as old_mcts
import numpy as np

from baselines import RandomPlayer


class AiPlayer:
    def __init__(self, game, m_name, board_size, k, use_swig=False, sim_count=50, thread_count=8, **kwargs):
        """
        AI player consisting of the board game parameters, a neural network and the mcts.

        :param game: type of board game
        :param m_name: wrapper for the neural network
        :param board_size: size n represents a board size of n*n
        :param k: number of consecutive stones required to win
        :param use_swig: use swig implementation of mcts, otherwise the python version
        :param sim_count: depth of the mcts
        :param thread_count: how many threads swig sh
        :param kwargs: used to load a trained neural network
        """
        self.game = game
        self.use_swig = use_swig
        self.sim_count = sim_count
        self.board_size = board_size
        self.k = k
        if type(m_name) == str:
            # self.model = NeuralNetwork(m_name,False,10)
            self.model = NetworkWrapper(**kwargs)
            self.model.load_model('training', 'checkpoint')
        else:
            self.model = m_name
            # self.model = NetworkWrapper(**kwargs)
            # self.model.load_model('training', 'checkpoint')
        if self.use_swig:
            self.mcts = MonteCarloTreeSearch(self.model, PII(self.board_size[0], self.board_size[1]), self.k, sim_count,
                                             thread_count)
        else:
            self.mcts = old_mcts.MonteCarloTreeSearch(self.game, self.model)
        self.fallback_agent = RandomPlayer(game)

    def get_action(self, obs):
        """
        Determines the next action of the AI player.

        :param obs: currently observed board state
        :return: integer representing a position on the board
        """
        action = -1
        if self.use_swig:
            action_probs = self.mcts.search(SearchState(obs[0].flatten().tolist(), obs[1]))
            action_probs = list(action_probs)
            action = np.argmax(action_probs)
        else:
            action = self.mcts.search(obs, self.sim_count)
        if not action >= 0:
            action = self.fallback_agent.get_action(obs)
            print("Warning: Using fallback agent to sample random move")
        return action

    def inform(self, action):
        """
        Passes the chosen action to the mcts.

        :param action: integer representing a position on the board
        """
        if self.use_swig:
            self.mcts.apply_action(int(action))

    def reset(self):
        if self.use_swig:
            self.mcts.reset()
