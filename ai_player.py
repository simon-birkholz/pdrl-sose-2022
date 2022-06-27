from models.neural_network import NetworkWrapper

from swig.distributed_mcts import MonteCarloTreeSearch, SearchState, PII
from swig.distributed_mcts import NeuralNetwork

import mcts as old_mcts
import numpy as np

from baselines import RandomPlayer


class AiPlayer:
    def __init__(self,game,m_name,board_size,k,use_swig=False,sim_count=50,thread_count=8,**kwargs):
        self.game = game
        self.use_swig = use_swig
        self.sim_count = sim_count
        self.board_size = board_size
        self.k = k
        if type(m_name) == str:
            #self.model = NeuralNetwork(m_name,False,10)
            self.model = NetworkWrapper(**kwargs)
            self.model.load_model('training','checkpoint')
        else:
            self.model = m_name
            #self.model = NetworkWrapper(**kwargs)
            #self.model.load_model('training', 'checkpoint')
        if self.use_swig:
            self.mcts = MonteCarloTreeSearch(self.model,PII(self.board_size[0],self.board_size[1]), self.k, sim_count,thread_count)
        else:
            self.mcts = old_mcts.MonteCarloTreeSearch(self.game, self.model)
        self.fallback_agent = RandomPlayer(game)

    def get_action(self, obs):
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

    def inform(self,action):
        if self.use_swig:
            self.mcts.apply_action(action)
