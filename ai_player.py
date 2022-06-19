from models.neural_network import NetworkWrapper

from swig.distributed_mcts import MonteCarloTreeSearch, SearchState
from swig.distributed_mcts import NeuralNetwork

import mcts as old_mcts

from baselines import RandomPlayer


class AiPlayer:
    def __init__(self,game,m_name,use_swig=False,sim_count=50,**kwargs):
        self.game = game
        self.use_swig = use_swig
        self.sim_count = sim_count
        if type(m_name) == str:
            #self.model = NeuralNetwork(m_name,False,10)
            self.model = NetworkWrapper(**kwargs)
            self.model.load_model('training','checkpoint')
        else:
            self.model = m_name
            #self.model = NetworkWrapper(**kwargs)
            #self.model.load_model('training', 'checkpoint')
        if self.use_swig:
            self.mcts = MonteCarloTreeSearch(self.model, self.sim_count)
        else:
            self.mcts = old_mcts.MonteCarloTreeSearch(self.game, self.model)
        self.fallback_agent = RandomPlayer(game)

    def get_action(self, obs):
        action = 0
        if self.use_swig:
            action = self.mcts.search(SearchState(obs[0].flatten().tolist(), obs[1]))
        else:
            action = self.mcts.search(obs, self.sim_count)
        if not action >= 0:
            action = self.fallback_agent.get_action(obs)
            print("Warning: Using fallback agent to sample random move")
        return action
