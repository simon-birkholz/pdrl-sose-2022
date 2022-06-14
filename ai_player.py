
from swig.distributed_mcts import MonteCarloTreeSearch
from swig.distributed_mcts import NeuralNetwork

from baselines import RandomPlayer

class AiPlayer:
    def __init__(self,game,m_name,sim_count=50):
        self.game = game
        if m_name is str:
            self.model = NeuralNetwork(m_name,False,10)
        else:
            self.model = m_name
        self.mcts = MonteCarloTreeSearch(self.game,self.model)
        self.sim_count = sim_count
        self.fallback_agent = RandomPlayer(game)

    def get_action(self,obs):
        action = self.mcts.search(self.model,obs)
        if not action > 0:
            action = self.fallback_agent.get_action(obs)
            print("Warning: Using fallback agent to sample random move")
        return action
