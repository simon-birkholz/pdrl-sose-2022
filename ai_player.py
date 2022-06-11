from tensorflow import keras
import tensorflow as tf

from mcts import MonteCarloTreeSearch

from baselines import RandomPlayer

class AiPlayer:
    def __init__(self,game,m_name,sim_count=50):
        self.game = game
        self.model = keras.models.load_model(m_name)
        self.mcts = MonteCarloTreeSearch(self.game,self.model)
        self.sim_count = sim_count
        self.fallback_agent = RandomPlayer(game)

    def get_action(self,obs):
        action = self.mcts.search(obs,n_sim=self.sim_count)
        if not action > 0:
            action = self.fallback_agent.get_action(obs)
            print("Warning: Using fallback agent to sample random move")
        return action
