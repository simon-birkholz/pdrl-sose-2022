from tensorflow import keras
import tensorflow as tf

from mcts import MonteCarloTreeSearch

class AiPlayer:
    def __init__(self,game,m_name,sim_count=50):
        self.game = game
        self.model = keras.models.load_model(m_name)
        self.mcts = MonteCarloTreeSearch(self.game,self.model)
        self.sim_count = sim_count

    def get_action(self,obs):
        return self.mcts.search(obs,n_sim=self.sim_count)
