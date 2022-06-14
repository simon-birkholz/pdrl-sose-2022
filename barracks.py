import threading

import gym
import boardgame2
from collections import deque
import time
import argparse
import os

import numpy as np
from sklearn.utils import shuffle

from swig.distributed_mcts import MonteCarloTreeSearch
from swig.distributed_mcts import NeuralNetwork

#from mcts import MonteCarloTreeSearch

from tqdm import tqdm
from baselines import RandomPlayer
from games.kinarow import TicTacToe

from colosseum import Colosseum
from ai_player import AiPlayer

import threading

from models.neural_network import NetworkWrapper

HYPER_PARAMS = {
    'lr' : 0.001,
    'l2' : 0.0001,
    'n_layers' : 4,
    'n_channels' : 64
}


def expand_action(action):
    v = np.zeros(9)
    v[action] = 1
    return v


class Worker(threading.Thread):
    global_episode = 0
    pBar = tqdm()
    save_lock = threading.Lock()

    def __init__(self, total_episodes, mcts_sims, global_model, result_queue):
        super(Worker, self).__init__()
        self.result_queue = result_queue
        self.global_model = global_model
        self.local_model = NeuralNetwork('./training/checkpoint',False,10)
        self.env = TicTacToe()
        self.fallback_agent = RandomPlayer(self.env)
        self.total_episodes = total_episodes
        self.mcts_sims = mcts_sims
        self.experience = []

    def run(self):

        while Worker.global_episode < self.total_episodes:
            observation = self.env.reset()
            observations = []
            mcts = MonteCarloTreeSearch(self.local_model, self.mcts_sims)
            #mcts = MonteCarloTreeSearch(self.env, self.global_model)
            step_count = 0
            while True:
                step_count += 1
                # todo return policy vector for learning
                selected_action = mcts.search(observation)
                if not selected_action > 0:
                    # mcts hasn't return valid value, use random baseline as fallback and report warning
                    selected_action = self.fallback_agent.get_action(observation)
                    print("Warning: Using fallback agent to sample random move")

                # todo hide away because game specific
                selected = np.unravel_index(selected_action, observation[0].shape)
                observations.append([observation, selected_action, None])
                observation, reward, done, info = self.env.step(selected)
                if done:
                    exp = [(self.env.to_nn_view(x[0]), expand_action(x[1]), (-1) ** (x[0][1] != reward)) for x in
                           observations]
                    self.experience.append(exp)

                    Worker.global_episode += 1
                    Worker.pBar.update(1)
                    # :/ sneaky bastard
                    break


def learn(n_iter=100, queue_length=10, threads=1, save_dir='tmp'):
    history_train_examples = deque([], maxlen=queue_length)
    global_model = NetworkWrapper(**HYPER_PARAMS)
    global_model.save_model('training', 'checkpoint')

    Worker.pBar = tqdm(total=n_iter)

    global_eps = int(n_iter / 100)

    for i in range(0, global_eps):

        Worker.global_episode = 0
        workers = [Worker(20,
                          100,
                          global_model,
                          history_train_examples) for i in range(threads)]

        for _, worker in enumerate(workers):
            worker.start()

        [w.join() for w in workers]
        all_exp = [w.experience for w in workers]
        all_exp = [x for xs in all_exp for x in xs]
        all_exp = [x for xs in all_exp for x in xs]

        t_boards, t_actions, t_values = zip(*all_exp)
        t_boards, t_actions, t_values = shuffle(np.array(t_boards), np.array(t_actions), np.array(t_values))

        global_model.save_model('training','checkpoint')

        model_candidate = NetworkWrapper(**HYPER_PARAMS)
        model_candidate.load_model('training','checkpoint')

        tData = list(zip(t_boards, t_actions, t_values))

        model_candidate.train(tData, 32,10)

        game = TicTacToe()
        c = Colosseum(game, AiPlayer(game, model_candidate, 100), AiPlayer(game, global_model, 100))
        print("Now Duel models to find out if better")
        n_wins, g_wins, draws = c.fight(32)
        print(f"New model won {n_wins} games, old model won {g_wins} games.")
        if n_wins > 16:
            print("Adapting weights")
            # adapt new model weights
            model_candidate.save_model('training', 'checkpoint')
            global_model.load_model('training', 'checkpoint')
            try:
                global_model.save_model(save_dir, 'checkpoint')
            except:
                print("Error on saving new model")

    global_model.save_model(save_dir,'final')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('iterations', type=int,
                        help='Number of iterations learned')
    parser.add_argument('--queue_size', type=int, default=10,
                        help='Size of the replay buffer queue')
    parser.add_argument('--out', type=str, help='name for saving the model', default='simple_agent')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to parallelize the learning process')

    args = parser.parse_args()

    learn(args.iterations, args.queue_size, args.threads, args.out)
