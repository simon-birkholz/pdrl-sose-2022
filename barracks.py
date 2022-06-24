import threading

import gym
import boardgame2
from collections import deque
import time
import argparse
import os

import numpy as np
from sklearn.utils import shuffle

import os

os.add_dll_directory('C:\\Program Files\\libtorch\\lib')

from models.neural_network import NetworkWrapper

from swig.distributed_mcts import MonteCarloTreeSearch, SearchState, PII
from swig.distributed_mcts import NeuralNetwork

import mcts as old_mcts
# from mcts import MonteCarloTreeSearch

from tqdm import tqdm
from baselines import RandomPlayer
from games.kinarow import TicTacToe, KInARow

from colosseum import Colosseum
from ai_player import AiPlayer

import threading

HYPER_PARAMS = {
    'lr': 0.001,
    'l2': 0.0001,
    'n_layers': 4,
    'n_channels': 256,
    'arch' : 'conv',
    'n' : 8
}


def expand_action(action,action_space):
    v = np.zeros(action_space)
    v[action] = 1
    return v


class Worker(threading.Thread):
    global_episode = 0
    pBar = tqdm()
    save_lock = threading.Lock()

    def __init__(self, total_episodes, mcts_sims, global_model, result_queue, use_swig=False):
        super(Worker, self).__init__()
        self.result_queue = result_queue
        self.global_model = global_model

        self.total_episodes = total_episodes
        self.mcts_sims = mcts_sims
        self.experience = []
        self.use_swig = use_swig
        self.board_shape = (8, 8)
        self.k = 4
        self.env = KInARow(self.board_shape,self.k)
        self.fallback_agent = RandomPlayer(self.env)
        self.action_space = 64

    def run(self):

        while Worker.global_episode < self.total_episodes:
            observation = self.env.reset()
            observations = []
            mcts = None
            if self.use_swig:
                mcts = MonteCarloTreeSearch(self.global_model, PII(self.board_shape[0],self.board_shape[1]), self.k, self.mcts_sims,8)
            else:
                mcts = old_mcts.MonteCarloTreeSearch(self.env, self.global_model)
            step_count = 0
            while True:
                step_count += 1
                # todo return policy vector for learning
                selected_action = 0
                if self.use_swig:
                    action_probs = mcts.search(SearchState(observation[0].flatten().tolist(), observation[1]))
                    action_probs = list(action_probs)
                    selected_action = np.argmax(action_probs)

                    # we need to propagate this the cpp tree
                    mcts.apply_action(int(selected_action))

                else:
                    selected_action = mcts.search(observation, self.mcts_sims)
                if not selected_action >= 0:
                    # mcts hasn't return valid value, use random baseline as fallback and report warning
                    selected_action = self.fallback_agent.get_action(observation)
                    print(f"Warning: Using fallback agent to sample random move | critical move: {selected_action}")

                # todo hide away because game specific

                valid = self.env.get_valid(observation)
                valid = np.array(valid).flatten()
                if not valid[selected_action]:
                    raise ValueError("Can't execute move, already executed")

                selected = np.unravel_index(selected_action, observation[0].shape)
                observations.append([observation, selected_action, None])
                observation, reward, done, info = self.env.step(selected)
                if done:
                    exp = [(self.env.to_nn_view(x[0]), expand_action(x[1], self.action_space), (-1) ** (x[0][1] != reward)) for x in
                           observations]
                    self.experience.append(exp)

                    Worker.global_episode += 1
                    Worker.pBar.update(1)
                    # :/ sneaky bastard
                    break


def learn(n_iter=100, queue_length=10, threads=1, save_dir='tmp'):
    use_swig = True
    libtorch_use_gpu=False

    history_train_examples = deque([], maxlen=queue_length)
    global_model = NetworkWrapper(**HYPER_PARAMS, train_use_gpu=True, libtorch_use_gpu=libtorch_use_gpu)
    global_model.save_model('training', 'checkpoint')

    swig_wrapper = None
    if use_swig:
        swig_wrapper = NeuralNetwork('./training/checkpoint.pt', libtorch_use_gpu, 10)

    Worker.pBar = tqdm(total=n_iter)

    its_per_ep = 20

    global_eps = int(n_iter / its_per_ep)

    for i in range(0, global_eps):

        Worker.global_episode = 0

        workers = [Worker(its_per_ep,
                          100,
                          swig_wrapper if use_swig else global_model,
                          history_train_examples, use_swig) for i in range(threads)]

        for _, worker in enumerate(workers):
            worker.start()

        [w.join() for w in workers]
        all_exp = [w.experience for w in workers]
        all_exp = [x for xs in all_exp for x in xs]
        all_exp = [x for xs in all_exp for x in xs]

        if len(all_exp) == 0:
            # something went wrong
            raise RuntimeError("Something probably went wrong")

        t_boards, t_actions, t_values = zip(*all_exp)
        t_boards, t_actions, t_values = shuffle(np.array(t_boards), np.array(t_actions), np.array(t_values))

        global_model.save_model('training', 'checkpoint')

        model_candidate = NetworkWrapper(**HYPER_PARAMS)
        model_candidate.load_model('training', 'checkpoint')

        tData = list(zip(t_boards, t_actions, t_values))

        model_candidate.train(tData, 32, 10)

        game = KInARow((8,8),4)
        c = Colosseum(game, AiPlayer(game, model_candidate, False, 100), AiPlayer(game, global_model, False, 100))
        print("Now Duel models to find out if better")
        n_wins, g_wins, draws = c.fight(32)
        print(f"New model won {n_wins} games, old model won {g_wins} games.")
        if n_wins  > 16:
            print("Adapting weights")
            # adapt new model weights
            model_candidate.save_model('training', 'checkpoint')
            global_model.load_model('training', 'checkpoint')
            if use_swig:
                swig_wrapper.reload_weights()
            try:
                global_model.save_model(save_dir, 'checkpoint')
            except:
                print("Error on saving new model")

    global_model.save_model(save_dir, 'final')


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
