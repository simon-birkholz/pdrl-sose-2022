"""
This file is used to configure and train neural networks.
"""

import threading

import gym
import boardgame2
from collections import deque
import time
import argparse
import os
import json

import glob
import re

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
from games.kinarow import TicTacToe, KInARow, generate_symmetries

from colosseum import Colosseum
from ai_player import AiPlayer

import threading

try:
    import wandb

    USE_WANDB = True
except ImportError:
    USE_WANDB = False


def expand_action(action, action_space):
    """
    Apply an action to the action_space.

    :param action: integer representing an action
    :param action_space: array keeping track of used actions
    :return: action_space with action applied
    """
    v = np.zeros(action_space)
    v[action] = 1
    return v


class Worker(threading.Thread):
    """
    Helper class to distribute the learning process.
    """
    global_episode = 0
    pBar = tqdm()
    save_lock = threading.Lock()

    def __init__(self, total_episodes, mcts_sims, global_model, board_size, k, threads=1, use_swig=False):
        super(Worker, self).__init__()

        self.global_model = global_model

        self.threads = threads
        self.total_episodes = total_episodes
        self.mcts_sims = mcts_sims
        self.experience = []
        self.use_swig = use_swig
        self.board_shape = board_size
        self.k = k
        self.env = KInARow(self.board_shape, self.k)
        self.fallback_agent = RandomPlayer(self.env)
        self.action_space = board_size[0] * board_size[1]
        self.use_symmetries = True
        self.random_start = 2  # make some random starting moves to improve learning

    def run(self):

        while Worker.global_episode < self.total_episodes:
            observation = self.env.reset()
            observations = []
            mcts = None
            if self.use_swig:
                mcts = MonteCarloTreeSearch(self.global_model, PII(self.board_shape[0], self.board_shape[1]), self.k,
                                            self.mcts_sims, self.threads)
            else:
                mcts = old_mcts.MonteCarloTreeSearch(self.env, self.global_model)
            step_count = 0
            start_time = time.perf_counter()
            while True:
                step_count += 1
                # todo return policy vector for learning
                # print("State in python")
                # self.env.render()
                # print("State in cpp")
                selected_action = 0
                if step_count <= self.random_start:
                    # sample random action
                    selected_action = self.fallback_agent.get_action(observation)
                elif self.use_swig:
                    action_probs = mcts.search(SearchState(observation[0].flatten().tolist(), observation[1]))
                    # print(f"Node count: {mcts.get_node_count()} Max depth: {mcts.get_max_depth()}")
                    # action_probs = list(action_probs)
                    selected_action = np.argmax(action_probs)

                    # we need to propagate this the cpp tree
                    # mcts.apply_action(int(selected_action))

                else:
                    selected_action = mcts.search(observation, self.mcts_sims)
                if not selected_action >= 0:
                    # mcts hasn't return valid value, use random baseline as fallback and report warning
                    selected_action = self.fallback_agent.get_action(observation)
                    print(
                        f"Warning: Using fallback agent to sample random move | critical move: {selected_action} (step no. {step_count})")
                    # self.env.render()

                valid = self.env.get_valid(observation)
                valid = np.array(valid).flatten()
                if not valid[selected_action]:
                    # raise ValueError("Can't execute move, already executed")
                    # mcts has returned garbage value, use random baseline as fallback and report warning
                    selected_action = self.fallback_agent.get_action(observation)
                    print(
                        f"Warning: Using fallback agent to sample random move | critical move: {selected_action} (step no. {step_count})")
                    # self.env.render()
                    if self.use_swig:
                        mcts.reset()

                if self.use_swig:
                    mcts.apply_action(int(selected_action))

                selected = np.unravel_index(selected_action, observation[0].shape)
                observations.append([observation, selected_action, None])
                observation, reward, done, info = self.env.step(selected)
                if done:
                    all_obs = []
                    if self.use_symmetries:
                        for state, action, _ in observations:
                            states, actions = generate_symmetries(state, action)
                            all_obs += [[s, a, None] for s, a in list(zip(states, actions))]
                    else:
                        all_obs = observations

                    exp = [
                        (self.env.to_nn_view(x[0]), expand_action(x[1], self.action_space), (-1) ** (x[0][1] != reward))
                        for x in
                        all_obs]
                    self.experience.append(exp)
                    # self.env.render()
                    Worker.global_episode += 1
                    Worker.pBar.update(1)
                    if self.use_swig:
                        mcts.reset()

                    # log episode length to wandb
                    if USE_WANDB:
                        elapsed_time = time.perf_counter() - start_time
                        wandb.log({'episode_length': step_count, 'elapsed_time': elapsed_time})
                    # :/ sneaky bastard
                    break


def learn(allparams, iterations, threads, save_dir, use_swig, libtorch_use_gpu, python_use_gpu, its_per_ep, sim_count,
          do_duelling,
          duelling_count, training_epochs, training_batchsize, **config):
    """
    Method the model uses to learn and save progress.

    :param allparams: extra copy of all parameters
    :param iterations: number of training iterations
    :param threads: number of threads to be used
    :param save_dir: direction for the save files
    :param use_swig: use swig implementation of mcts, else the python version
    :param its_per_ep: size of an episode
    :param sim_count: how many simulations per mcts
    :param do_duelling: whether the model learns by duelling previous iterations or continuously
    :param duelling_count: how many duels the model fights
    :param config: parameters of the config
    """
    # initialize wandb
    if USE_WANDB:
        run = wandb.init(project='alpha-zero', entity='pdrl-alpha-zero-2022', config=dict(params=allparams))

    last_episode = max([int(ep) for f in glob.glob(save_dir + '/*.pt') for ep in re.findall(r'\d+', f)] + [0])

    global_model = NetworkWrapper(train_use_gpu=python_use_gpu, libtorch_use_gpu=libtorch_use_gpu, **config)

    if last_episode > 0:
        global_model.load_model(save_dir, f'checkpoint-ep{last_episode}')
        print(f"Reloading model checkpoint-ep{last_episode}")

    global_model.save_model('training', 'checkpoint')

    swig_wrapper = None
    if use_swig:
        swig_wrapper = NeuralNetwork('./training/checkpoint.pt', libtorch_use_gpu, 10)

    Worker.pBar = tqdm(total=iterations - last_episode)

    global_eps = int(iterations / its_per_ep)
    last_glob_ep = last_episode // its_per_ep
    w_threads = threads
    tree_threads = 1
    if use_swig:
        w_threads = 1
        tree_threads = threads

    BOARD_SIZE = (int(config['n']), int(config['n']))
    K = int(config['k'])

    for i in range(last_glob_ep, global_eps):

        Worker.global_episode = 0

        workers = [Worker(its_per_ep,
                          sim_count,
                          swig_wrapper if use_swig else global_model,
                          BOARD_SIZE, K,
                          tree_threads, use_swig) for i in range(w_threads)]

        for _, worker in enumerate(workers):
            worker.start()

        for w in workers:
            w.join()

        all_exp = [w.experience for w in workers]
        all_exp = [x for xs in all_exp for x in xs]
        all_exp = [x for xs in all_exp for x in xs]

        if len(all_exp) == 0:
            # something went wrong
            raise RuntimeError("Something probably went wrong")

        t_boards, t_actions, t_values = zip(*all_exp)
        t_boards, t_actions, t_values = shuffle(np.array(t_boards), np.array(t_actions), np.array(t_values))

        global_model.save_model('training', 'checkpoint')

        model_candidate = NetworkWrapper(**config)
        model_candidate.load_model('training', 'checkpoint')

        tData = list(zip(t_boards, t_actions, t_values))

        model_candidate.train(tData, training_batchsize, training_epochs)
        model_candidate.save_model('training', 'arena-new')

        game = KInARow(BOARD_SIZE, K)

        adapting = False
        if do_duelling:
            c = None
            if use_swig:
                arena_wrapper = NeuralNetwork('./training/arena-new.pt', libtorch_use_gpu, 10)
                c = Colosseum(game, AiPlayer(game, arena_wrapper, BOARD_SIZE, K, True, sim_count, tree_threads),
                              AiPlayer(game, swig_wrapper, BOARD_SIZE, K, True, sim_count, tree_threads))
            else:
                c = Colosseum(game, AiPlayer(game, model_candidate, BOARD_SIZE, K, False, sim_count, 1),
                              AiPlayer(game, global_model, BOARD_SIZE, K, False, sim_count, 1))
            print("Now Duel models to find out if better")
            n_wins, g_wins, draws = c.fight(duelling_count)
            print(f"New model won {n_wins} games, old model won {g_wins} games.")
            if n_wins > duelling_count // 2:
                adapting = True
            del arena_wrapper
            del c
        else:
            adapting = True
        if adapting:
            print("Adapting weights")
            # adapt new model weights
            if use_swig:
                # maybe need to be released from memory
                swig_wrapper.unload_model()
            model_candidate.save_model('training', 'checkpoint')
            global_model.load_model('training', 'checkpoint')
            if use_swig:
                swig_wrapper.reload_weights()
            try:
                global_model.save_model(save_dir, f'checkpoint-ep{(i + 1) * its_per_ep}')
            except:
                print("Error on saving new model")

    global_model.save_model(save_dir, 'final')
    if USE_WANDB:
        run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Config file for models')
    parser.add_argument('--out', type=str, help='name for saving the model', default='simple_agent')

    args = parser.parse_args()
    if not os.path.exists(args.config_file):
        raise ValueError("No config file found")
    with open(args.config_file) as f:
        config = json.load(f)
    config['save_dir'] = args.out
    allparams = config.copy()
    learn(allparams, **config)
