"""
This file is used to configure and train neural networks.
"""

import argparse
import glob
import json
import os
import re
import threading
import time

import numpy as np
from sklearn.utils import shuffle

from models.neural_network import NetworkWrapper

from swig.distributed_mcts import MonteCarloTreeSearch, SearchState, PII
from swig.distributed_mcts import NeuralNetwork

import mcts as old_mcts

from tqdm import tqdm
from games.kinarow import KInARow, generate_symmetries

from colosseum import Colosseum
from ai_player import AiPlayer

try:
    import wandb

    USE_WANDB = True
except ImportError:
    USE_WANDB = False


class Worker(threading.Thread):
    """
    Helper class to distribute the learning process.
    """
    global_episode = 0
    pBar = tqdm()
    save_lock = threading.Lock()

    def __init__(self, total_episodes, mcts_sims, global_model, board_size, k, threads=1, use_swig=False,
                 c_puct=1.5, dirichlet_alpha=0.3, temperature_moves=8):
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
        self.use_symmetries = True
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.temperature_moves = temperature_moves  # moves sampled from the visit distribution before playing greedily

    def run(self):
        while Worker.global_episode < self.total_episodes:
            self.experience.append(self.play_episode())
            Worker.global_episode += 1
            Worker.pBar.update(1)

    def play_episode(self):
        """
        Plays one game of self-play.

        :return: list of (network input, visit distribution, outcome for the player to move)
        """
        observation = self.env.reset()
        if self.use_swig:
            mcts = MonteCarloTreeSearch(self.global_model, PII(self.board_shape[0], self.board_shape[1]), self.k,
                                        self.mcts_sims, self.threads, self.c_puct, self.dirichlet_alpha)
        else:
            mcts = old_mcts.MonteCarloTreeSearch(self.env, self.global_model, self.c_puct, self.dirichlet_alpha)

        history = []
        start_time = time.perf_counter()
        while True:
            if self.use_swig:
                policy = np.array(mcts.search(SearchState(observation[0].flatten().tolist(), observation[1])))
            else:
                policy = mcts.search(observation, self.mcts_sims)

            if len(history) < self.temperature_moves:
                action = np.random.choice(len(policy), p=policy)
            else:
                action = int(np.argmax(policy))
            history.append((observation, policy))

            if self.use_swig:
                mcts.apply_action(int(action))
            observation, reward, done, _ = self.env.step(np.unravel_index(action, observation[0].shape))
            if done:
                break

        if USE_WANDB:
            wandb.log({'episode_length': len(history), 'elapsed_time': time.perf_counter() - start_time})

        samples = [sample for state, policy in history
                   for sample in (generate_symmetries(state, policy) if self.use_symmetries else [(state, policy)])]
        # reward is the winner of the game, 0 for a draw
        return [(self.env.to_nn_view(state), policy, reward * state[1]) for state, policy in samples]


def learn(allparams, iterations, threads, save_dir, use_swig, libtorch_use_gpu, python_use_gpu, its_per_ep, sim_count,
          do_duelling,
          duelling_count, training_epochs, training_batchsize, c_puct=1.5, dirichlet_alpha=0.3, temperature_moves=8,
          **config):
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
    :param c_puct: exploration constant of the search
    :param dirichlet_alpha: concentration of the root noise in self-play, 0 disables it
    :param temperature_moves: number of opening moves sampled from the visit distribution
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

    w_threads = threads
    tree_threads = 1
    if use_swig:
        w_threads = 1
        tree_threads = threads

    # one batch collects the requests of all search threads
    swig_wrapper = None
    if use_swig:
        swig_wrapper = NeuralNetwork('./training/checkpoint.pt', libtorch_use_gpu, tree_threads)

    Worker.pBar = tqdm(total=iterations - last_episode)

    global_eps = int(iterations / its_per_ep)
    last_glob_ep = last_episode // its_per_ep

    BOARD_SIZE = (int(config['n']), int(config['n']))
    K = int(config['k'])

    for i in range(last_glob_ep, global_eps):

        Worker.global_episode = 0

        workers = [Worker(its_per_ep,
                          sim_count,
                          swig_wrapper if use_swig else global_model,
                          BOARD_SIZE, K,
                          tree_threads, use_swig, c_puct, dirichlet_alpha, temperature_moves) for i in range(w_threads)]

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

        model_candidate = NetworkWrapper(train_use_gpu=python_use_gpu, libtorch_use_gpu=libtorch_use_gpu, **config)
        model_candidate.load_model('training', 'checkpoint')

        tData = list(zip(t_boards, t_actions, t_values))

        model_candidate.train(tData, training_batchsize, training_epochs)
        model_candidate.save_model('training', 'arena-new')

        game = KInARow(BOARD_SIZE, K)

        adapting = False
        if do_duelling:
            c = None
            if use_swig:
                arena_wrapper = NeuralNetwork('./training/arena-new.pt', libtorch_use_gpu, tree_threads)
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
