import os.path

from ai_player import AiPlayer
from baselines import RandomPlayer, GreedyPlayer
from colosseum import Colosseum
from games.kinarow import KInARow
from models.neural_network import NetworkWrapper
from operator import add
import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm

from swig.distributed_mcts import NeuralNetwork

BOARD_SIZE = (6, 6)
K = 4

USE_SWIG = True

PARAMS_1 = {
    'lr': 0.01,
    'l2': 0.0001,
    'n_layers': 3,
    'n_channels': 256,
    'arch': 'densenet',
    'n': 6
}

PARAMS_2 = {
    'lr': 0.001,
    'l2': 0.0001,
    'n_layers': 3,
    'n_channels': 256,
    'arch': 'densenet',
    'n': 6
}


def make_bar_plot_stacked(values, title='test'):
    episodes, p1_wins, p2_wins, draws = zip(*values)

    episodes = list(episodes)
    p1_wins = list(p1_wins)
    p2_wins = list(p2_wins)
    draws = list(draws)

    tmp = list(map(add, draws, p1_wins))

    w = 100

    plt.bar(episodes, p1_wins, w, label='Model 1 wins')
    plt.bar(episodes, draws, w, bottom=p1_wins, label='draws')
    plt.bar(episodes, p2_wins, w, bottom=tmp, label='Model 2 wins')

    plt.ylabel('Games')
    plt.xlabel('Episode')
    plt.title('Playing performance over time')
    plt.legend()

    # plt.show()
    plt.savefig(f'{title}.png')


def get_model_performances(m1_folder, m2_folder):
    game = KInARow(BOARD_SIZE, K)

    player1 = None
    player2 = None
    model1 = None
    model2 = None
    c = None
    if not USE_SWIG:
        model1 = NetworkWrapper(**PARAMS_1, train_use_gpu=False, libtorch_use_gpu=False)
        model2 = NetworkWrapper(**PARAMS_2, train_use_gpu=False, libtorch_use_gpu=False)

        player1 = AiPlayer(game, model1, BOARD_SIZE, K, False, 100, 1)
        player2 = AiPlayer(game, model2, BOARD_SIZE, K, False, 100, 1)
        # player2 = GreedyPlayer(game)

        c = Colosseum(game, player1,
                      player2)

    max_eps = 2000
    delta_eps = 200

    result = []
    for episode in tqdm(range(delta_eps, max_eps, delta_eps)):
        if not USE_SWIG:
            model1.load_model(m1_folder, f'checkpoint-ep{episode}')
            model2.load_model(m2_folder, f'checkpoint-ep{episode}')

            p1_won, p2_won, draws = c.fight(20)
            result.append((episode, p1_won, p2_won, draws))

        else:
            model1 = NeuralNetwork(f'./{m1_folder}/checkpoint-ep{episode}.pt', False, 10)
            model2 = NeuralNetwork(f'./{m2_folder}/checkpoint-ep{episode}.pt', False, 10)

            player1 = AiPlayer(game, model1, BOARD_SIZE, K, True, 100, 1)
            player2 = AiPlayer(game, model2, BOARD_SIZE, K, True, 100, 1)

            c = Colosseum(game, player1,
                          player2)

            p1_won, p2_won, draws = c.fight(20)
            result.append((episode, p1_won, p2_won, draws))

    return result


if __name__ == '__main__':

    if not os.path.exists('experiments/evaluation'):
        os.mkdir('experiments/evaluation')

    # val = None
    # with open(f'{f_name}.pickle', 'rb') as f:
    #    val = pickle.load(f)

    val = get_model_performances(m_name, m_name2)

    with open(f'experiments/evaluation/{f_name}.pickle', 'wb') as f:
        pickle.dump(val, f)

    make_bar_plot_stacked(val, f_name)
