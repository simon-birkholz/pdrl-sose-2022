from ai_player import AiPlayer
from baselines import RandomPlayer
from colosseum import Colosseum
from games.kinarow import KInARow
from models.neural_network import NetworkWrapper
from operator import add
import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm

BOARD_SIZE = (3, 3)
K = 3

HYPER_PARAMS = {
    'lr': 0.001,
    'l2': 0.0001,
    'n_layers': 4,
    'n_channels': 256,
    'arch': 'conv',
    'n': 3
}


def make_bar_plot_stacked(values):
    episodes, p1_wins, p2_wins, draws = zip(*values)

    episodes = list(episodes)
    p1_wins = list(p1_wins)
    p2_wins = list(p2_wins)
    draws = list(draws)

    tmp = list(map(add, draws, p1_wins))

    plt.bar(episodes, p1_wins, label='Model wins')
    plt.bar(episodes, draws, bottom=p1_wins, label='draws')
    plt.bar(episodes, p2_wins, bottom=tmp, label='Random')

    plt.ylabel('Games')
    plt.xlabel('Episode')
    plt.title('Playing performance over time')
    plt.legend()

    plt.show()
    # plt.savefig('test.png')


def get_model_performances(m1_folder, m2_folder):
    game = KInARow(BOARD_SIZE, K)

    model1 = NetworkWrapper(**HYPER_PARAMS, train_use_gpu=False, libtorch_use_gpu=False)

    player1 = AiPlayer(game, model1, BOARD_SIZE, K, False, 100, 1)

    player2 = RandomPlayer(game)

    c = Colosseum(game, player1,
                  player2)

    max_eps = 5000
    delta_eps = 200

    result = []
    for episode in tqdm(range(0, max_eps, delta_eps)):
        model1.load_model(m1_folder, f'checkpoint-ep{episode}')

        p1_won, p2_won, draws = c.fight(20)
        result.append((episode, p1_won, p2_won, draws))

    return result


if __name__ == '__main__':
    f_name = 'whhhhhy'
    m_name = 'random_start_test'

    #val = None
    #with open(f'{f_name}.pickle', 'rb') as f:
    #    val = pickle.load(f)

    val = get_model_performances(m_name,'whut')

    with open(f'{f_name}.pickle','wb') as f:
        pickle.dump(val,f)

    make_bar_plot_stacked(val)
