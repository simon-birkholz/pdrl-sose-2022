import json
import os.path
import random

from ai_player import AiPlayer
from baselines import RandomPlayer, GreedyPlayer
from colosseum import Colosseum
from games.kinarow import KInARow

from tqdm import tqdm

from swig.distributed_mcts import NeuralNetwork

BOARD_SIZE = (6, 6)
K = 4

OUTFILE = 'elo-calc-out2.jsonl'
USE_SWIG = True
env = KInARow(BOARD_SIZE, K)

DELTA = 500

SIM_COUNT = 400
ELO_K = 20

players = {'random': RandomPlayer(env), 'greedy': GreedyPlayer(env)}

# model_names = ['6x4_conv_lr0.1', '6x4_conv_lr0.01', '6x4_conv_lr0.001', '6x4_resnet_lr0.1',
#               '6x4_resnet_lr0.01', '6x4_resnet_lr0.001', '6x4_densenet_lr0.1', '6x4_densenet_lr0.01',
#               '6x4_densenet_lr0.001']

model_names = ['6x4_conv_lr0.01', '6x4_resnet_lr0.001', '6x4_densenet_lr0.01', ]


def load_model_to_players(model_name):
    global env
    global players
    episode = DELTA
    while os.path.exists(f'{model_name}/checkpoint-ep{episode}.pt'):
        if not USE_SWIG:
            raise NotImplementedError("Non Swig not supported")
            # takes waaaaay to long
        else:
            model = NeuralNetwork(f'./{model_name}/checkpoint-ep{episode}.pt', False, 10)
            player = AiPlayer(env, model, BOARD_SIZE, K, True, 100, 1)
            players[f'{model_name}_{episode}'] = player
            episode += DELTA


# Load all models which are going to be elo evaluated
for model_name in model_names:
    load_model_to_players(model_name)

print(f"Loaded {len(players)} agents")
# print(players)

# Init elo score for models
scores = {name: 800 for name in players.keys()}

prev_scores = scores.copy()

for sim in tqdm(range(0, SIM_COUNT)):
    player1 = random.choice(list(players.keys()))
    player2 = random.choice(list(players.keys()))
    while player1 == player2:
        player2 = random.choice(list(players.keys()))
    # Make sure to have two seperate players

    elo1 = scores[player1]
    elo2 = scores[player2]

    # calculate elo expected value
    E1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    E2 = 1 - E1

    c = Colosseum(env, players[player1],
                  players[player2])

    p1_won, p2_won, draw = c.fight(1)
    p1_won = p1_won == 1
    p2_won = p2_won == 1
    draw = draw == 1
    # Adapt elo ratings

    S1 = (1.0 if p1_won else (0.0 if p2_won else 0.5))
    S2 = 1 - S1

    n_elo1 = elo1 + ELO_K * (S1 - E1)
    n_elo2 = elo2 + ELO_K * (S2 - E2)

    print(f"Adapting elo of {player1}: {elo1} -> {n_elo1} (delta: {n_elo1 - elo1})")
    print(f"Adapting elo of {player2}: {elo2} -> {n_elo2} (delta: {n_elo2 - elo2})")

    scores[player1] = n_elo1
    scores[player2] = n_elo2

    if sim % 10 == 0:
        elo_delta = sum([abs(scores[p] - prev_scores[p]) for p in players.keys()])

        print(f"All elo changed by {elo_delta}")

        scores['sim_count'] = sim
        with open(OUTFILE, 'a+') as f:
            f.write(f"{json.dumps(scores)}\n")

        prev_scores = scores.copy()
