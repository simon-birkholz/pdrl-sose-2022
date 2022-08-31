import json

import matplotlib.pyplot as plt

from pathlib import Path

DELTA = 500
ENDING = 5500

def make_elo_diagram(filename):
    scores = dict()
    titles = dict()
    # load last data entry
    with open(filename, 'r') as f:
        for line in f:
            scores = json.loads(line.strip())
            #if scores['sim_count'] == 100:
            #    break

    model_names = [('6x4_conv_lr0.1', 'orangered', 'Conv (lr=0.1)'), ('6x4_conv_lr0.01', 'red', 'Conv (lr=0.01)'),
                   ('6x4_conv_lr0.001', 'maroon', 'Conv (lr=0.001)'),
                   ('6x4_resnet_lr0.1', 'limegreen', 'ResNet (lr=0.1)'),
                   ('6x4_resnet_lr0.01', 'forestgreen', 'ResNet (lr=0.01)'),
                   ('6x4_resnet_lr0.001', 'darkgreen', 'ResNet (lr=0.001)'),
                   ('6x4_densenet_lr0.1', 'teal', 'DenseNet (lr=0.1)'),
                   ('6x4_densenet_lr0.01', 'dodgerblue', 'DenseNet (lr=0.01)'),
                   ('6x4_densenet_lr0.001', 'royalblue', 'DenseNet (lr=0.001)')]

    model_names = [('6x4_conv_lr0.01', 'red', 'Conv (lr=0.01)'),
                   ('6x4_resnet_lr0.001', 'darkgreen', 'ResNet (lr=0.001)'),
                   ('6x4_densenet_lr0.01', 'dodgerblue', 'DenseNet (lr=0.01)')]

    episodes = [i for i in range(DELTA, ENDING, DELTA)]

    # aggregate model data
    aggregated_scores = dict()
    colors = dict()
    for model, color, title in model_names:
        colors[model] = color
        titles[model] = title
        for ep in episodes:
            score = scores.get(f'{model}_{ep}', -1)
            if score > 0:
                aggregated_scores.setdefault(model, []).append(score)

    # aggregate baseline data
    aggregated_scores['random'] = [scores['random'] for i in episodes]
    colors['random'] = 'orange'
    titles['random'] = 'Random Player'
    aggregated_scores['greedy'] = [scores['greedy'] for i in episodes]
    colors['greedy'] = 'gold'
    titles['greedy'] = 'Greedy Player'

    #for model in aggregated_scores.keys():
    #    aggregated_scores[model] = [800 + (800 - w) for w in aggregated_scores[model]]


    for model in aggregated_scores.keys():
        plt.plot(episodes[:len(aggregated_scores[model])], aggregated_scores[model], label=titles[model], color=colors[model])

    plt.ylabel('Elo-Score')
    plt.xlabel('Episode')
    plt.title('Elo performance over time')
    plt.legend()
    plt.savefig(f'diagrams/{Path(filename).stem}.svg')
    plt.close()


if __name__ == '__main__':
    make_elo_diagram('elo-calc-out2.jsonl')
