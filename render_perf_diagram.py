import os.path
from operator import add
import matplotlib.pyplot as plt
import pickle


def make_bar_plot_stacked(file_name, m1_name, m2_name, c1, c2, *, right=2000):
    with open(f'experiments/evaluation/{file_name}.pickle', 'rb') as f:
        values = pickle.load(f)

    episodes, p1_wins, p2_wins, draws = zip(*values)

    episodes = list(episodes)
    p1_wins = list(p1_wins)
    p2_wins = list(p2_wins)
    draws = list(draws)

    no_draw = sum(draws) == 0

    tmp = list(map(add, draws, p1_wins))

    w = 100

    plt.bar(episodes, p1_wins, w, label=f'{m1_name} wins', color=c1)
    if not no_draw:
        plt.bar(episodes, draws, w, bottom=p1_wins, label='draws', color='tab:grey')
    plt.bar(episodes, p2_wins, w, bottom=tmp, label=f'{m2_name} wins', color=c2)

    plt.xlim(right=right)
    plt.ylabel('Games')
    plt.xlabel('Episode')
    plt.title('Playing performance over time')
    plt.legend()

    # plt.show()
    plt.savefig(f'diagrams/{file_name}.pdf')
    plt.close()


if __name__ == '__main__':

    if not os.path.exists('diagrams'):
        os.mkdir('diagrams')

    make_bar_plot_stacked('6x4_conv_lr0.1_vs_6x4_conv_lr0.01', 'Conv (lr=0.1)', 'Conv (lr=0.01)', 'tab:red',
                          'tab:green')
    make_bar_plot_stacked('6x4_conv_lr0.1_vs_6x4_conv_lr0.001', 'Conv (lr=0.1)', 'Conv (lr=0.001)', 'tab:red',
                          'tab:blue')
    make_bar_plot_stacked('6x4_conv_lr0.01_vs_6x4_conv_lr0.001', 'Conv (lr=0.01)', 'Conv (lr=0.001)', 'tab:green',
                          'tab:blue')
    make_bar_plot_stacked('6x4_resnet_lr0.1_vs_6x4_resnet_lr0.01', 'ResNet (lr=0.1)', 'ResNet (lr=0.01)', 'tab:red',
                          'tab:green')
    make_bar_plot_stacked('6x4_resnet_lr0.1_vs_6x4_resnet_lr0.001', 'ResNet (lr=0.1)', 'ResNet (lr=0.001)', 'tab:red',
                          'tab:blue')
    make_bar_plot_stacked('6x4_resnet_lr0.01_vs_6x4_resnet_lr0.001', 'ResNet (lr=0.01)', 'ResNet (lr=0.001)',
                          'tab:green',
                          'tab:blue')
    make_bar_plot_stacked('6x4_densenet_lr0.1_vs_6x4_densenet_lr0.01', 'DenseNet (lr=0.1)', 'DenseNet (lr=0.01)',
                          'tab:red',
                          'tab:green')
    make_bar_plot_stacked('6x4_densenet_lr0.1_vs_6x4_densenet_lr0.001', 'DenseNet (lr=0.1)', 'DenseNet (lr=0.001)',
                          'tab:red',
                          'tab:blue')
    make_bar_plot_stacked('6x4_densenet_lr0.01_vs_6x4_densenet_lr0.001', 'DenseNet (lr=0.01)', 'DenseNet (lr=0.001)',
                          'tab:green',
                          'tab:blue')
    make_bar_plot_stacked('6x4_conv_vs_resnet', 'Conv (lr=0.01)','Resnet (lr=0.001)',
                          'tab:red',
                          'darkgreen', right=3000)
    make_bar_plot_stacked('6x4_densenet_vs_resnet', 'DenseNet (lr=0.01)', 'Resnet (lr=0.001)',
                          'dodgerblue',
                          'darkgreen', right=3000)
    make_bar_plot_stacked('6x4_densenet_vs_conv', 'DenseNet (lr=0.01)', 'Conv (lr=0.01)',
                          'dodgerblue',
                          'tab:red', right=3000)
