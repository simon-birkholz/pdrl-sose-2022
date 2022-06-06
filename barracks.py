import gym
import boardgame2
from collections import deque
import time
import argparse

import numpy as np
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, ReLU, Add, BatchNormalization, GlobalAveragePooling2D, Dense, MaxPooling2D

from mcts import MonteCarloTreeSearch
from tqdm import tqdm
from baselines import RandomPlayer
from games.kinarow import TicTacToe

#tf.config.set_visible_devices([], 'GPU')
# use only first GPU
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.set_visible_devices(physical_devices[:1], 'GPU')
gpus = tf.config.list_physical_devices('GPU')
tpus = tf.config.list_physical_devices('TPU')

print(f'Found {len(gpus)} GPUs ')
for gpu in gpus:
    print(gpu)
print(f'Found {len(tpus)} TPUs ')

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

def simpleCNN(n_filters=9):
    inputs = Input(shape=(3, 3, 2))
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

    x = Conv2D(filters=n_filters, kernel_size=3, padding='same', strides=(1, 1))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=n_filters, kernel_size=3, padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=n_filters, kernel_size=3, padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=n_filters, kernel_size=3, padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(512, kernel_initializer=initializer)(x)

    policy = Dense(9, activation='softmax', kernel_initializer=initializer)(x)
    value = Dense(1, activation='tanh', kernel_initializer=initializer)(x)

    outputs = policy, value
    model = Model(inputs=inputs, outputs=outputs, name='Oscar', )
    model.summary()

    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'])

    return model

def expand_action(action):
    v = np.zeros(9)
    v[action] = 1
    return v

m = simpleCNN(64)

env = TicTacToe()

fallback_agent = RandomPlayer(env)

def train_episode():
    """
    Executes one episode of the self-play algorithm
    :return:
    """
    observations = []
    observation = env.reset()
    mcts = MonteCarloTreeSearch(env, m)

    step_count = 0
    while True:
        #env.render()
        #print("\n")
        step_count += 1

        # todo return policy vector for learning
        selected_action = mcts.search(observation, 100)

        if not selected_action > 0:
            # mcts hasn't return valid value, use random baseline as fallback and report warning
            selected_action = fallback_agent.get_action(observation)
            print("Warning: Using fallback agent to sample random move")
            
        # todo hide away because game specific
        selected = np.unravel_index(selected_action, observation[0].shape)

        observations.append([observation, selected_action, None])

        observation, reward, done, info = env.step(selected)
        if done:
            #print("Done with game. Outcome: ", reward)

            return [(env.to_nn_view(x[0]), x[1], (-1) ** (x[0][1] != reward)) for x in observations]


def learn(n_iter=100, queue_length=10):
    history_train_examples = deque([],maxlen=queue_length)
    for i in tqdm(range(0,n_iter)):
        examples = train_episode()

        train_board = np.array([x[0] for x in examples])
        train_actions = np.array([expand_action(x[1]) for x in examples])
        train_values = np.array([x[2] for x in examples])

        history_train_examples.append((train_board,train_actions,train_values))

        t_boards = list()
        t_actions = list()
        t_values = list()
        for b,a,v in history_train_examples:
            t_boards += b.tolist()
            t_actions += a.tolist()
            t_values += v.tolist()
        t_boards, t_actions, t_values = shuffle(np.array(t_boards),np.array(t_actions), np.array(t_values))

        m.fit(x=t_boards, y=[t_actions, t_values], verbose=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('iterations', type=int,
                        help='Number of iterations learned')
    parser.add_argument('--queue_size', type=int, default=10,
                        help='Size of the replay buffer queue')
    parser.add_argument('--out', type=str, help='name for saving the model', default='simple_agent')

    args = parser.parse_args()

    learn(args.iterations,args.queue_size)
    m.save(args.out)
    #print("Used ", fallback_counter, " fallback moves")
