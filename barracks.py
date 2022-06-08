import threading

import gym
import boardgame2
from collections import deque
import time
import argparse

import numpy as np
from sklearn.utils import shuffle

import tensorflow as tf
from keras.models import *
from keras.layers import *

from keras import backend as K

from mcts import MonteCarloTreeSearch
from tqdm import tqdm
from baselines import RandomPlayer
from games.kinarow import TicTacToe

from multiprocessing import Pool
import threading

tf.config.set_visible_devices([], 'GPU')
# use only first GPU
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[:1], 'GPU')
gpus = tf.config.list_physical_devices('GPU')
tpus = tf.config.list_physical_devices('TPU')

print(f'Found {len(gpus)} GPUs ')
for gpu in gpus:
    print(gpu)
print(f'Found {len(tpus)} TPUs ')


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

def simpleCNN(n_filters=9):
    inputs = Input(batch_shape=(None, 3, 3, 2))
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

    # model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'])
    model.make_predict_function()

    return model


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
        self.local_model = simpleCNN(32)
        self.env = TicTacToe()
        self.fallback_agent = RandomPlayer(self.env)
        self.total_episodes = total_episodes
        self.mcts_sims = mcts_sims
        self.opt = tf.keras.optimizers.Adam()

    def compute_loss(self,
                     exp):
        policies, values = self.local_model([exp[0]])

        loss_policy = tf.keras.losses.categorical_crossentropy(policies, exp[1])
        loss_value = tf.keras.losses.MSE(values, exp[2])
        total_loss = tf.reduce_mean(loss_policy + loss_value)
        return total_loss

    def run(self):

        while Worker.global_episode < self.total_episodes:
            observation = self.env.reset()
            observations = []
            mcts = MonteCarloTreeSearch(self.env, self.local_model)
            step_count = 0
            while True:
                step_count += 1
                # todo return policy vector for learning
                selected_action = mcts.search(observation, self.mcts_sims)
                if not selected_action > 0:
                    # mcts hasn't return valid value, use random baseline as fallback and report warning
                    selected_action = self.fallback_agent.get_action(observation)
                    print("Warning: Using fallback agent to sample random move")

                # todo hide away because game specific
                selected = np.unravel_index(selected_action, observation[0].shape)
                observations.append([observation, selected_action, None])
                observation, reward, done, info = self.env.step(selected)
                if done:
                    experience = [(self.env.to_nn_view(x[0]), expand_action(x[1]), (-1) ** (x[0][1] != reward)) for x in
                                  observations]
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(experience)
                    # Calculate local gradients
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    # Push local gradients to global model
                    self.opt.apply_gradients(zip(grads, self.global_model.trainable_weights))
                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())
                    Worker.global_episode += 1
                    Worker.pBar.update(1)
                    break


def learn(n_iter=100, queue_length=10, threads=1, save_dir='tmp'):
    history_train_examples = deque([], maxlen=queue_length)
    global_model = simpleCNN(32)
    pool = Pool(threads)

    Worker.pBar = tqdm(total=n_iter)

    workers = [Worker(n_iter,
                      100,
                      global_model,
                      history_train_examples) for i in range(threads)]

    for i, worker in enumerate(workers):
        worker.start()

    [w.join() for w in workers]

    global_model.save(save_dir)


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
