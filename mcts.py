"""
This file is used to perform a mcts (Monte Carlo Tree Search) to find promising actions for a given board position.
It is the single threaded python counterpart of the C++ search in distributed_mcts.
"""

import math

import numpy as np


class Node:
    """
    A node is reached by a move. Its value sum is seen from the player who made that move.
    """

    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}


class MonteCarloTreeSearch:
    """
    Wrap the Monte Carlo Tree Search in a class which accepts all necessary parameters
    and hides the unnecessary details for the user
    """

    def __init__(self, game, model, c_puct=1.5, dirichlet_alpha=0.0, noise_fraction=0.25):
        """
        :param game: the game environment used for the search
        :param model: network wrapper with a predict method
        :param c_puct: exploration constant
        :param dirichlet_alpha: if > 0, Dirichlet noise is mixed into the root priors for self-play exploration
        :param noise_fraction: weight of the noise in the root priors
        """
        self.game = game
        self.model = model
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction

    def search(self, state, n_sim=10):
        """
        Performs the mcts for the current move.

        :param state: tuple of (board, player to move)
        :param n_sim: number of simulations
        :return: visit distribution over all board cells
        """
        root = Node(1.0)
        self._simulate(root, state)
        if self.dirichlet_alpha > 0 and root.children:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
            for child, n in zip(root.children.values(), noise):
                child.prior = (1 - self.noise_fraction) * child.prior + self.noise_fraction * n
        for _ in range(n_sim - 1):
            self._simulate(root, state)

        visits = np.zeros(state[0].size)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        if visits.sum() == 0:
            raise RuntimeError("Search visited no moves, the position is terminal")
        return visits / visits.sum()

    def _simulate(self, node, state):
        """
        Runs one simulation below node.

        :return: value of state for the player to move
        """
        winner = self.game.get_winner(state)
        if winner is not None:
            value = winner * state[1]
        elif not node.children:
            priors, value = self._evaluate(state)
            node.children = {action: Node(prior) for action, prior in priors.items()}
        else:
            sqrt_visits = math.sqrt(node.visit_count)
            action, child = max(node.children.items(), key=lambda item: self._puct(item[1], sqrt_visits))
            next_state = self.game.get_next_state(state, np.unravel_index(action, state[0].shape))
            value = -self._simulate(child, next_state)

        node.visit_count += 1
        node.value_sum -= value
        return value

    def _puct(self, child, sqrt_parent_visits):
        q = child.value_sum / child.visit_count if child.visit_count > 0 else 0.0
        return q + self.c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)

    def _evaluate(self, state):
        """
        :return: dict of legal move to prior, and the value for the player to move
        """
        policy, value = self.model.predict(np.array([self.game.to_nn_view(state)]))
        legal = np.flatnonzero(np.array(self.game.get_valid(state)).flatten())
        priors = np.maximum(policy[0][legal], 0)
        if priors.sum() <= 0:
            # all mass on illegal moves, fall back to uniform over legal ones
            priors = np.ones(len(legal))
        return dict(zip(legal.tolist(), priors / priors.sum())), float(value[0][0])
