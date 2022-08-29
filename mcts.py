"""
This file is used to perform a mcts (Monte Carlo Tree Search) to find promising actions for a given board position.
"""

import numpy as np


def normalize_pd(pd):
    """
    Method to normalize a policy distribution.

    :param pd: policy distribution
    :return: normalized policy distribution
    """
    whole = sum(pd)
    if whole == 0:
        raise ValueError("Invalid policy normalization")
    new_pd = np.zeros(len(pd))
    for i in range(0, len(pd)):
        new_pd[i] = pd[i] / whole
    return new_pd


class MonteCarloTreeSearch:
    """
    Wrap the Monte Carlo Tree Search in a class which accepts all necessary parameters
    and hides the unnecessary details for the user
    """

    def __init__(self, game, model):
        self.game = game  # the game enviroment used for the mcts search
        self.model = model  # the neural network used by the mcts search
        self.head = None  # the intro node for the tree

    def search(self, current_state, n_sim=10):
        """
        Performs the mcts for the current move
        :return: valid action deemed best by the mcts.
        """
        tObs = self.game.to_nn_view(current_state)

        # if self.head is None:
        # Init initial node
        policy, value = self.model.predict(np.array([tObs]))

        policy = self.game.correct_policy(policy, current_state)

        policy = normalize_pd(policy)

        self.head = Node(self, None, current_state, policy, value[0])

        # self.head.expand()

        for i in range(0, n_sim):
            self.head.search()

        # select most visited move

        most_visit_count = -1
        selected_a = -1

        for action in self.head.N.keys():
            if self.head.N[action] > most_visit_count:
                most_visit_count = self.head.N[action]
                selected_a = action

        # apply move to tree search
        # self.head = self.head.children[selected_a]
        # self.head.print_node('-')

        return selected_a


class Node:
    """
    A node represents the transition from a previous board state to the current one,
    including additional data, like the prior probability, the action value, the mean value and the visit count.
    """
    def __init__(self, mcts, parent, state, P, V):
        self.mcts = mcts  # reference to the wrapper of the search
        self.parent = parent  # the parent node for the current node
        self.state = state  # the game state represented by the current node. Tuple of (board,player)
        self.P = P  # prior probabilities for all s_a to child nodes
        self.Q = {}  # action value
        self.V = V
        self.N = {}  # visit count
        self.children = {}

    def __str__(self):
        return f"Node: V: {self.V} Exp: {len(self.children) > 0} T: {self.is_terminal_state()}"

    def print_node(self, a, p_a=0, q_a=0, n_a=0, depth=0):
        node_str = f"Node: P: {p_a} Q: {q_a} N: {n_a} V: {self.V} Exp: {len(self.children) > 0} T: {self.is_terminal_state()} "
        print('\t' * depth + str(a) + ' ' + node_str)
        for a_c, c in self.children.items():
            c.print_node(a_c, self.P[a_c], self.Q.get(a_c), self.N.get(a_c), depth + 1)

    def upper_confidence_bound(self, action):
        """
        Calculates the upper confidence bound as given in the paper
        :return:
        """
        if action in self.N.keys():
            U = self.P[action] / (1 + self.N[action])
            return self.Q[action] + U
        else:
            # edge has never been visited
            U = self.P[action]
            return U

    def is_terminal_state(self):
        """
        Returns if the current node is a terminal state
        :return:
        """
        res = self.mcts.game.get_winner(self.state)
        if res is None:
            return False
        return True

    def search(self):
        """
        Searches the tree by applying the mcts search algorithm
        :return: the current Q value for the node
        """
        if self.is_terminal_state():
            # terminal node found return game value
            return (-1) ** (self.state[1] == self.mcts.game.get_winner(self.state))

        if len(self.children) == 0:
            # leaf node, expand and return network value
            self.expand()
            return -self.V

        s_a, selected_child = self.select()
        value = selected_child.search()

        if s_a in self.Q.keys():
            # average the Q action value and increase visit count
            self.Q[s_a] = (self.N[s_a] * self.Q[s_a] + value) / (self.N[s_a] + 1)
            self.N[s_a] += 1
        else:
            self.Q[s_a] = value
            self.N[s_a] = 1

        return -value

    def select(self):
        """
        Selects a node by the rules given in the paper
        :return: tuple of (a,node) where a is the action represented by the node
        """
        selected = None
        max_ucb = -float('inf')
        for action, child in self.children.items():
            ucb = self.upper_confidence_bound(action)
            if ucb > max_ucb:
                max_ucb = ucb
                selected = action, self.children[action]
        return selected

    def expand(self):
        """
        Expands the children of the current selected node to the game tree.
        """
        if len(self.children) > 0:
            # node already expanded
            return

        for action, probability in enumerate(self.P):

            # todo because policy probabilities can be zero, we need to normalize the remaining probabilities

            if probability > 0:
                # child_state = apply_action(node.state, action)

                s_a = np.unravel_index(action, self.state[0].shape)

                child_state, v, done, _ = self.mcts.game.next_step(self.state, s_a)

                tChildState = self.mcts.game.to_nn_view(child_state)

                child_policy, child_value = self.mcts.model.predict(np.array([tChildState]))
                if not done:
                    child_policy = self.mcts.game.correct_policy(child_policy, child_state)
                    child_policy = normalize_pd(child_policy)
                    self.children[action] = Node(self.mcts, self, child_state, child_policy, child_value[0])
                else:
                    inv_policy = np.zeros(child_policy.shape)
                    self.children[action] = Node(self.mcts, self, child_state, inv_policy, child_value[0])
