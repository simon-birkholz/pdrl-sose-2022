import numpy as np


def transform_state(observation):
    board, player = observation
    res = np.zeros((3, 3, 2))
    nBoard = np.array(board * player)
    f1 = nBoard == 1
    f2 = nBoard == -1
    res[:, :, 0] = f1
    res[:, :, 1] = f2
    return np.asarray(res)


def get_board(observation):
    board, _ = observation
    res = np.zeros((3, 3, 2))
    f1 = board == 1
    f2 = board == -1
    res[:, :, 0] = f1
    res[:, :, 1] = f2
    return np.asarray(res)


def flip_colors(board):
    res = np.zeros((3, 3, 2))
    res[:, :, 0] = board[:, :, 1]
    res[:, :, 1] = board[:, :, 0]
    return res


def expand_action(action):
    v = np.zeros(9)
    v[action] = 1
    return v

def normalize_pd(pd):
    whole = sum(pd)
    new_pd = np.zeros(len(pd))
    for i in range(0,len(pd)):
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
        :return:
        """
        tObs = transform_state(current_state)

        # if self.head is None:
        # Init initial node
        policy, value = self.model.predict(np.array([tObs]), verbose=0)

        valid = self.game.get_valid(current_state)

        # todo write better wrapper class to hide the concrete shape of the policy
        policy = policy[0].reshape((3, 3))

        # give invalid move a probability of zero
        policy[valid == 0] = 0

        # transform back todo also hide this
        policy = policy.flatten()

        policy = normalize_pd(policy)

        self.head = Node(self, None, current_state, 1, policy, value[0, 0])

        # self.head.expand()

        for i in range(0, n_sim):
            self.head.search()

        # select most visited move

        most_visit_count = -1
        selected_a = -1

        for action, child in self.head.children.items():
            if child.N > most_visit_count:
                most_visit_count = child.N
                selected_a = action

        # apply move to tree search
        # self.head = self.head.children[selected_a]
        # self.head.print_node('-')

        return selected_a


class Node:
    def __init__(self, mcts, parent, state, P, policy, V):
        self.mcts = mcts  # reference to the wrapper of the search
        self.parent = parent  # the parent node for the current node
        self.state = state  # the game state represented by the current node. Tuple of (board,player)
        self.P = P  # prior probability (policy value from parent node)
        self.policy = policy  # the initial policy of the neural network
        self.Q = 0  # action value (not known yet)
        self.V = V
        self.N = 0  # visit count
        self.children = {}

    def __str__(self):
        return f"Node: P: {self.P} Q: {self.Q} V: {self.V} N: {self.N} Exp: {len(self.children) > 0}"

    def print_node(self,a,depth=0):
        print('\t' * depth + str(a) + ' '  + str(self))
        for a_c,c in self.children.items():
            c.print_node(a_c,depth+1)


    def upper_confidence_bound(self):
        """
        Calculates the upper confidence bound as given in the paper
        :return:
        """
        U = self.P / (1 + self.N)
        return self.Q + U

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
            self.N = 0
            return -self.V

        selected_child = self.select()
        value = selected_child.search()

        # average the Q action value and increase visit count
        self.Q = (self.N * self.Q + value) / (self.N + 1)
        self.N += 1

        return -value

    def select(self):
        """
        Selects a node by the rules given in the paper
        :return: tuple of (a,node) where a is the action represented by the node
        """
        selected = None
        max_ucb = float('-inf')
        for action, child in self.children.items():
            ucb = child.upper_confidence_bound()
            if ucb > max_ucb:
                max_ucb = ucb
                selected = child
        return selected

    def expand(self):
        if len(self.children) > 0:
            # node already expanded
            return

        for action, probability in enumerate(self.policy):

            # todo because policy probabilites can be zero, we need to normalize the remaining probabilities

            if probability > 0:
                # child_state = apply_action(node.state, action)

                s_a = np.unravel_index(action, self.state[0].shape)

                child_state, v, done, _ = self.mcts.game.next_step(self.state, s_a)
                tChildState = transform_state(child_state)

                child_policy, child_value = self.mcts.model.predict(np.array([tChildState]), verbose=0)

                valid = self.mcts.game.get_valid(child_state)

                # todo hide this details away (because game specific)
                child_policy = child_policy[0].reshape((3, 3))
                child_policy[valid == 0] = 0
                child_policy = child_policy.flatten()

                child_policy = normalize_pd(child_policy)

                self.children[action] = Node(self.mcts, self, child_state, probability, child_policy, child_value[0, 0])
