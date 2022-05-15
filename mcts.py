# TODO
class Node:
    def __init__(self, parent, state, P, Q, player):
        self.parent = parent
        self.state = state
        self.P = P  # prior probability
        self.Q = Q  # action value
        self.N = 0  # visit count
        self.player = player
        self.children = {}


def select(node):
    selected_node = None
    max_ucb = 0
    for child in node.children:
        ucb = upper_confidence_bound(child)
        if ucb > max_ucb:
            max_ucb = ucb
            selected_node = child

    return selected_node


def upper_confidence_bound(node):
    U = node.P / (1 + node.N)
    return node.Q + U


def expand(node):
    policy, values = [], []  # TODO resnet(node)
    for action, (probability, value) in enumerate(zip(policy, values)):
        child_state = apply_action(node.state, action)
        node.children[action] = Node(node, child_state, probability, value * -node.player, -node.player)


# TODO
def apply_action(state, action):
    return state


# TODO
def mcts(root, simulations=1600):
    expand(root)

    for i in range(simulations):
        node = select(root)
        while True:
            if terminal_state(node) == 1 or -1:
                back_propagate(node)
                return
            if len(node.children) == 0:
                expand(node)
            node = select(node)

    return


# TODO
def terminal_state(node):
    return 1 * node.player


def back_propagate(node):
    return
