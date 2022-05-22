import numpy as np




def transform_state(observation):
    board, player = observation
    res = np.zeros((3,3,2))
    nBoard = np.array(board * player)
    f1 = nBoard == 1
    f2 = nBoard == -1
    res[:,:,0] = f1
    res[:,:,1] = f2
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
    res[:,:,0] = board[:,:,1]
    res[:,:,1] = board[:,:,0]
    return res

def expand_action(action):
    v = np.zeros(9)
    v[action] = 1
    return v


class Node:
    def __init__(self, parent, state,policy, P,V, Q, player):
        self.parent = parent
        self.state = state
        self.policy = policy
        self.P = P  # prior probability
        self.Q = Q  # action value
        self.V = V
        self.N = 0  # visit count
        self.player = player
        self.children = {}

    def __str__(self):
        return f"Node: P: {self.P} Q: {self.Q} V: {self.V} Exp: {self.children > 0}"

def select(node):
    selected_node = None
    max_ucb = 0
    for action, child in node.children.items():
        ucb = upper_confidence_bound(child)
        if ucb > max_ucb:
            max_ucb = ucb
            selected_node = child

    return selected_node


def upper_confidence_bound(node):
    U = node.P / (1 + node.N)
    return node.Q + U


def expand(node,model, env):
    for action, probability in enumerate(node.policy):

        if probability > 0:
            #child_state = apply_action(node.state, action)

            s_a = np.unravel_index(action, node.state.shape)

            child_state, v, done,_ = env.next_step((node.state,node.player),s_a)
            tChildState = transform_state(child_state)

            child_policy, child_value = model.predict(np.array([tChildState]))

            valid = env.get_valid(child_state)

            child_policy = child_policy[0].reshape((3, 3))
            child_policy[valid == 0] = 0
            child_policy = child_policy.flatten()
            node.children[action] = Node(node, child_state[0],child_policy, probability, child_value * child_state[1],v, child_state[1])


# TODO
def mcts(obs, model, env, simulations=10):

    tObs = transform_state(obs)

    policy , value = model.predict(np.array([tObs]))

    valid = env.get_valid(obs)

    policy = policy[0].reshape((3, 3))
    policy[valid == 0] = 0
    policy = policy.flatten()


    root = Node(None,obs[0],policy,1,value,-1,obs[1])

    expand(root,model,env)

    for i in range(simulations):
        node = select(root)
        while True:
            if terminal_state(node,env):
                back_propagate(node)
                break
            if len(node.children) == 0:
                expand(node,model,env)
            node = select(node)

    #return np.array([n.Q for a,n in root.children.items()]).argmax()
    return max(root.children, key=(lambda new_k: root.children[new_k].Q))

# TODO
def terminal_state(node,env):
    res = env.get_winner((node.state,node.player))
    if res is None:
        return False
    return True


def back_propagate(node):
    return
