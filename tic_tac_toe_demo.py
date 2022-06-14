
import gym
import boardgame2
import numpy as np


from tqdm import tqdm
from games.kinarow import TicTacToe
from ai_player import AiPlayer

env = TicTacToe()
agent = AiPlayer(env, 'first-tic-tac-toe-agent',False, 100)

# play against a simple ai agent
observation = env.reset()
step_count = 0
while True:
    env.render()
    print("\n")
    step_count += 1

    selected_action = -1
    if step_count % 2 == 1:
        selected_action = int(input("Enter your move: ")) - 1
    else:
        # selected_action = get_ai_move(m, observation)
        selected_action = agent.get_action(observation)

    selected = np.unravel_index(selected_action, observation[0].shape)

    observation, reward, done, info = env.step(selected)
    if done:
        env.render()
        if reward == 1:
            print("You won")
        elif reward == -1:
            print("You lost")
        else:
            print("A Draw")
        break
