"""
This file is used define the behavior of game classes.
"""

from abc import ABCMeta, abstractmethod


class Game(metaclass=ABCMeta):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_valid(self, state):
        """
                Get all valid locations for the current state.

                Parameters
                ----
                state : (np.array, int)    board and player

                Returns
                ----
                valid : np.array     current valid place for the player
                """
        pass

    @abstractmethod
    def get_winner(self, state):
        """
                Check whether the game has ended. If so, who is the winner.

                Parameters
                ----
                state : (np.array, int)   board and player. only board info is used

                Returns
                ----
                winner : None or int
                    - None       The game is not ended and the winner is not determined.
                    - env.BLACK  The game is ended with the winner BLACK.
                    - env.WHITE  The game is ended with the winner WHITE.
                    - env.EMPTY  The game is ended tie.
                """
        pass

    @abstractmethod
    def get_next_state(self, state, action):
        """
                Get the next state.

                Parameters
                ----
                state : (np.array, int)    board and current player
                action : np.array    location and skip indicator

                Returns
                ----
                next_state : (np.array, int)    next board and next player

                Raise
                ----
                ValueError : location in action is not valid
                """
        pass

    @abstractmethod
    def next_step(self, state, action):
        """
                Get the next observation, reward, done, and info.

                Parameters
                ----
                state : (np.array, int)    board and current player
                action : np.array    location

                Returns
                ----
                next_state : (np.array, int)    next board and next player
                reward : float               the winner or zeros
                done : bool           whether the game end or not
                info : {'valid' : np.array}    a dict shows the valid place for the next player
                """
        pass

    @abstractmethod
    def step(self, action):
        """
                See gym.Env.step().

                Parameters
                ----
                action : np.array    location

                Returns
                ----
                next_state : (np.array, int)    next board and next player
                reward : float        the winner or zero
                done : bool           whether the game end or not
                info : {}
                """
        pass

    @abstractmethod
    def to_nn_view(self, state):
        pass

    @abstractmethod
    def correct_policy(self, policy, state):
        pass

    @abstractmethod
    def render(self):
        """
                See gym.Env.render().
                """
        pass
