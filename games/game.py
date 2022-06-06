from abc import ABCMeta, abstractmethod

class Game(metaclass=ABCMeta):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_valid(self, state):
        pass

    @abstractmethod
    def get_winner(self, state):
        pass

    @abstractmethod
    def get_next_state(self, state, action):
        pass

    @abstractmethod
    def next_step(self, state, action):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def to_nn_view(self, state):
        pass

    @abstractmethod
    def correct_policy(self, policy, state):
        pass

    @abstractmethod
    def render(self):
        pass
