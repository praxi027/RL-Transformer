from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def select_action(self, obs, explore=True):
        ...

    @abstractmethod
    def update(self):
        """Perform one gradient step. Returns dict of metrics."""
        ...

    @abstractmethod
    def store_transition(self, obs, action, reward, next_obs, done):
        ...

    @abstractmethod
    def save(self, path):
        ...

    @abstractmethod
    def load(self, path):
        ...
