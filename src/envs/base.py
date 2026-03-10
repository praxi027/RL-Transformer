from abc import ABC, abstractmethod


class EnvWrapper(ABC):
    @abstractmethod
    def reset(self, seed=None):
        """Returns (observation_tensor, info_dict)."""
        ...

    @abstractmethod
    def step(self, action):
        """Returns (obs, reward, terminated, truncated, info)."""
        ...

    @property
    @abstractmethod
    def observation_dim(self):
        ...

    @property
    @abstractmethod
    def num_actions(self):
        ...

    def close(self):
        pass
