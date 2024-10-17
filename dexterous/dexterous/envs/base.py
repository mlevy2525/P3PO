from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from gymnasium import core

STATE_TYPE = Dict[str, np.ndarray]


class BaseDexterousArmEnv(core.Env):
    """Implements abstract environment class with minimal functions that are called by replayer scripts"""

    @abstractmethod
    def compute_observation(self, obs_type: str) -> np.ndarray:
        """Returns the specified observation type as an array"""
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> STATE_TYPE:
        """Returns the state of the environment as dictionary mapping string keys to arrays"""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> STATE_TYPE:
        """Resets all robot states to home"""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray) -> STATE_TYPE:
        """Executes an action in the environment and returns the next observation"""
        raise NotImplementedError