import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.basketball.basketball_env import BasketballEnv
from envs.driving.driving_env import DrivingEnv
from envs.aiming.aiming_env import AimingEnv


class MultiTaskEnv(gym.Env):
    """One env that IS all three skills. Each episode picks one at random."""

    def __init__(self):
        super().__init__()
        self.envs = [BasketballEnv(), DrivingEnv(), AimingEnv()]
        self.observation_space = spaces.Box(-np.inf, np.inf, (6,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.current = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = int(self.np_random.integers(0, len(self.envs)))
        self.current = self.envs[idx]
        return self.current.reset(seed=seed, options=options)

    def step(self, action):
        return self.current.step(action)
