import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.basketball.basketball_env import BasketballEnv
from envs.driving.driving_env import DrivingEnv
from envs.aiming.aiming_env import AimingEnv

N_TASKS = 3
# driving's reach bonus is 5.0 vs ~1.0 for others -> scale it down
# so no task dominates the shared gradient
REWARD_SCALE = {1: 0.1}


def add_task_id(obs, idx):
    onehot = np.zeros(N_TASKS, dtype=np.float32)
    onehot[idx] = 1.0
    return np.concatenate([obs, onehot])


class MultiTaskEnv(gym.Env):
    """One env that IS all three skills. Each episode picks one at random,
    and the observation carries a one-hot task ID so the shared policy
    knows which skill it's playing."""

    def __init__(self):
        super().__init__()
        self.envs = [BasketballEnv(), DrivingEnv(), AimingEnv()]
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (6 + N_TASKS,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.current = None
        self.current_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = int(self.np_random.integers(0, N_TASKS))
        self.current = self.envs[self.current_idx]
        obs, info = self.current.reset(seed=seed, options=options)
        return add_task_id(obs, self.current_idx), info

    def step(self, action):
        obs, r, term, trunc, info = self.current.step(action)
        if self.current_idx in REWARD_SCALE:
            r *= REWARD_SCALE[self.current_idx]
        return add_task_id(obs, self.current_idx), r, term, trunc, info
