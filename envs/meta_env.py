import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.basketball.basketball_env import BasketballEnv
from envs.driving.driving_env import DrivingEnv
from envs.aiming.aiming_env import AimingEnv

TASK_NAMES = ["basketball", "driving", "aiming"]
N_TASKS = len(TASK_NAMES)
OBS_PAD = 6
REWARD_SCALE = {"driving": 0.2, "aiming": 0.2}


def encode_obs(obs, task_idx):
    padded = np.zeros(OBS_PAD, dtype=np.float32)
    padded[:len(obs)] = obs
    onehot = np.zeros(N_TASKS, dtype=np.float32)
    onehot[task_idx] = 1.0
    return np.concatenate([padded, onehot])


class MultiTaskEnv(gym.Env):
    """One episode = one random task. One-hot routes to the per-task
    projection; it never passes through shared weights."""

    def __init__(self):
        super().__init__()
        self.envs = [BasketballEnv(), DrivingEnv(), AimingEnv()]
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (OBS_PAD + N_TASKS,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.current = None
        self.current_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = int(self.np_random.integers(0, N_TASKS))
        self.current = self.envs[self.current_idx]
        obs, info = self.current.reset(seed=seed, options=options)
        info["task"] = self.current_idx
        return encode_obs(obs, self.current_idx), info

    def step(self, action):
        obs, r, term, trunc, info = self.current.step(action)
        scale = REWARD_SCALE.get(TASK_NAMES[self.current_idx], 1.0)
        info = dict(info)
        info["task"] = self.current_idx
        return encode_obs(obs, self.current_idx), r * scale, term, trunc, info
