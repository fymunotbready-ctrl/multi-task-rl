import gymnasium as gym
from gymnasium import spaces
import numpy as np

DT = 0.1
MAX_SPEED = 5.0
REACH_DIST = 0.5


class DrivingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.zeros(2, dtype=np.float64)
        self.heading = self.np_random.uniform(-np.pi, np.pi)
        self.speed = 0.0
        self.steps = 0
        dist = self.np_random.uniform(5.0, 10.0)
        ang = self.np_random.uniform(0, 2 * np.pi)
        self.target = np.array([dist * np.cos(ang), dist * np.sin(ang)])
        self.prev_dist = np.linalg.norm(self.target - self.pos)
        return self._obs(), {}

    def _obs(self):
        return np.array(
            [self.pos[0], self.pos[1], self.heading, self.speed,
             self.target[0], self.target[1]],
            dtype=np.float32,
        )

    def step(self, action):
        self.steps += 1
        steer = float(np.clip(action[0], -1, 1))
        throttle = float(np.clip(action[1], -1, 1))

        self.speed += throttle * 2.0 * DT
        self.speed -= 0.2 * self.speed * DT          # drag
        self.speed = float(np.clip(self.speed, 0.0, MAX_SPEED))
        self.heading += steer * self.speed * 0.8 * DT  # no turning in place
        self.pos += np.array([np.cos(self.heading), np.sin(self.heading)]) * self.speed * DT

        dist = float(np.linalg.norm(self.target - self.pos))
        reward = (self.prev_dist - dist) - 0.005   # progress minus time cost
        self.prev_dist = dist

        reached = dist < REACH_DIST
        if reached:
            reward = 1.0

        terminated = reached
        truncated = self.steps >= 300
        return self._obs(), reward, terminated, truncated, {}
