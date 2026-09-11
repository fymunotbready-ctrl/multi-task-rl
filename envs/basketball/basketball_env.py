import gymnasium as gym
from gymnasium import spaces
import numpy as np

G = 9.8
DT = 0.05


class BasketballEnv(gym.Env):
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
        self.ball = np.array([0.0, 1.0], dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.launched = False
        self.steps = 0
        self.min_dist = 999.0
        self.hoop = np.array(
            [6.0 + self.np_random.uniform(-1.5, 1.5), 3.0]
        )
        return self._obs(), {}

    def _obs(self):
        return np.array(
            [self.ball[0], self.ball[1],
             self.vel[0], self.vel[1],
             self.hoop[0], self.hoop[1]],
            dtype=np.float32,
        )

    def step(self, action):
        self.steps += 1
        if not self.launched:
            angle = np.interp(action[0], [-1, 1], [15, 85])
            power = np.interp(action[1], [-1, 1], [3.0, 12.0])
            rad = np.deg2rad(angle)
            self.vel = np.array(
                [power * np.cos(rad), power * np.sin(rad)]
            )
            self.launched = True

        prev_x = self.ball[0]
        self.vel[1] -= G * DT
        self.ball += self.vel * DT

        self.min_dist = min(
            self.min_dist,
            float(np.linalg.norm(self.ball - self.hoop))
        )

        reward = -0.01
        scored = False
        if prev_x < self.hoop[0] <= self.ball[0]:
            if abs(self.ball[1] - self.hoop[1]) < 0.3:
                reward = 1.0
                scored = True

        terminated = scored or self.ball[1] < 0
        truncated = self.steps >= 200

        if terminated and not scored:
            reward += max(0.0, 0.6 - 0.15 * self.min_dist)

        return self._obs(), reward, terminated, truncated, {}
