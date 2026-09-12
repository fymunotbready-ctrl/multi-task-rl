import gymnasium as gym
from gymnasium import spaces
import numpy as np

G = 9.8
DT = 0.05
HIT_DIST = 0.5


class AimingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.fired = False
        self.ball = np.array([0.0, 0.0], dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        y = self.np_random.uniform(-3.0, 3.0)
        self.target = np.array([8.0, y], dtype=np.float64)
        direction = 1.0 if y < 0 else -1.0
        self.tvel = np.array([0.0, direction * self.np_random.uniform(0.5, 2.0)])
        self.min_dist = 999.0
        return self._obs(), {}

    def _obs(self):
        return np.array(
            [self.target[0], self.target[1], self.tvel[0], self.tvel[1],
             self.ball[0], self.ball[1]], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        if not self.fired:
            angle = np.interp(action[0], [-1, 1], [10, 80])
            power = np.interp(action[1], [-1, 1], [4.0, 14.0])
            rad = np.deg2rad(angle)
            self.vel = np.array([power * np.cos(rad), power * np.sin(rad)])
            self.fired = True
        self.target += self.tvel * DT
        self.vel[1] -= G * DT
        self.ball += self.vel * DT
        self.min_dist = min(self.min_dist, float(np.linalg.norm(self.ball - self.target)))
        reward = -0.005
        hit = self.min_dist < HIT_DIST
        if hit:
            reward = 1.0
        terminated = hit or self.ball[1] < -5 or self.ball[0] > 20
        truncated = self.steps >= 300
        if terminated and not hit:
            reward += max(0.0, 0.8 - 0.2 * self.min_dist)
        return (self._obs(), reward, terminated, truncated,
                {"success": bool(hit and terminated)})
