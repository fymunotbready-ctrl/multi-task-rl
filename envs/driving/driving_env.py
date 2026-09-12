import gymnasium as gym
from gymnasium import spaces
import numpy as np

DT = 0.1
MAX_SPEED = 5.0
MIN_TURN_SPEED = 1.0
REACH_DIST = 0.5
MAX_STEPS = 300


class DrivingEnv(gym.Env):
    """v3: sin/cos heading obs, steer authority at zero speed, stall penalty.
    Fixes the verified failure: PPO collapsing to a standstill."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

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
        rel = self.target - self.pos
        return np.array(
            [np.sin(self.heading), np.cos(self.heading),
             self.speed, rel[0], rel[1]], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        steer = float(np.clip(action[0], -1, 1))
        throttle = float(np.clip(action[1], -1, 1))
        self.speed += throttle * 2.0 * DT
        self.speed -= 0.2 * self.speed * DT
        self.speed = float(np.clip(self.speed, 0.0, MAX_SPEED))
        self.heading += steer * max(self.speed, MIN_TURN_SPEED) * 0.8 * DT
        self.pos += np.array([np.cos(self.heading), np.sin(self.heading)]) * self.speed * DT

        dist = float(np.linalg.norm(self.target - self.pos))
        progress = self.prev_dist - dist
        self.prev_dist = dist

        reward = progress - 0.01
        if self.speed < 0.1:
            reward -= 0.05
        reached = dist < REACH_DIST
        if reached:
            reward = 5.0
        return (self._obs(), float(reward), reached,
                self.steps >= MAX_STEPS, {"success": bool(reached)})
