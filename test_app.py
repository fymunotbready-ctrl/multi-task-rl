import inspect
import random

import numpy as np

from app import stream_rollout, surprise_command
from commands import COMMAND_MAP


class RecordingModel:
    def __init__(self):
        self.calls = 0

    def predict(self, observation, deterministic):
        assert deterministic
        self.calls += 1
        return np.zeros(28, dtype=np.float32), None


class RecordingEnv:
    def __init__(self):
        self.motion_idx = None
        self.steps = 0

    def reset(self, motion_idx):
        self.motion_idx = motion_idx
        self.steps = 0
        return np.zeros(1, dtype=np.float32), {}

    def render(self):
        return np.full((2, 2, 3), self.steps, dtype=np.uint8)

    def step(self, action):
        assert action.shape == (28,)
        self.steps += 1
        return np.zeros(1, dtype=np.float32), 0.0, False, self.steps == 3, {}


assert inspect.isgeneratorfunction(stream_rollout)
for command, motion_idx in COMMAND_MAP.items():
    model = RecordingModel()
    env = RecordingEnv()
    stream = stream_rollout(model, env, command, frame_interval=0.0)

    first = next(stream)
    assert env.motion_idx == motion_idx
    assert model.calls == 0
    assert np.all(first == 0)

    remaining = list(stream)
    assert model.calls == 3
    assert [int(frame[0, 0, 0]) for frame in remaining] == [1, 2, 3]

rng = random.Random(7)
choices = {surprise_command(rng) for _ in range(100)}
assert choices == set(COMMAND_MAP)

print("OK - frames stream step-by-step and Surprise me covers all commands")
