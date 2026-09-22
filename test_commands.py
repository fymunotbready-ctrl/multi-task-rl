import numpy as np

from commands import COMMAND_MAP, resolve_command, run_command


class RecordingEnv:
    def __init__(self):
        self.motion_idx = None

    def reset(self, motion_idx=None):
        self.motion_idx = motion_idx
        return np.zeros(1, dtype=np.float32), {}

    def render(self):
        value = 255 if self.motion_idx is None else self.motion_idx
        return np.full((2, 2, 3), value, dtype=np.uint8)

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}


class ZeroModel:
    def predict(self, observation, deterministic):
        assert deterministic
        return np.zeros(28, dtype=np.float32), None


env = RecordingEnv()
model = ZeroModel()
checks = 0
for command, motion_idx in COMMAND_MAP.items():
    for variant in (command, command.upper(), f"  {command}  "):
        assert resolve_command(variant) == motion_idx
        frames = run_command(model, env, variant)
        assert env.motion_idx == motion_idx
        assert len(frames) == 2
        checks += 1

for command in ("jump", "shoot", ""):
    try:
        resolve_command(command)
    except ValueError as error:
        message = str(error)
        assert "valid commands" in message
        assert all(valid in message for valid in COMMAND_MAP)
        checks += 1
    else:
        raise AssertionError(f"unknown command {command!r} was accepted")

expected_checks = 3 * len(COMMAND_MAP) + 3
assert checks == expected_checks
print(f"OK - {checks}/{expected_checks} command and unknown-command routing checks passed")
