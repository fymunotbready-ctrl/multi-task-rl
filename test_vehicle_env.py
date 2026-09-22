import math

import numpy as np
from gymnasium.utils.env_checker import check_env

from envs.vehicle import VehicleEnv


env = VehicleEnv()
check_env(env, skip_render_check=True)
observation, _ = env.reset(seed=17)
assert observation.shape == (18,)
assert env.action_space.shape == (2,)

for _ in range(80):
    observation, reward, terminated, truncated, info = env.step(
        np.array([0.8, 1.0], dtype=np.float32)
    )
    assert np.isfinite(observation).all()
    assert math.isfinite(reward)
    assert 0.0 <= info["slip_angle"] <= math.pi / 2.0
    if terminated or truncated:
        break

env.close()
print("OK - VehicleEnv passes check_env and bounded-action physics smoke checks")
