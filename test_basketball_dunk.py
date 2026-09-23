import numpy as np

from basketball_shoot import BASKETBALL_ENV_KWARGS
from envs.ragdoll import BasketballDunkEnv


env = BasketballDunkEnv(**BASKETBALL_ENV_KWARGS, jump_force=0.0)
observation, info = env.reset(seed=31)
assert observation.shape == (113,)
assert info["motion_idx"] == 2
assert not info["ever_airborne"]

last_info = info
while True:
    observation, reward, terminated, truncated, last_info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    assert np.isfinite(observation).all()
    assert np.isfinite(reward)
    if terminated or truncated:
        break
assert last_info["ball_released"]
assert last_info["peak_base_height"] >= 3.0
env.close()

jump_env = BasketballDunkEnv(**BASKETBALL_ENV_KWARGS, jump_force=1000.0)
observation, _ = jump_env.reset(seed=31)
jump_info = {}
while True:
    observation, reward, terminated, truncated, jump_info = jump_env.step(
        np.zeros(jump_env.action_space.shape, dtype=np.float32)
    )
    if terminated or truncated:
        break
assert jump_info["ever_airborne"]
assert jump_info["landed_after_jump"]
assert jump_info["peak_base_height"] > 3.5
jump_env.close()

try:
    BasketballDunkEnv(**BASKETBALL_ENV_KWARGS, jump_force=-1.0)
except ValueError:
    pass
else:
    raise AssertionError("negative jump force was accepted")

print("OK - dunk uses motion 2, physical ball release, hoop observations, and jump metrics")
