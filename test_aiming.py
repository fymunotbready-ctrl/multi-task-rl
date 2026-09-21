import numpy as np

from envs.aiming.aiming_env import AimingEnv, HIT_REWARD


env = AimingEnv()
obs, _ = env.reset(seed=123)
assert obs.shape == env.observation_space.shape

total_reward = 0.0
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(
        np.array([-0.65, 0.675], dtype=np.float32)
    )
    total_reward += reward
    done = terminated or truncated

assert info["success"]
assert terminated and not truncated
assert reward == HIT_REWARD
assert total_reward > 4.0
print(f"OK - deterministic hit earned {total_reward:.3f}")
