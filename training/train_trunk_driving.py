import sys

sys.path.insert(0, ".")

import torch
from stable_baselines3 import PPO

from envs.driving.driving_env import DrivingEnv
from shared.shared_trunk import save_trunk, trunk_policy_kwargs

torch.set_num_threads(1)
STEPS = 150_000

env = DrivingEnv()
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    seed=7,
    policy_kwargs=trunk_policy_kwargs("driving"),
)

print(f"=== driving builds the trunk ({STEPS} steps) ===")
model.learn(total_timesteps=STEPS)
model.save("models/trunk_driving")
save_trunk(model, "models/trunk_driving.pt")

reached = 0
for i in range(100):
    obs, _ = env.reset(seed=1000 + i)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    reached += int(info.get("success", False))
print(f"DRIVING SUCCESS: {reached}/100")
