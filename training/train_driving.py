import sys
sys.path.insert(0, ".")
from envs.driving.driving_env import DrivingEnv
from stable_baselines3 import PPO

env = DrivingEnv()
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
            n_steps=1024, batch_size=64, seed=7)
print("=== driving v3: 100k steps ===")
model.learn(total_timesteps=100_000)
model.save("models/driving_ppo")

reached = 0
for i in range(100):
    obs, _ = env.reset(seed=1000 + i)
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
    if term and r >= 0.99:
        reached += 1
print(f"\nDRIVING GATE: {reached}/100  (need >= 50, expect ~100)")
