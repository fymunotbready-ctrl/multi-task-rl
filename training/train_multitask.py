import sys
sys.path.insert(0, ".")
from envs.meta_env import MultiTaskEnv
from stable_baselines3 import PPO

env = MultiTaskEnv()
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
            n_steps=1024, batch_size=64, seed=3)
print("=== Training SHARED model: one brain, 3 skills, 300k steps ===")
model.learn(total_timesteps=300_000)
model.save("multitask_ppo")
print("saved multitask_ppo")
