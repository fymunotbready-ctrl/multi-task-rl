import sys
sys.path.insert(0, ".")
from envs.basketball.basketball_env import BasketballEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = BasketballEnv()
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
            n_steps=1024, batch_size=64, seed=42)

print("=== Training 150k steps ===")
model.learn(total_timesteps=150_000)
model.save("basketball_ppo")

mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=100)
print(f"\nFinal: avg reward {mean_r:.3f} +/- {std_r:.3f} over 100 shots")
