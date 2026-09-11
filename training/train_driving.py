import sys
sys.path.insert(0, ".")
from envs.driving.driving_env import DrivingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = DrivingEnv()
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
            n_steps=1024, batch_size=64, seed=7)

print("=== Training 50k steps ===")
model.learn(total_timesteps=50_000)
model.save("driving_ppo")

mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=100)
print(f"\nFinal: avg reward {mean_r:.3f} +/- {std_r:.3f} over 100 episodes")
print("(~1.0 = reaches target nearly every time)")
