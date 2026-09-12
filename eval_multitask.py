import sys
sys.path.insert(0, ".")
from stable_baselines3 import PPO
from envs.basketball.basketball_env import BasketballEnv
from envs.driving.driving_env import DrivingEnv
from envs.aiming.aiming_env import AimingEnv


def success_rate(model, env, n=50, seed=999999):
    hits = 0
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            if term and r >= 0.99:
                hits += 1
    return hits / n


model = PPO.load("models/multitask_ppo.zip")
for name, env_cls in [("basketball", BasketballEnv),
                      ("driving", DrivingEnv),
                      ("aiming", AimingEnv)]:
    rate = success_rate(model, env_cls())
    print(f"SHARED MODEL  {name:12s} success: {rate*100:5.1f}%")
