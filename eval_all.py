import sys
sys.path.insert(0, ".")
from stable_baselines3 import PPO
from envs.basketball.basketball_env import BasketballEnv
from envs.driving.driving_env import DrivingEnv
from envs.aiming.aiming_env import AimingEnv


def success_rate(model, env, n=50, seed=999999):
    hits = 0
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)   # seeds the agent NEVER saw
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            if term and r >= 0.99:
                hits += 1
    return hits / n


tasks = [
    ("basketball", BasketballEnv, "models/basketball_ppo.zip"),
    ("driving",    DrivingEnv,    "models/driving_ppo.zip"),
    ("aiming",     AimingEnv,     "models/aiming_ppo.zip"),
]

for name, env_cls, path in tasks:
    model = PPO.load(path)
    rate = success_rate(model, env_cls())
    print(f"{name:12s} success rate: {rate*100:5.1f}%  ({path})")
