import sys
sys.path.insert(0, ".")
from stable_baselines3 import PPO
from envs.basketball.basketball_env import BasketballEnv
from envs.driving.driving_env import DrivingEnv
from envs.aiming.aiming_env import AimingEnv
from envs.meta_env import add_task_id


def success_rate(model, env, idx, n=50, seed=999999):
    hits = 0
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        obs = add_task_id(obs, idx)          # same 9-dim obs as training
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            obs = add_task_id(obs, idx)
            done = term or trunc
            if term and r >= 0.99:           # RAW reward (scaling is train-only)
                hits += 1
    return hits / n


model = PPO.load("models/multitask_ppo.zip")
tasks = [(0, "basketball", BasketballEnv),
         (1, "driving", DrivingEnv),
         (2, "aiming", AimingEnv)]
for idx, name, env_cls in tasks:
    rate = success_rate(model, env_cls(), idx)
    print(f"SHARED MODEL  {name:12s} success: {rate*100:5.1f}%")
