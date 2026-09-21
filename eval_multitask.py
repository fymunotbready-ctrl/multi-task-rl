import sys
sys.path.insert(0, ".")
from stable_baselines3 import PPO
from envs.meta_env import TASK_NAMES
from envs.basketball.basketball_env import BasketballEnv
from envs.driving.driving_env import DrivingEnv
from envs.aiming.aiming_env import AimingEnv
from envs.meta_env import encode_obs

EVAL_ENVS = [BasketballEnv(), DrivingEnv(), AimingEnv()]


def task_success(model, env, idx, n=100):
    hits = 0
    for i in range(n):
        obs, _ = env.reset(seed=999_000 + i)
        obs = encode_obs(obs, idx)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            obs = encode_obs(obs, idx)
            done = term or trunc
        if term and r >= 0.99:
            hits += 1
    return hits / n


path = sys.argv[1] if len(sys.argv) > 1 else "models/joint_trunk_final_v2.zip"
model = PPO.load(path)
for i, name in enumerate(TASK_NAMES):
    print(f"JOINT  {name:12s} success: {task_success(model, EVAL_ENVS[i], i)*100:5.1f}%  ({path})")
