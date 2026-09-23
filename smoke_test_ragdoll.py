import argparse

import numpy as np
from stable_baselines3.common.env_checker import check_env

from envs.ragdoll import RagdollEnv


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--check-env", action="store_true")
args = parser.parse_args()

if args.steps <= 0:
    parser.error("--steps must be positive")

env = RagdollEnv(max_steps=args.steps)
try:
    if args.check_env:
        check_env(env, warn=True)
        print("check_env: passed")

    env.action_space.seed(42)
    observation, _ = env.reset(seed=42)
    initial_observation = observation.copy()
    for _ in range(args.steps):
        action = env.action_space.sample()
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    delta = float(np.linalg.norm(observation - initial_observation))
    print(f"steps_run: {env.steps}")
    print(f"no_errors: {env.steps == args.steps}")
    print(f"observation_shape: {observation.shape}")
    print(f"action_shape: {action.shape}")
    print(f"observation_delta_norm: {delta:.6f}")
finally:
    env.close()
