import argparse

from gymnasium.utils.env_checker import check_env

from envs.vehicle import VehicleEnv


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--check-env", action="store_true")
parser.add_argument("--seed", type=int, default=123)
args = parser.parse_args()

env = VehicleEnv()
if args.check_env:
    check_env(env, skip_render_check=True)
observation, _ = env.reset(seed=args.seed)
completed = 0
for step in range(args.steps):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        completed += 1
        observation, _ = env.reset(seed=args.seed + completed)
env.close()
print(f"OK - {args.steps} random VehicleEnv steps, {completed} completed episodes")
