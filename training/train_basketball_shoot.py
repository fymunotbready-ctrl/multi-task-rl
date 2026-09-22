import argparse
import sys

sys.path.insert(0, ".")

import torch
from stable_baselines3 import PPO

from basketball_shoot import BASKETBALL_ENV_KWARGS
from envs.ragdoll import BasketballShootEnv


torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=100_000)
parser.add_argument("--input-model", default="models/motion_conditioned_ppo.zip")
parser.add_argument("--output", default="models/basketball_shoot_ppo")
parser.add_argument("--seed", type=int, default=113)
args = parser.parse_args()

if not args.input_model:
    parser.error("--input-model is required; Stage 1 fine-tunes the Phase 4 policy")

env = BasketballShootEnv(**BASKETBALL_ENV_KWARGS)
model = PPO.load(args.input_model, env=env)
model.set_random_seed(args.seed)
print(
    f"=== basketball shoot fine-tune: {args.steps} requested timesteps, "
    f"seed {args.seed}, starting at {model.num_timesteps} ==="
)
model.learn(total_timesteps=args.steps, reset_num_timesteps=False)
model.save(args.output)
env.close()
print(f"actual cumulative timesteps: {model.num_timesteps}")
print(f"model -> {args.output}.zip")
