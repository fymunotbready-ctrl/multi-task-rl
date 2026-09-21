import argparse
import sys

sys.path.insert(0, ".")

import torch
from stable_baselines3 import PPO

from envs.ragdoll import RagdollEnv
from multi_motion import MOTION_ENV_KWARGS


torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=600_000)
parser.add_argument("--output", default="models/motion_conditioned_ppo")
parser.add_argument("--force-max", type=float, default=1200.0)
parser.add_argument("--input-model")
parser.add_argument("--seed", type=int, default=97)
args = parser.parse_args()

env_kwargs = {
    **MOTION_ENV_KWARGS,
    "push_force_range": (0.0, args.force_max),
}
env = RagdollEnv(**env_kwargs)
if args.input_model:
    model = PPO.load(args.input_model, env=env)
    model.set_random_seed(args.seed)
else:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        seed=args.seed,
    )
print(
    f"=== conditioned motions: {args.steps} timesteps, seed {args.seed}, "
    f"starting at {model.num_timesteps} ==="
)
model.learn(total_timesteps=args.steps, reset_num_timesteps=not args.input_model)
model.save(args.output)
env.close()
print(f"model -> {args.output}.zip")
