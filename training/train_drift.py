import argparse
from pathlib import Path
import sys

sys.path.insert(0, ".")

import torch
from stable_baselines3 import PPO

from envs.vehicle import VehicleEnv


torch.set_num_threads(1)
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=498_000)
parser.add_argument("--output", default="models/drift_ppo")
parser.add_argument("--seed", type=int, default=409)
parser.add_argument("--rear-friction", type=float, default=0.3)
parser.add_argument("--front-friction", type=float, default=1.2)
parser.add_argument("--motor-force", type=float, default=20.0)
args = parser.parse_args()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
env = VehicleEnv(
    rear_friction=args.rear_friction,
    front_friction=args.front_friction,
    motor_force=args.motor_force,
)
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    seed=args.seed,
    verbose=1,
)
model.learn(total_timesteps=args.steps)
model.save(args.output)
env.close()
print(f"actual timesteps: {model.num_timesteps}")
print(f"model -> {args.output}.zip")
