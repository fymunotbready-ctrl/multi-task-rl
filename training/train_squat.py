import argparse
import csv
import sys

sys.path.insert(0, ".")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from envs.ragdoll import RagdollEnv


torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=100_000)
parser.add_argument("--motion", default="motions/squat.npy")
parser.add_argument("--output", default="models/squat_ppo")
args = parser.parse_args()


class EpisodeLogger(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        with open(path, "w", newline="") as file:
            csv.writer(file, lineterminator="\n").writerow(
                ["steps", "episode_reward", "episode_length", "completed"]
            )

    def _on_step(self):
        rows = []
        for info in self.locals.get("infos", []):
            if "episode" in info:
                rows.append(
                    [
                        self.num_timesteps,
                        info["episode"]["r"],
                        info["episode"]["l"],
                        bool(info.get("completed", False)),
                    ]
                )
        if rows:
            with open(self.path, "a", newline="") as file:
                csv.writer(file, lineterminator="\n").writerows(rows)
        return True


env = RagdollEnv(reference_motion=args.motion)
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=128,
    seed=17,
)
print(f"=== squat imitation: {args.steps} timesteps ===")
model.learn(total_timesteps=args.steps, callback=EpisodeLogger("logs/squat_train.csv"))
model.save(args.output)
env.close()
print(f"model -> {args.output}.zip")
print("log -> logs/squat_train.csv")
