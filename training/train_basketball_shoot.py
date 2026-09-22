import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from basketball_shoot import BASKETBALL_ENV_KWARGS
from envs.ragdoll import BasketballShootEnv


torch.set_num_threads(1)


class BasketballTrainingCallback(BaseCallback):
    def __init__(
        self,
        output,
        starting_timesteps,
        total_additional_timesteps,
        checkpoint_steps,
        hoop_scale_start,
        hoop_scale_end,
    ):
        super().__init__()
        self.output = output
        self.starting_timesteps = starting_timesteps
        self.total_additional_timesteps = total_additional_timesteps
        self.checkpoint_steps = checkpoint_steps
        self.hoop_scale_start = hoop_scale_start
        self.hoop_scale_end = hoop_scale_end
        self.checkpoint_saved = False

    def _on_step(self):
        additional = self.model.num_timesteps - self.starting_timesteps
        if self.n_calls == 1 or additional % 2048 == 0:
            progress = min(1.0, additional / self.total_additional_timesteps)
            hoop_scale = self.hoop_scale_start + progress * (
                self.hoop_scale_end - self.hoop_scale_start
            )
            self.training_env.env_method("set_hoop_radius_scale", hoop_scale)
        if (
            self.checkpoint_steps
            and not self.checkpoint_saved
            and additional > self.checkpoint_steps
        ):
            checkpoint_path = f"{self.output}_step{self.checkpoint_steps}"
            self.model.save(checkpoint_path)
            self.checkpoint_saved = True
            print(f"intermediate checkpoint -> {checkpoint_path}.zip")
        return True


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=498_000)
parser.add_argument("--input-model", default="models/motion_conditioned_ppo.zip")
parser.add_argument("--output", default="models/basketball_shoot_stage1b")
parser.add_argument("--seed", type=int, default=113)
parser.add_argument("--shot-weight", type=float, default=30.0)
parser.add_argument("--progress-scale", type=float, default=1.0)
parser.add_argument("--release-quality-scale", type=float, default=0.5)
parser.add_argument("--ballistic-distance-sigma", type=float, default=0.75)
parser.add_argument("--hoop-scale-start", type=float, default=1.0)
parser.add_argument("--hoop-scale-end", type=float, default=1.0)
parser.add_argument("--checkpoint-steps", type=int, default=249_856)
args = parser.parse_args()

if not args.input_model:
    parser.error("--input-model is required; Stage 1b fine-tunes the Phase 4 policy")
if args.hoop_scale_start < args.hoop_scale_end:
    parser.error("--hoop-scale-start must be at least --hoop-scale-end")
if args.hoop_scale_end < 1.0:
    parser.error("--hoop-scale-end cannot be smaller than the real rim")

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
env = BasketballShootEnv(
    **BASKETBALL_ENV_KWARGS,
    shot_outcome_weight=args.shot_weight,
    progress_reward_scale=args.progress_scale,
    release_quality_scale=args.release_quality_scale,
    ballistic_distance_sigma=args.ballistic_distance_sigma,
    hoop_radius_scale=args.hoop_scale_start,
)
model = PPO.load(args.input_model, env=env)
model.set_random_seed(args.seed)
starting_timesteps = model.num_timesteps
callback = BasketballTrainingCallback(
    output=args.output,
    starting_timesteps=starting_timesteps,
    total_additional_timesteps=args.steps,
    checkpoint_steps=args.checkpoint_steps,
    hoop_scale_start=args.hoop_scale_start,
    hoop_scale_end=args.hoop_scale_end,
)
print(
    f"=== basketball shoot fine-tune: {args.steps} requested timesteps, "
    f"seed {args.seed}, starting at {starting_timesteps}, "
    f"shot weight {args.shot_weight}, progress {args.progress_scale}, "
    f"release quality {args.release_quality_scale}, "
    f"hoop {args.hoop_scale_start}x->{args.hoop_scale_end}x ==="
)
model.learn(total_timesteps=args.steps, reset_num_timesteps=False, callback=callback)
model.save(args.output)
env.close()
actual_additional = model.num_timesteps - starting_timesteps
print(f"actual additional timesteps: {actual_additional}")
print(f"actual cumulative timesteps: {model.num_timesteps}")
print(f"model -> {args.output}.zip")
