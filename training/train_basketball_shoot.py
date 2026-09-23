import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from basketball_shoot import BASKETBALL_ENV_KWARGS
from envs.ragdoll import BasketballDunkEnv, BasketballMissEnv, BasketballShootEnv


torch.set_num_threads(1)

def load_model_with_expanded_observation(input_model, env, seed):
    source_model = PPO.load(input_model)
    source_dim = source_model.observation_space.shape[0]
    target_dim = env.observation_space.shape[0]
    if source_dim == target_dim:
        source_model.set_env(env)
        source_model.set_random_seed(seed)
        return source_model
    if target_dim != source_dim + 4:
        raise ValueError(
            f"expected four new hoop-relative observations, got {source_dim}->{target_dim}"
        )

    model = PPO(
        source_model.policy_class,
        env,
        learning_rate=source_model.learning_rate,
        n_steps=source_model.n_steps,
        batch_size=source_model.batch_size,
        n_epochs=source_model.n_epochs,
        gamma=source_model.gamma,
        gae_lambda=source_model.gae_lambda,
        clip_range=source_model.clip_range,
        clip_range_vf=source_model.clip_range_vf,
        normalize_advantage=source_model.normalize_advantage,
        ent_coef=source_model.ent_coef,
        vf_coef=source_model.vf_coef,
        max_grad_norm=source_model.max_grad_norm,
        use_sde=source_model.use_sde,
        sde_sample_freq=source_model.sde_sample_freq,
        rollout_buffer_class=source_model.rollout_buffer_class,
        rollout_buffer_kwargs=source_model.rollout_buffer_kwargs,
        target_kl=source_model.target_kl,
        stats_window_size=source_model._stats_window_size,
        policy_kwargs=source_model.policy_kwargs,
        seed=seed,
        device=source_model.device,
        verbose=1,
    )
    source_state = source_model.policy.state_dict()
    target_state = model.policy.state_dict()
    expanded_layers = {
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.value_net.0.weight",
    }
    for name, target in target_state.items():
        source = source_state[name]
        if source.shape == target.shape:
            target.copy_(source)
        elif name in expanded_layers and target.shape[1] == source.shape[1] + 4:
            target.zero_()
            target[:, : source.shape[1]].copy_(source)
        else:
            raise ValueError(
                f"cannot transfer policy parameter {name}: {source.shape}->{target.shape}"
            )
    model.policy.load_state_dict(target_state)
    model.num_timesteps = source_model.num_timesteps
    print(f"expanded observation: {source_dim}->{target_dim}; new inputs zero-initialized")
    return model


class BasketballTrainingCallback(BaseCallback):
    def __init__(
        self,
        output,
        starting_timesteps,
        total_additional_timesteps,
        checkpoint_steps,
        hoop_scale_start,
        hoop_scale_end,
        release_imitation_weight_start,
        release_imitation_weight_end,
    ):
        super().__init__()
        self.output = output
        self.starting_timesteps = starting_timesteps
        self.total_additional_timesteps = total_additional_timesteps
        self.checkpoint_steps = checkpoint_steps
        self.hoop_scale_start = hoop_scale_start
        self.hoop_scale_end = hoop_scale_end
        self.release_imitation_weight_start = release_imitation_weight_start
        self.release_imitation_weight_end = release_imitation_weight_end
        self.checkpoint_saved = False

    def _on_step(self):
        additional = self.model.num_timesteps - self.starting_timesteps
        if self.n_calls == 1 or additional % 2048 == 0:
            progress = min(1.0, additional / self.total_additional_timesteps)
            hoop_scale = self.hoop_scale_start + progress * (
                self.hoop_scale_end - self.hoop_scale_start
            )
            imitation_weight = self.release_imitation_weight_start + progress * (
                self.release_imitation_weight_end
                - self.release_imitation_weight_start
            )
            self.training_env.env_method("set_hoop_radius_scale", hoop_scale)
            self.training_env.env_method(
                "set_release_imitation_weight", imitation_weight
            )
        if (
            self.checkpoint_steps
            and not self.checkpoint_saved
            and additional >= self.checkpoint_steps
        ):
            checkpoint_path = f"{self.output}_step{self.checkpoint_steps}"
            self.model.save(checkpoint_path)
            self.checkpoint_saved = True
            print(f"intermediate checkpoint -> {checkpoint_path}.zip")
        return True


parser = argparse.ArgumentParser()
parser.add_argument("--objective", choices=("shoot", "miss", "dunk"), default="shoot")
parser.add_argument("--steps", type=int, default=1_248_000)
parser.add_argument("--input-model", default="models/basketball_shoot_stage1b.zip")
parser.add_argument("--output", default="models/basketball_shoot_stage1c")
parser.add_argument("--seed", type=int, default=251)
parser.add_argument("--shot-weight", type=float, default=30.0)
parser.add_argument("--progress-scale", type=float, default=1.0)
parser.add_argument("--release-quality-scale", type=float, default=0.5)
parser.add_argument("--ballistic-distance-sigma", type=float, default=0.75)
parser.add_argument("--hoop-scale-start", type=float, default=1.0)
parser.add_argument("--hoop-scale-end", type=float, default=1.0)
parser.add_argument("--release-window-frames", type=int, default=25)
parser.add_argument("--release-imitation-weight-start", type=float, default=1.0)
parser.add_argument("--release-imitation-weight-end", type=float, default=0.1)
parser.add_argument("--start-position-randomization", type=float, default=0.0)
parser.add_argument("--start-yaw-randomization-degrees", type=float, default=0.0)
parser.add_argument("--checkpoint-steps", type=int, default=624_640)
parser.add_argument("--jump-force", type=float, default=0.0)
args = parser.parse_args()

if not args.input_model:
    parser.error("--input-model is required; Stage 1c fine-tunes Stage 1b")
if args.hoop_scale_start < args.hoop_scale_end:
    parser.error("--hoop-scale-start must be at least --hoop-scale-end")
if args.hoop_scale_end < 1.0:
    parser.error("--hoop-scale-end cannot be smaller than the real rim")
if not 0.0 <= args.release_imitation_weight_end <= 1.0:
    parser.error("--release-imitation-weight-end must be in [0, 1]")
if not 0.0 <= args.release_imitation_weight_start <= 1.0:
    parser.error("--release-imitation-weight-start must be in [0, 1]")
if args.release_imitation_weight_start < args.release_imitation_weight_end:
    parser.error(
        "--release-imitation-weight-start must be at least "
        "--release-imitation-weight-end"
    )

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
env_classes = {
    "shoot": BasketballShootEnv,
    "miss": BasketballMissEnv,
    "dunk": BasketballDunkEnv,
}
env_class = env_classes[args.objective]
env_kwargs = {
    **BASKETBALL_ENV_KWARGS,
    "shot_outcome_weight": args.shot_weight,
    "progress_reward_scale": args.progress_scale,
    "release_quality_scale": args.release_quality_scale,
    "ballistic_distance_sigma": args.ballistic_distance_sigma,
    "hoop_radius_scale": args.hoop_scale_start,
    "release_imitation_weight": args.release_imitation_weight_start,
    "release_window_frames": args.release_window_frames,
    "start_position_randomization": args.start_position_randomization,
    "start_yaw_randomization_degrees": args.start_yaw_randomization_degrees,
}
if args.objective == "dunk":
    env_kwargs["jump_force"] = args.jump_force
env = env_class(**env_kwargs)
model = load_model_with_expanded_observation(args.input_model, env, args.seed)
starting_timesteps = model.num_timesteps
callback = BasketballTrainingCallback(
    output=args.output,
    starting_timesteps=starting_timesteps,
    total_additional_timesteps=args.steps,
    checkpoint_steps=args.checkpoint_steps,
    hoop_scale_start=args.hoop_scale_start,
    hoop_scale_end=args.hoop_scale_end,
    release_imitation_weight_start=args.release_imitation_weight_start,
    release_imitation_weight_end=args.release_imitation_weight_end,
)
print(
    f"=== basketball {args.objective} fine-tune: {args.steps} requested timesteps, "
    f"seed {args.seed}, starting at {starting_timesteps}, "
    f"shot weight {args.shot_weight}, progress {args.progress_scale}, "
    f"release quality {args.release_quality_scale}, "
    f"release imitation {args.release_imitation_weight_start}"
    f"->{args.release_imitation_weight_end} over frames "
    f"{env_class.RELEASE_FRAME - args.release_window_frames}"
    f"-{env_class.RELEASE_FRAME}, "
    f"start randomization +/-{args.start_position_randomization}m, "
    f"+/-{args.start_yaw_randomization_degrees}deg, "
    f"hoop {args.hoop_scale_start}x->{args.hoop_scale_end}x, "
    f"jump force {args.jump_force}N ==="
)
model.learn(total_timesteps=args.steps, reset_num_timesteps=False, callback=callback)
model.save(args.output)
env.close()
actual_additional = model.num_timesteps - starting_timesteps
print(f"actual additional timesteps: {actual_additional}")
print(f"actual cumulative timesteps: {model.num_timesteps}")
print(f"model -> {args.output}.zip")
