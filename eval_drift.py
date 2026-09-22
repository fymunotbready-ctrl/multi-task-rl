import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from envs.vehicle import VehicleEnv


def evaluate(policy, episodes, seed_start, env_kwargs, capture=False):
    env = VehicleEnv(render_mode="rgb_array" if capture else None, **env_kwargs)
    episode_metrics = []
    visual_frames = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed_start + episode)
        slips = []
        speeds = []
        safe = True
        frames = []
        done = False
        while not done:
            if policy is None:
                action = env.np_random.uniform(-1.0, 1.0, size=2).astype(np.float32)
            else:
                action, _ = policy.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            slips.append(info["slip_angle_degrees"])
            speeds.append(info["speed"])
            safe = safe and info["safe"]
            if capture and episode == 0 and len(frames) < 6 and env.steps % 50 == 0:
                frames.append(env.render())
            done = terminated or truncated
        active_speeds = [speed for speed in speeds if speed >= 1.0]
        active_slips = [slip for slip, speed in zip(slips, speeds) if speed >= 1.0]
        episode_metrics.append(
            {
                "mean_slip_degrees": float(np.mean(active_slips)) if active_slips else 0.0,
                "peak_slip_degrees": float(np.max(active_slips)) if active_slips else 0.0,
                "mean_drift_speed": float(np.mean(active_speeds)) if active_speeds else 0.0,
                "safe": safe,
            }
        )
        if frames and not visual_frames:
            visual_frames = frames
    env.close()
    return {
        "mean_slip_degrees": float(np.mean([m["mean_slip_degrees"] for m in episode_metrics])),
        "mean_peak_slip_degrees": float(np.mean([m["peak_slip_degrees"] for m in episode_metrics])),
        "mean_drift_speed": float(np.mean([m["mean_drift_speed"] for m in episode_metrics])),
        "safe_episodes": int(sum(m["safe"] for m in episode_metrics)),
        "episodes": episodes,
    }, visual_frames


def save_visual(frames, output):
    if not frames:
        return
    figure, axes = plt.subplots(1, len(frames), figsize=(3 * len(frames), 3))
    for axis, frame in zip(np.atleast_1d(axes), frames):
        axis.imshow(frame)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=120)
    plt.close(figure)


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/drift_ppo.zip")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--seed-start", type=int, default=10000)
parser.add_argument("--rear-friction", type=float, default=0.3)
parser.add_argument("--front-friction", type=float, default=1.2)
parser.add_argument("--motor-force", type=float, default=20.0)
parser.add_argument("--visual", default="assets/drift_rollout.png")
parser.add_argument("--results-json", default="")
args = parser.parse_args()

env_kwargs = {
    "rear_friction": args.rear_friction,
    "front_friction": args.front_friction,
    "motor_force": args.motor_force,
}
model = PPO.load(args.model)
ppo_metrics, frames = evaluate(model, args.episodes, args.seed_start, env_kwargs, True)
random_metrics, _ = evaluate(None, args.episodes, args.seed_start, env_kwargs)
results = {"seed_start": args.seed_start, "ppo": ppo_metrics, "random": random_metrics}
print(json.dumps(results, indent=2))
if args.visual:
    save_visual(frames, args.visual)
    print(f"visual: {args.visual}")
if args.results_json:
    Path(args.results_json).write_text(json.dumps(results, indent=2) + "\n")
