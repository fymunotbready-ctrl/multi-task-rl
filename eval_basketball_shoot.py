import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from basketball_shoot import BASKETBALL_ENV_KWARGS
from envs.ragdoll import BasketballShootEnv


torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/basketball_shoot_ppo.zip")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--seed-start", type=int, default=10_000)
parser.add_argument("--visual", default="assets/basketball_shoot_made.png")
parser.add_argument("--results-json")
parser.add_argument("--skip-zero", action="store_true")
args = parser.parse_args()

model = PPO.load(args.model)


def save_visual(frames, labels):
    if not frames or not args.visual:
        return None
    path = Path(args.visual)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 4))
    for axis, frame, label in zip(np.atleast_1d(axes), frames, labels):
        axis.imshow(frame)
        axis.set_title(label, fontsize=10)
        axis.axis("off")
    figure.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.02)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return str(path)


def evaluate(policy, capture_made=False):
    made = 0
    completed = 0
    release_speeds = []
    release_angles = []
    imitation_rewards = []
    minimum_distances = []
    release_alignments = []
    release_qualities = []
    predicted_distances = []
    saved_frames = []
    saved_labels = []
    for episode in range(args.episodes):
        capture = capture_made and not saved_frames
        env = BasketballShootEnv(
            render_mode="rgb_array" if capture else None,
            **BASKETBALL_ENV_KWARGS,
        )
        episode_frames = []
        try:
            observation, _ = env.reset(seed=args.seed_start + episode)
            done = False
            info = {}
            while not done:
                if policy == "ppo":
                    action, _ = model.predict(observation, deterministic=True)
                else:
                    action = np.zeros(28, dtype=np.float32)
                observation, _, terminated, truncated, info = env.step(action)
                imitation_rewards.append(info["imitation_reward"])
                if capture and env.frame_idx in (1, 55, 65, 72, 80, 90, 110):
                    episode_frames.append((env.frame_idx, env.render()))
                done = terminated or truncated
            made += int(info["made"])
            completed += int(info["completed"])
            if info["ball_released"]:
                release_speeds.append(info["release_speed"])
                release_angles.append(info["release_angle_degrees"])
                minimum_distances.append(info["min_hoop_distance"])
                release_alignments.append(info["release_alignment"])
                release_qualities.append(info["release_quality"])
                predicted_distances.append(info["predicted_closest_distance"])
            if capture and info["made"]:
                saved_frames = [frame for _, frame in episode_frames]
                saved_labels = [f"frame {frame_idx}" for frame_idx, _ in episode_frames]
        finally:
            env.close()
    return {
        "made": made,
        "completed": completed,
        "episodes": args.episodes,
        "mean_release_speed": float(np.mean(release_speeds)),
        "mean_release_angle_degrees": float(np.mean(release_angles)),
        "mean_imitation_reward": float(np.mean(imitation_rewards)),
        "mean_min_hoop_distance": float(np.mean(minimum_distances)),
        "mean_release_alignment": float(np.mean(release_alignments)),
        "mean_release_quality": float(np.mean(release_qualities)),
        "mean_predicted_closest_distance": float(np.mean(predicted_distances)),
        "frames": saved_frames,
        "labels": saved_labels,
    }


results = {}
for policy in ("ppo",) if args.skip_zero else ("ppo", "zero"):
    result = evaluate(policy, capture_made=policy == "ppo")
    results[policy] = result
    print(
        f"{policy}: made={result['made']}/{result['episodes']} "
        f"({100 * result['made'] / result['episodes']:.1f}%), "
        f"completed={result['completed']}/{result['episodes']} "
        f"({100 * result['completed'] / result['episodes']:.1f}%), "
        f"release_speed={result['mean_release_speed']:.3f}, "
        f"release_angle={result['mean_release_angle_degrees']:.2f}deg, "
        f"alignment={result['mean_release_alignment']:.3f}, "
        f"predicted_distance={result['mean_predicted_closest_distance']:.3f}, "
        f"imitation_reward={result['mean_imitation_reward']:.6f}, "
        f"min_distance={result['mean_min_hoop_distance']:.3f}"
    )

visual = save_visual(results["ppo"].pop("frames"), results["ppo"].pop("labels"))
if "zero" in results:
    results["zero"].pop("frames")
    results["zero"].pop("labels")
print(f"visual: {visual or 'no made PPO episode'}")

if args.results_json:
    report = {
        "seed_start": args.seed_start,
        "made_definition": (
            "ball center crosses downward through the rim plane within "
            "hoop_radius - ball_radius of the hoop center"
        ),
        "visual": visual,
        "results": results,
    }
    Path(args.results_json).write_text(json.dumps(report, indent=2) + "\n")
