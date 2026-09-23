import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from basketball_shoot import BASKETBALL_ENV_KWARGS
from envs.ragdoll import BasketballDunkEnv, BasketballMissEnv, BasketballShootEnv


torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("--objective", choices=("shoot", "miss", "dunk"), default="shoot")
parser.add_argument("--model", default="models/basketball_shoot_ppo.zip")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--seed-start", type=int, default=10_000)
parser.add_argument("--visual", default="assets/basketball_shoot_made.png")
parser.add_argument("--results-json")
parser.add_argument("--skip-zero", action="store_true")
parser.add_argument(
    "--jump-force", type=float, default=BasketballDunkEnv.DEFAULT_JUMP_FORCE
)
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
    released = 0
    intentional_misses = 0
    completed = 0
    airborne_episodes = 0
    landed_episodes = 0
    peak_base_heights = []
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
        env_classes = {
            "shoot": BasketballShootEnv,
            "miss": BasketballMissEnv,
            "dunk": BasketballDunkEnv,
        }
        env_kwargs = dict(BASKETBALL_ENV_KWARGS)
        if args.objective == "dunk":
            env_kwargs["jump_force"] = args.jump_force
        env = env_classes[args.objective](
            render_mode="rgb_array" if capture else None,
            **env_kwargs,
        )
        episode_frames = []
        try:
            observation, _ = env.reset(seed=args.seed_start + episode)
            done = False
            info = {}
            while not done:
                if policy == "ppo":
                    policy_observation = observation[
                        : model.observation_space.shape[0]
                    ]
                    action, _ = model.predict(
                        policy_observation, deterministic=True
                    )
                else:
                    action = np.zeros(28, dtype=np.float32)
                observation, _, terminated, truncated, info = env.step(action)
                imitation_rewards.append(info["imitation_reward"])
                if capture and env.frame_idx in (1, 32, 44, 55, 65, 72, 90, 110):
                    episode_frames.append((env.frame_idx, env.render()))
                done = terminated or truncated
            made += int(info["made"])
            released += int(info["ball_released"])
            intentional_misses += int(info["ball_released"] and not info["made"])
            completed += int(info["completed"])
            airborne_episodes += int(info.get("ever_airborne", False))
            landed_episodes += int(info.get("landed_after_jump", False))
            if "peak_base_height" in info:
                peak_base_heights.append(info["peak_base_height"])
            if info["ball_released"]:
                release_speeds.append(info["release_speed"])
                release_angles.append(info["release_angle_degrees"])
                minimum_distances.append(info["min_hoop_distance"])
                release_alignments.append(info["release_alignment"])
                release_qualities.append(info["release_quality"])
                predicted_distances.append(info["predicted_closest_distance"])
            if args.objective == "miss":
                evidence_outcome = info["ball_released"] and not info["made"]
            else:
                evidence_outcome = info["made"]
            if capture and evidence_outcome:
                saved_frames = [frame for _, frame in episode_frames]
                saved_labels = [f"frame {frame_idx}" for frame_idx, _ in episode_frames]
        finally:
            env.close()
    return {
        "made": made,
        "completed": completed,
        "released": released,
        "intentional_misses": intentional_misses,
        "episodes": args.episodes,
        "airborne_episodes": airborne_episodes,
        "landed_episodes": landed_episodes,
        "mean_peak_base_height": (
            float(np.mean(peak_base_heights)) if peak_base_heights else 0.0
        ),
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
        f"intentional_misses={result['intentional_misses']}/{result['episodes']} "
        f"({100 * result['intentional_misses'] / result['episodes']:.1f}%), "
        f"released={result['released']}/{result['episodes']}, "
        f"completed={result['completed']}/{result['episodes']} "
        f"({100 * result['completed'] / result['episodes']:.1f}%), "
        f"airborne={result['airborne_episodes']}/{result['episodes']}, "
        f"landed={result['landed_episodes']}/{result['episodes']}, "
        f"peak_height={result['mean_peak_base_height']:.3f}, "
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
print(f"visual: {visual or 'no captured PPO rollout'}")

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
