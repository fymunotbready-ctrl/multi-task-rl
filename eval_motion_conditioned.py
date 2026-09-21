import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from envs.ragdoll import RagdollEnv
from multi_motion import (
    MOTION_ENV_KWARGS,
    MOTION_LABELS,
    MOTION_NAMES,
    MOTION_PATHS,
    UNDISTURBED_MOTION_ENV_KWARGS,
)


torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/motion_conditioned_ppo.zip")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--fidelity-episodes", type=int, default=20)
parser.add_argument("--seed-start", type=int, default=10_000)
parser.add_argument("--visual-dir", default="assets")
parser.add_argument("--results-json")
args = parser.parse_args()

motions = tuple(np.load(path) for path in MOTION_PATHS)
model = PPO.load(args.model)


def action_for(policy, observation, rng):
    if policy == "ppo":
        return model.predict(observation, deterministic=True)[0]
    if policy == "zero":
        return np.zeros(28, dtype=np.float32)
    return rng.uniform(-1.0, 1.0, 28).astype(np.float32)


def evaluate(motion_idx, policy, env_kwargs, capture=False):
    env = RagdollEnv(render_mode="rgb_array" if capture else None, **env_kwargs)
    completed = 0
    rewards = []
    lengths = []
    saved_frames = []
    saved_labels = []
    try:
        for episode in range(args.episodes):
            seed = args.seed_start + episode
            observation, _ = env.reset(seed=seed, motion_idx=motion_idx)
            rng = np.random.default_rng(seed + 10_000)
            push_start = env._pushes[0][0] if env._pushes else -1
            capture_at = {
                push_start - 1: "before push",
                push_start + 2: "during push",
                push_start + 10: "after push",
                len(env.reference_motion) - 1: "motion end",
            }
            episode_frames = [env.render()] if capture and not saved_frames else []
            episode_labels = ["motion start"] if episode_frames else []
            done = False
            length = 0
            info = {}
            while not done:
                action = action_for(policy, observation, rng)
                observation, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                length += 1
                done = terminated or truncated
                if capture and not saved_frames and env.frame_idx in capture_at:
                    episode_frames.append(env.render())
                    episode_labels.append(capture_at[env.frame_idx])
            completed += int(info.get("completed", False))
            lengths.append(length)
            if capture and info.get("completed", False) and not saved_frames:
                saved_frames = episode_frames
                saved_labels = episode_labels
    finally:
        env.close()
    return {
        "completed": completed,
        "reward": float(np.mean(rewards)),
        "length": float(np.mean(lengths)),
        "frames": saved_frames,
        "labels": saved_labels,
    }


def fidelity(motion_idx):
    env = RagdollEnv(**MOTION_ENV_KWARGS)
    correct = 0
    errors = np.zeros(len(motions), dtype=np.float64)
    try:
        for episode in range(args.fidelity_episodes):
            observation, _ = env.reset(
                seed=args.seed_start + episode, motion_idx=motion_idx
            )
            trajectory = []
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(action)
                trajectory.append(env.get_joint_angles())
                done = terminated or truncated
            trajectory = np.asarray(trajectory)
            episode_errors = np.array(
                [
                    np.mean(np.abs(trajectory - motion[: len(trajectory)]))
                    for motion in motions
                ]
            )
            errors += episode_errors
            correct += int(np.argmin(episode_errors) == motion_idx)
    finally:
        env.close()
    return correct, errors / args.fidelity_episodes


def save_visual(name, result):
    if not result["frames"]:
        print(f"visual,{name}: no completed disturbed PPO episode")
        return
    path = Path(args.visual_dir) / f"{name}_conditioned_eval.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1, len(result["frames"]), figsize=(4 * len(result["frames"]), 4)
    )
    for axis, frame, label in zip(
        np.atleast_1d(axes), result["frames"], result["labels"]
    ):
        axis.imshow(frame)
        axis.set_title(label, fontsize=10, pad=4)
        axis.axis("off")
    figure.subplots_adjust(
        left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.02
    )
    figure.savefig(path, dpi=120)
    plt.close(figure)
    print(f"visual,{name}: {path}")


results = {}
for motion_idx, name in enumerate(MOTION_NAMES):
    for setting, env_kwargs in (
        ("disturbed", MOTION_ENV_KWARGS),
        ("undisturbed", UNDISTURBED_MOTION_ENV_KWARGS),
    ):
        for policy in ("ppo", "zero", "random"):
            result = evaluate(
                motion_idx,
                policy,
                env_kwargs,
                capture=setting == "disturbed" and policy == "ppo",
            )
            results[(name, setting, policy)] = result
            print(
                f"{name},{setting},{policy}: "
                f"completed={result['completed']}/{args.episodes} "
                f"({100 * result['completed'] / args.episodes:.1f}%), "
                f"mean_reward={result['reward']:.6f}, "
                f"mean_length={result['length']:.2f}"
            )
    save_visual(name, results[(name, "disturbed", "ppo")])

fidelity_results = {}
for motion_idx, name in enumerate(MOTION_NAMES):
    correct, errors = fidelity(motion_idx)
    fidelity_results[name] = {
        "correct": correct,
        "episodes": args.fidelity_episodes,
        "errors": {
            candidate: float(error)
            for candidate, error in zip(MOTION_NAMES, errors)
        },
    }
    formatted_errors = ",".join(
        f"{candidate}={error:.6f}"
        for candidate, error in zip(MOTION_NAMES, errors)
    )
    print(
        f"fidelity,{name}: correct={correct}/{args.fidelity_episodes} "
        f"({100 * correct / args.fidelity_episodes:.1f}%), errors={formatted_errors}"
    )

gate_results = {}
for name in MOTION_NAMES:
    disturbed_ppo = results[(name, "disturbed", "ppo")]["completed"]
    disturbed_zero = results[(name, "disturbed", "zero")]["completed"]
    undisturbed_ppo = results[(name, "undisturbed", "ppo")]["completed"]
    passed = (
        100 * undisturbed_ppo / args.episodes >= 90.0
        and disturbed_ppo >= disturbed_zero
    )
    gate_results[name] = {
        "disturbed_gap_pp": 100
        * (disturbed_ppo - disturbed_zero)
        / args.episodes,
        "passed": passed,
    }
    print(
        f"gate,{MOTION_LABELS[name]}: disturbed_gap="
        f"{100 * (disturbed_ppo - disturbed_zero) / args.episodes:.1f}pp, "
        f"passed={passed}"
    )

if args.results_json:
    report = {
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "results": {
            f"{name},{setting},{policy}": {
                key: value
                for key, value in result.items()
                if key in ("completed", "reward", "length")
            }
            for (name, setting, policy), result in results.items()
        },
        "fidelity": fidelity_results,
        "gates": gate_results,
    }
    Path(args.results_json).write_text(json.dumps(report, indent=2) + "\n")
