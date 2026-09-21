import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from disturbed_squat import DISTURBED_ENV_KWARGS, UNDISTURBED_ENV_KWARGS
from envs.ragdoll import RagdollEnv

torch.set_num_threads(1)


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/squat_disturbed_ppo.zip")
parser.add_argument("--motion", default="motions/squat.npy")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument(
    "--settings",
    choices=("disturbed", "undisturbed"),
    nargs="+",
    default=("disturbed", "undisturbed"),
)
parser.add_argument(
    "--policies",
    choices=("ppo", "zero", "random"),
    nargs="+",
    default=("ppo", "zero", "random"),
)
parser.add_argument("--visual", default="assets/squat_disturbed_eval.png")
args = parser.parse_args()

motion = np.load(args.motion)
model = PPO.load(args.model)


def evaluate(policy, env_kwargs, capture=False):
    env = RagdollEnv(
        render_mode="rgb_array" if capture else None,
        reference_motion=motion,
        **env_kwargs,
    )
    completed = 0
    rewards = []
    lengths = []
    visual_frames = []
    visual_labels = []
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=10_000 + episode)
            rng = np.random.default_rng(20_000 + episode)
            push_start = env._pushes[0][0] if env._pushes else -1
            capture_at = {
                push_start - 1: "before push",
                push_start + 2: "during push",
                push_start + 10: "after push",
                len(motion) - 1: "motion end",
            }
            done = False
            length = 0
            info = {}
            episode_frames = []
            episode_labels = []
            if capture and not visual_frames:
                episode_frames.append(env.render())
                episode_labels.append("motion start")
            while not done:
                if policy == "ppo":
                    action, _ = model.predict(observation, deterministic=True)
                elif policy == "zero":
                    action = np.zeros(28, dtype=np.float32)
                else:
                    action = rng.uniform(-1.0, 1.0, 28).astype(np.float32)
                observation, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                length += 1
                done = terminated or truncated
                if capture and not visual_frames and env.frame_idx in capture_at:
                    episode_frames.append(env.render())
                    episode_labels.append(capture_at[env.frame_idx])
            completed += int(info.get("completed", False))
            lengths.append(length)
            if capture and info.get("completed", False) and not visual_frames:
                visual_frames = episode_frames
                visual_labels = episode_labels
    finally:
        env.close()
    return {
        "completed": completed,
        "reward": float(np.mean(rewards)),
        "length": float(np.mean(lengths)),
        "frames": visual_frames,
        "labels": visual_labels,
    }


results = {}
for setting, kwargs in (
    ("disturbed", DISTURBED_ENV_KWARGS),
    ("undisturbed", UNDISTURBED_ENV_KWARGS),
):
    if setting not in args.settings:
        continue
    for policy in args.policies:
        results[(setting, policy)] = evaluate(
            policy, kwargs, capture=(setting == "disturbed" and policy == "ppo")
        )

for (setting, policy), result in results.items():
    print(
        f"{setting},{policy}: completed={result['completed']}/{args.episodes} "
        f"({100 * result['completed'] / args.episodes:.1f}%), "
        f"mean_reward={result['reward']:.6f}, "
        f"mean_length={result['length']:.2f}"
    )

visual = results.get(("disturbed", "ppo"))
if visual and visual["frames"]:
    visual_path = Path(args.visual)
    visual_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1, len(visual["frames"]), figsize=(3 * len(visual["frames"]), 4)
    )
    axes = np.atleast_1d(axes)
    for axis, frame, label in zip(axes, visual["frames"], visual["labels"]):
        axis.imshow(frame)
        axis.set_title(label)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(visual_path, dpi=120)
    plt.close(figure)
    print(f"visual: {visual_path}")
else:
    print("visual: no completed disturbed PPO episode")

disturbed_ppo = results.get(("disturbed", "ppo"))
disturbed_zero = results.get(("disturbed", "zero"))
if disturbed_ppo and disturbed_zero:
    advantage = (
        100
        * (disturbed_ppo["completed"] - disturbed_zero["completed"])
        / args.episodes
    )
    passed = advantage >= 15.0
    print(f"gate: advantage={advantage:.1f}pp, passed={passed}")
    if not passed:
        raise SystemExit(1)
