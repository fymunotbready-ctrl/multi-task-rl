import argparse
from pathlib import Path
import sys

sys.path.insert(0, ".")

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from envs.ragdoll import RagdollEnv


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/squat_ppo.zip")
parser.add_argument("--motion", default="motions/squat.npy")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--visual", default="assets/squat_eval.png")
args = parser.parse_args()

motion = np.load(args.motion)
env = RagdollEnv(render_mode="rgb_array", reference_motion=motion)
model = PPO.load(args.model)
completed = 0
episode_lengths = []
frame_rewards = []
visual_frames = []
visual_labels = []
capture_at = {0, len(motion) // 4, len(motion) // 2, 3 * len(motion) // 4, len(motion) - 1}

try:
    for episode in range(args.episodes):
        observation, _ = env.reset(seed=10_000 + episode)
        if episode == 0:
            visual_frames.append(env.render())
            visual_labels.append("frame 0")

        done = False
        length = 0
        info = {}
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            length += 1
            frame_rewards.append(reward)
            done = terminated or truncated
            if episode == 0 and env.frame_idx in capture_at:
                visual_frames.append(env.render())
                visual_labels.append(f"frame {env.frame_idx}")

        completed += int(info.get("completed", False))
        episode_lengths.append(length)
finally:
    env.close()

visual_path = Path(args.visual)
visual_path.parent.mkdir(parents=True, exist_ok=True)
figure, axes = plt.subplots(1, len(visual_frames), figsize=(3 * len(visual_frames), 4))
axes = np.atleast_1d(axes)
for axis, frame, label in zip(axes, visual_frames, visual_labels):
    axis.imshow(frame)
    axis.set_title(label)
    axis.axis("off")
figure.tight_layout()
figure.savefig(visual_path, dpi=120)
plt.close(figure)

success_rate = completed / args.episodes
print(f"episodes: {args.episodes}")
print(f"completed: {completed}")
print(f"success_rate: {success_rate * 100:.1f}%")
print(f"mean_per_frame_imitation_reward: {np.mean(frame_rewards):.6f}")
print(f"mean_episode_length: {np.mean(episode_lengths):.2f}")
print(f"reference_length: {len(motion)}")
print(f"visual: {visual_path}")
