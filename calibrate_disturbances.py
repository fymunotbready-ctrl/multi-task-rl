import argparse

import numpy as np

from disturbed_squat import DISTURBED_ENV_KWARGS
from envs.ragdoll import RagdollEnv


parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--forces", type=float, nargs="+", default=[0, 400, 700, 1000, 1200])
parser.add_argument(
    "--policies", choices=("zero", "random"), nargs="+", default=("zero", "random")
)
args = parser.parse_args()
motion = np.load("motions/squat.npy")


def evaluate(force, policy):
    env_kwargs = {
        **DISTURBED_ENV_KWARGS,
        "push_force_range": (0.8 * force, force),
    }
    env = RagdollEnv(reference_motion=motion, **env_kwargs)
    completed = 0
    rewards = []
    lengths = []
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=10_000 + episode)
            del observation
            rng = np.random.default_rng(20_000 + episode)
            done = False
            length = 0
            info = {}
            while not done:
                if policy == "zero":
                    action = np.zeros(28, dtype=np.float32)
                else:
                    action = rng.uniform(-1.0, 1.0, 28).astype(np.float32)
                _, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                length += 1
                done = terminated or truncated
            completed += int(info.get("completed", False))
            lengths.append(length)
    finally:
        env.close()
    return completed, np.mean(rewards), np.mean(lengths)


print("force_max,policy,completed,episodes,completion_pct,mean_reward,mean_length")
for force in args.forces:
    for policy in args.policies:
        completed, reward, length = evaluate(force, policy)
        print(
            f"{force:g},{policy},{completed},{args.episodes},"
            f"{100 * completed / args.episodes:.1f},{reward:.6f},{length:.2f}",
            flush=True,
        )
