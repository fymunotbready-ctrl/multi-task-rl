import argparse
import time

import numpy as np

from envs.ragdoll import RagdollEnv
from reward import pose_similarity_reward_cpp, pose_similarity_reward_python


parser = argparse.ArgumentParser()
parser.add_argument("--calls", type=int, default=100_000)
parser.add_argument("--steps", type=int, default=10_000)
args = parser.parse_args()

if pose_similarity_reward_cpp is None:
    raise RuntimeError("reward_cpp is not built; run ./build_reward_cpp.sh")

rng = np.random.default_rng(2468)
current = rng.normal(size=28)
reference = rng.normal(size=28)


def time_reward(reward):
    start = time.perf_counter()
    total = 0.0
    for _ in range(args.calls):
        total += reward(current, reference)
    elapsed = time.perf_counter() - start
    return elapsed, total


def time_environment(backend):
    motion = np.load("motions/squat.npy")
    env = RagdollEnv(reference_motion=motion, reward_backend=backend)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    episode = 0
    observation, _ = env.reset(seed=10_000)
    del observation
    start = time.perf_counter()
    try:
        for _ in range(args.steps):
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                episode += 1
                env.reset(seed=10_000 + episode)
    finally:
        elapsed = time.perf_counter() - start
        env.close()
    return elapsed


python_elapsed, python_total = time_reward(pose_similarity_reward_python)
cpp_elapsed, cpp_total = time_reward(pose_similarity_reward_cpp)
assert np.isclose(python_total, cpp_total)
python_env_elapsed = time_environment("python")
cpp_env_elapsed = time_environment("cpp")

python_ns = python_elapsed * 1e9 / args.calls
cpp_ns = cpp_elapsed * 1e9 / args.calls
python_steps_per_second = args.steps / python_env_elapsed
cpp_steps_per_second = args.steps / cpp_env_elapsed

print(f"calls: {args.calls}")
print(f"python_reward_ns_per_call: {python_ns:.1f}")
print(f"cpp_reward_ns_per_call: {cpp_ns:.1f}")
print(f"reward_speedup_cpp_over_python: {python_ns / cpp_ns:.3f}x")
print(f"environment_steps: {args.steps}")
print(f"python_environment_steps_per_second: {python_steps_per_second:.2f}")
print(f"cpp_environment_steps_per_second: {cpp_steps_per_second:.2f}")
print(
    "environment_speedup_cpp_over_python: "
    f"{cpp_steps_per_second / python_steps_per_second:.3f}x"
)
