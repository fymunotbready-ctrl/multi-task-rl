import numpy as np

try:
    from reward_cpp import pose_similarity_reward as pose_similarity_reward_cpp
except ImportError:
    pose_similarity_reward_cpp = None


def pose_similarity_reward_python(current, reference):
    current = np.asarray(current)
    reference = np.asarray(reference)
    if current.ndim != 1 or reference.ndim != 1:
        raise ValueError("current and reference must be 1-D arrays")
    if len(current) != len(reference):
        raise ValueError("current and reference must have the same length")
    return float(np.exp(-2.0 * np.sum((current - reference) ** 2)))


def select_pose_similarity_reward(backend="auto"):
    if backend not in ("auto", "python", "cpp"):
        raise ValueError(f"unsupported reward backend: {backend}")
    if backend == "cpp" and pose_similarity_reward_cpp is None:
        raise RuntimeError("reward_cpp is not built; run ./build_reward_cpp.sh")
    if backend == "python" or pose_similarity_reward_cpp is None:
        return pose_similarity_reward_python, "python"
    return pose_similarity_reward_cpp, "cpp"
