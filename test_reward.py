import numpy as np

import reward as reward_module
from reward import pose_similarity_reward_cpp, pose_similarity_reward_python


if pose_similarity_reward_cpp is None:
    raise RuntimeError("reward_cpp is not built; run ./build_reward_cpp.sh")

compiled_reward = pose_similarity_reward_cpp
reward_module.pose_similarity_reward_cpp = None
fallback_reward, fallback_name = reward_module.select_pose_similarity_reward()
assert fallback_name == "python"
assert fallback_reward(np.zeros(28), np.zeros(28)) == 1.0
reward_module.pose_similarity_reward_cpp = compiled_reward

rng = np.random.default_rng(12345)
cases = [
    (np.zeros(28), np.zeros(28)),
    (np.ones(28), np.ones(28)),
    (np.full(28, 100.0), np.full(28, -100.0)),
]
for _ in range(10_000):
    cases.append((rng.normal(size=28), rng.normal(size=28)))

max_absolute_difference = max(
    abs(
        pose_similarity_reward_cpp(current, reference)
        - pose_similarity_reward_python(current, reference)
    )
    for current, reference in cases
)

for reward in (pose_similarity_reward_cpp, pose_similarity_reward_python):
    try:
        reward(np.zeros(27), np.zeros(28))
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("length mismatch must raise ValueError")

    try:
        reward(np.zeros((2, 2)), np.zeros((2, 2)))
    except ValueError as error:
        assert "1-D" in str(error)
    else:
        raise AssertionError("non-1-D input must raise ValueError")

print(f"cases: {len(cases)}")
print(f"max_absolute_difference: {max_absolute_difference:.3e}")
print("length_mismatch: passed")
print("non_1d: passed")
print("python_fallback: passed")
