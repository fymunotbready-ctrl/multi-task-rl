import numpy as np

from envs.ragdoll import RagdollEnv


motion = np.load("motions/squat.npy")
assert motion.shape == (120, 28)
assert np.allclose(motion[0], 0.0)
assert np.allclose(motion[-1], 0.0, atol=1e-6)

env = RagdollEnv(reference_motion=motion)
try:
    env.reset(seed=123)
    assert env.frame_idx == 0
    assert np.allclose(env.get_joint_angles(), motion[0], atol=1e-5)

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, reward, terminated, truncated, info = env.step(action)
    expected = np.exp(-2.0 * np.sum((env.get_joint_angles() - motion[0]) ** 2))
    assert np.isclose(reward, expected)
    assert env.frame_idx == 1
    assert not terminated and not truncated

    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(action)

    assert info["completed"] and not info["fallen"]
    assert env.frame_idx == len(motion)
finally:
    env.close()

print("OK - squat reference completes with PD targets")
