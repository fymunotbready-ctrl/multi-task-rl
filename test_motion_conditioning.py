import numpy as np

from envs.ragdoll import RagdollEnv
from multi_motion import MOTION_PATHS


motions = tuple(np.load(path) for path in MOTION_PATHS)
env = RagdollEnv(reference_motions=MOTION_PATHS, include_reference_observation=True)
try:
    for motion_idx, motion in enumerate(motions):
        observation, info = env.reset(seed=123, motion_idx=motion_idx)
        assert observation.shape == (109,)
        assert info["motion_idx"] == motion_idx
        assert env.motion_idx == motion_idx
        assert np.array_equal(env.reference_motion, motion)
        assert np.array_equal(observation[-3:], np.eye(3, dtype=np.float32)[motion_idx])
        done = False
        while not done:
            _, _, terminated, truncated, step_info = env.step(
                np.zeros(28, dtype=np.float32)
            )
            done = terminated or truncated
        assert step_info["motion_idx"] == motion_idx
        assert step_info["completed"], f"motion {motion_idx} fell at frame {step_info['frame_idx']}"

    first_idx = env.reset(seed=456)[1]["motion_idx"]
    second_idx = env.reset(seed=456)[1]["motion_idx"]
    assert first_idx == second_idx

    try:
        env.reset(motion_idx=3)
    except ValueError as error:
        assert "motion_idx" in str(error)
    else:
        raise AssertionError("invalid motion index was accepted")
finally:
    env.close()

reference_free = RagdollEnv()
try:
    observation, info = reference_free.reset(seed=123)
    assert observation.shape == (71,)
    assert info == {}
finally:
    reference_free.close()

print("OK - motion selection is conditioned and reference-free API is unchanged")
