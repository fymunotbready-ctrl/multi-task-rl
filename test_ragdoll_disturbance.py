import numpy as np

from disturbed_squat import DISTURBED_ENV_KWARGS
from envs.ragdoll import RagdollEnv


motion = np.load("motions/squat.npy")
base = RagdollEnv(reference_motion=motion)
first = RagdollEnv(reference_motion=motion, **DISTURBED_ENV_KWARGS)
second = RagdollEnv(reference_motion=motion, **DISTURBED_ENV_KWARGS)
try:
    base_observation, _ = base.reset(seed=123)
    first_observation, _ = first.reset(seed=123)
    second_observation, _ = second.reset(seed=123)

    assert base_observation.shape == (71,)
    assert first_observation.shape == (106,)
    assert np.array_equal(first_observation, second_observation)
    assert len(first._pushes) == 1
    assert first._pushes[0][0] == second._pushes[0][0]
    assert first._pushes[0][1] == second._pushes[0][1]
    assert np.array_equal(first._pushes[0][2], second._pushes[0][2])

    action = np.full(28, 0.1, dtype=np.float32)
    for _ in range(30):
        first_observation, first_reward, first_done, _, first_info = first.step(action)
        second_observation, second_reward, second_done, _, second_info = second.step(action)
        assert np.array_equal(first_observation, second_observation)
        assert first_reward == second_reward
        assert first_done == second_done
        assert np.array_equal(first_info["push_force"], second_info["push_force"])
        if first_done:
            break
finally:
    base.close()
    first.close()
    second.close()

reference_free = RagdollEnv(push_count=1, push_force_range=(100.0, 100.0))
try:
    observation, _ = reference_free.reset(seed=123)
    assert observation.shape == (71,)
    _, reward, _, _, info = reference_free.step(np.zeros(28, dtype=np.float32))
    assert reward == 0.0
    assert info == {}
finally:
    reference_free.close()

print("OK - disturbances are opt-in, seeded, and reference-conditioned")
