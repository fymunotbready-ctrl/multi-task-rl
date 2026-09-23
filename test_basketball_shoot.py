import numpy as np
import pybullet as p

from basketball_shoot import UNDISTURBED_BASKETBALL_ENV_KWARGS
from envs.ragdoll import BasketballShootEnv


env = BasketballShootEnv(
    **UNDISTURBED_BASKETBALL_ENV_KWARGS,
    release_imitation_weight=0.2,
)
try:
    observation, info = env.reset(seed=10_000)
    assert observation.shape == (113,)
    hoop_vector = env.HOOP_POSITION - env._hand_position()
    np.testing.assert_allclose(observation[-4:-1], hoop_vector, rtol=1e-6)
    assert np.isclose(observation[-1], np.linalg.norm(hoop_vector))
    assert info["motion_idx"] == env.SHOOT_MOTION_IDX
    assert len(env.rim_ids) == env.RIM_SEGMENTS
    assert (
        p.getDynamicsInfo(env.rim_ids[0], -1, physicsClientId=env.client_id)[0]
        == 0.0
    )
    assert p.getCollisionShapeData(
        env.rim_ids[0], -1, physicsClientId=env.client_id
    )
    assert not info["ball_released"]

    final_info = info
    for _ in range(env.RELEASE_FRAME):
        observation, reward, terminated, truncated, final_info = env.step(
            np.zeros(28, dtype=np.float32)
        )
        assert np.isfinite(reward)
        expected_imitation_weight = (
            0.2
            if final_info["frame_idx"]
            > env.RELEASE_FRAME - env.release_window_frames
            else 1.0
        )
        assert final_info["imitation_weight"] == expected_imitation_weight
        assert reward == (
            final_info["weighted_imitation_reward"]
            + env.shot_outcome_weight * final_info["shot_outcome_reward"]
        )
        assert not truncated
        if terminated:
            break
    assert final_info["ball_released"]
    assert final_info["release_speed"] > 0.0
    assert np.isfinite(final_info["release_angle_degrees"])
    assert final_info["shot_release_quality_reward"] > 0.0
    assert np.isfinite(final_info["predicted_closest_distance"])

    env._release_reward_pending = False
    env._release_hoop_distance = 4.0
    env._previous_best_hoop_distance = 2.0
    env._previous_ball_position = env.HOOP_POSITION + np.array([0.0, 0.0, 1.5])
    p.resetBasePositionAndOrientation(
        env.ball_id,
        env.HOOP_POSITION + np.array([0.0, 0.0, 1.5]),
        (0.0, 0.0, 0.0, 1.0),
        physicsClientId=env.client_id,
    )
    progress_components = env._shot_outcome()
    assert progress_components["progress"] > 0.0

    env.ball_released = True
    env.made = False
    env._made_rewarded = False
    env._previous_ball_position = env.HOOP_POSITION + np.array([0.0, 0.0, 0.1])
    p.resetBasePositionAndOrientation(
        env.ball_id,
        env.HOOP_POSITION - np.array([0.0, 0.0, 0.1]),
        (0.0, 0.0, 0.0, 1.0),
        physicsClientId=env.client_id,
    )
    make_components = env._shot_outcome()
    assert make_components["make"] == 1.0
    assert env.made
    assert env._shot_outcome()["make"] == 0.0

    env.set_hoop_radius_scale(2.0)
    _, curriculum_info = env.reset(seed=10_001)
    assert curriculum_info["hoop_radius"] == 2.0 * env.HOOP_RADIUS
    rim_position = np.asarray(
        p.getBasePositionAndOrientation(
            env.rim_ids[0], physicsClientId=env.client_id
        )[0]
    )
    assert np.isclose(
        np.linalg.norm(rim_position[:2] - env.HOOP_POSITION[:2]),
        curriculum_info["hoop_radius"],
    )
finally:
    env.close()

randomized_env = BasketballShootEnv(
    **UNDISTURBED_BASKETBALL_ENV_KWARGS,
    start_position_randomization=0.2,
    start_yaw_randomization_degrees=5.0,
)
try:
    randomized_observation, _ = randomized_env.reset(seed=10_002)
    base_position = p.getBasePositionAndOrientation(
        randomized_env.humanoid_id, physicsClientId=randomized_env.client_id
    )[0]
    assert abs(base_position[0]) <= 0.21
    assert abs(base_position[1]) <= 0.21
    randomized_vector = randomized_env.HOOP_POSITION - randomized_env._hand_position()
    np.testing.assert_allclose(
        randomized_observation[-4:-1], randomized_vector, rtol=1e-6
    )
finally:
    randomized_env.close()

print(
    "OK - hoop observation, imitation annealing, start randomization, "
    "dense reward, rim, release, and make checks passed"
)
