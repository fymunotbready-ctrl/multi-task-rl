import numpy as np
import pybullet as p

from basketball_shoot import UNDISTURBED_BASKETBALL_ENV_KWARGS
from envs.ragdoll import BasketballShootEnv


env = BasketballShootEnv(**UNDISTURBED_BASKETBALL_ENV_KWARGS)
try:
    observation, info = env.reset(seed=10_000)
    assert observation.shape == (109,)
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
        assert reward == (
            final_info["imitation_reward"]
            + env.SHOT_OUTCOME_WEIGHT * final_info["shot_outcome_reward"]
        )
        assert not truncated
        if terminated:
            break
    assert final_info["ball_released"]
    assert final_info["release_speed"] > 0.0
    assert np.isfinite(final_info["release_angle_degrees"])

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
    assert env._shot_outcome(done=False) == 1.0
    assert env.made
    assert env._shot_outcome(done=False) == 0.0
finally:
    env.close()

print("OK - ball release, physical rim, and one-shot make detection passed")
