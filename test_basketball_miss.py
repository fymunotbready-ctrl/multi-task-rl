import numpy as np

from basketball_shoot import UNDISTURBED_BASKETBALL_ENV_KWARGS
from commands import COMMAND_MAP, resolve_command
from envs.ragdoll import BasketballMissEnv


assert COMMAND_MAP["miss the target"] == COMMAND_MAP["shoot the target"]
assert resolve_command(" MISS THE TARGET ") == 1

env = BasketballMissEnv(**UNDISTURBED_BASKETBALL_ENV_KWARGS)
try:
    observation, _ = env.reset(seed=10_000)
    assert observation.shape == (113,)

    env.ball_released = True
    env.made = True
    adjustment, outcome = env._miss_reward_adjustment(
        {"shot_make_reward": 1.0}, episode_done=False
    )
    assert adjustment == -2.0 * env.shot_outcome_weight
    assert outcome == 0.0

    env.made = False
    adjustment, outcome = env._miss_reward_adjustment(
        {"shot_make_reward": 0.0}, episode_done=True
    )
    assert adjustment == env.shot_outcome_weight
    assert outcome == 1.0

    env.ball_released = False
    adjustment, outcome = env._miss_reward_adjustment(
        {"shot_make_reward": 0.0}, episode_done=True
    )
    assert adjustment == -env.shot_outcome_weight
    assert outcome == -1.0

    env.reset(seed=10_001)
    done = False
    info = {}
    while not done:
        _, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert np.isfinite(reward)
        done = terminated or truncated
    assert info["ball_released"]
    assert info["intentional_miss"] == (not info["made"])
    assert info["miss_outcome_reward"] == (1.0 if not info["made"] else 0.0)
finally:
    env.close()

print("OK - miss command mirrors makes, rewards released misses, and penalizes non-release")
