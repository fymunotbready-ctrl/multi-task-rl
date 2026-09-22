from envs.ragdoll.basketball_shoot_env import BasketballShootEnv


class BasketballMissEnv(BasketballShootEnv):
    def _miss_reward_adjustment(self, info, episode_done):
        adjustment = -2.0 * self.shot_outcome_weight * info["shot_make_reward"]
        miss_outcome_reward = 0.0
        if episode_done:
            if not self.ball_released:
                miss_outcome_reward = -1.0
            elif not self.made:
                miss_outcome_reward = 1.0
            adjustment += self.shot_outcome_weight * miss_outcome_reward
        return adjustment, miss_outcome_reward

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        episode_done = terminated or truncated
        adjustment, miss_outcome_reward = self._miss_reward_adjustment(
            info, episode_done
        )
        reward += adjustment

        info["intentional_miss"] = bool(
            episode_done and self.ball_released and not self.made
        )
        info["miss_outcome_reward"] = miss_outcome_reward
        info["weighted_miss_outcome_reward"] = (
            self.shot_outcome_weight * miss_outcome_reward
        )
        return observation, reward, terminated, truncated, info
