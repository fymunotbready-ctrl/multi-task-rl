import numpy as np
import pybullet as p

from envs.ragdoll.basketball_shoot_env import BasketballShootEnv


class BasketballDunkEnv(BasketballShootEnv):
    DUNK_MOTION_IDX = 2
    SHOOT_MOTION_IDX = DUNK_MOTION_IDX
    RELEASE_FRAME = 72
    HOOP_POSITION = np.array([-2.55, -0.75, 4.1], dtype=np.float64)
    JUMP_START_FRAME = 32
    JUMP_END_FRAME = 44
    DEFAULT_JUMP_FORCE = 1500.0

    def __init__(self, *args, jump_force=DEFAULT_JUMP_FORCE, **kwargs):
        if jump_force < 0.0:
            raise ValueError("jump_force must be non-negative")
        self.jump_force = float(jump_force)
        self.ever_airborne = False
        self.landed_after_jump = False
        self.peak_base_height = 0.0
        super().__init__(*args, **kwargs)

    def _apply_active_push(self):
        super()._apply_active_push()
        if self.JUMP_START_FRAME <= self.frame_idx < self.JUMP_END_FRAME:
            base_position = p.getBasePositionAndOrientation(
                self.humanoid_id, physicsClientId=self.client_id
            )[0]
            p.applyExternalForce(
                self.humanoid_id,
                -1,
                (0.0, 0.0, self.jump_force),
                base_position,
                p.WORLD_FRAME,
                physicsClientId=self.client_id,
            )

    def reset(self, seed=None, options=None, motion_idx=None):
        observation, info = super().reset(
            seed=seed, options=options, motion_idx=self.DUNK_MOTION_IDX
        )
        self.ever_airborne = False
        self.landed_after_jump = False
        self.peak_base_height = p.getBasePositionAndOrientation(
            self.humanoid_id, physicsClientId=self.client_id
        )[0][2]
        info.update(self._jump_info(False))
        return observation, info

    def _jump_info(self, airborne):
        return {
            "airborne": bool(airborne),
            "ever_airborne": bool(self.ever_airborne),
            "landed_after_jump": bool(self.landed_after_jump),
            "peak_base_height": float(self.peak_base_height),
            "jump_force": self.jump_force,
        }

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        base_height = p.getBasePositionAndOrientation(
            self.humanoid_id, physicsClientId=self.client_id
        )[0][2]
        self.peak_base_height = max(self.peak_base_height, base_height)
        ground_contacts = p.getContactPoints(
            bodyA=self.humanoid_id,
            bodyB=self.plane_id,
            physicsClientId=self.client_id,
        )
        airborne = self.frame_idx >= self.JUMP_START_FRAME and not ground_contacts
        if airborne:
            self.ever_airborne = True
        elif self.ever_airborne:
            self.landed_after_jump = True
        info.update(self._jump_info(airborne))
        return observation, reward, terminated, truncated, info
