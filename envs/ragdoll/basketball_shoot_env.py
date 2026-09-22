import numpy as np
import pybullet as p

from envs.ragdoll.ragdoll_env import RagdollEnv


class BasketballShootEnv(RagdollEnv):
    BALL_RADIUS = 0.12
    BALL_MASS = 0.62
    HAND_LINK = 5
    RELEASE_FRAME = 65
    SHOOT_MOTION_IDX = 1
    HOOP_POSITION = np.array([-3.2, -0.5, 3.45], dtype=np.float64)
    HOOP_RADIUS = 0.45
    RIM_TUBE_RADIUS = 0.035
    RIM_SEGMENTS = 20
    SHOT_OUTCOME_WEIGHT = 10.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.reference_motions) <= self.SHOOT_MOTION_IDX:
            raise ValueError("basketball shoot requires the conditioned shoot motion")
        self.ball_id = None
        self.rim_ids = []
        self.ball_released = False
        self.made = False
        self.min_hoop_distance = float("inf")
        self.release_speed = 0.0
        self.release_angle_degrees = 0.0
        self._made_rewarded = False
        self.release_alignment = 0.0
        self._previous_ball_position = None

    def _hand_position(self):
        position = p.getLinkState(
            self.humanoid_id,
            self.HAND_LINK,
            computeLinkVelocity=1,
            physicsClientId=self.client_id,
        )[0]
        return np.asarray(position, dtype=np.float64) + np.array([-0.14, 0.0, 0.04])

    def _create_ball_and_hoop(self):
        ball_collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self.BALL_RADIUS,
            physicsClientId=self.client_id,
        )
        ball_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.BALL_RADIUS,
            rgbaColor=(0.95, 0.35, 0.05, 1.0),
            physicsClientId=self.client_id,
        )
        hand_position = self._hand_position()
        self.ball_id = p.createMultiBody(
            baseMass=self.BALL_MASS,
            baseCollisionShapeIndex=ball_collision,
            baseVisualShapeIndex=ball_visual,
            basePosition=hand_position,
            physicsClientId=self.client_id,
        )
        p.changeDynamics(
            self.ball_id,
            -1,
            restitution=0.75,
            lateralFriction=0.7,
            rollingFriction=0.01,
            physicsClientId=self.client_id,
        )
        for link_index in range(-1, p.getNumJoints(self.humanoid_id, physicsClientId=self.client_id)):
            p.setCollisionFilterPair(
                self.ball_id,
                self.humanoid_id,
                -1,
                link_index,
                0,
                physicsClientId=self.client_id,
            )

        rim_collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self.RIM_TUBE_RADIUS,
            physicsClientId=self.client_id,
        )
        rim_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.RIM_TUBE_RADIUS,
            rgbaColor=(0.85, 0.1, 0.05, 1.0),
            physicsClientId=self.client_id,
        )
        self.rim_ids = []
        for angle in np.linspace(0.0, 2.0 * np.pi, self.RIM_SEGMENTS, endpoint=False):
            position = self.HOOP_POSITION + self.HOOP_RADIUS * np.array(
                [np.cos(angle), np.sin(angle), 0.0]
            )
            rim_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=rim_collision,
                baseVisualShapeIndex=rim_visual,
                basePosition=position,
                physicsClientId=self.client_id,
            )
            self.rim_ids.append(rim_id)

    def _hold_ball(self):
        hand_position = self._hand_position()
        p.resetBasePositionAndOrientation(
            self.ball_id,
            hand_position,
            (0.0, 0.0, 0.0, 1.0),
            physicsClientId=self.client_id,
        )
        p.resetBaseVelocity(
            self.ball_id,
            linearVelocity=(0.0, 0.0, 0.0),
            angularVelocity=(0.0, 0.0, 0.0),
            physicsClientId=self.client_id,
        )

    def _release_ball(self):
        hand_state = p.getLinkState(
            self.humanoid_id,
            self.HAND_LINK,
            computeLinkVelocity=1,
            physicsClientId=self.client_id,
        )
        velocity = np.asarray(hand_state[6], dtype=np.float64)
        position = np.asarray(
            p.getBasePositionAndOrientation(
                self.ball_id, physicsClientId=self.client_id
            )[0],
            dtype=np.float64,
        )
        p.resetBaseVelocity(
            self.ball_id,
            linearVelocity=velocity,
            physicsClientId=self.client_id,
        )
        self.ball_released = True
        self.release_speed = float(np.linalg.norm(velocity))
        horizontal_speed = float(np.linalg.norm(velocity[:2]))
        self.release_angle_degrees = float(
            np.degrees(np.arctan2(velocity[2], horizontal_speed))
        )
        target_vector = self.HOOP_POSITION - position
        denominator = np.linalg.norm(velocity) * np.linalg.norm(target_vector)
        self.release_alignment = (
            float(np.dot(velocity, target_vector) / denominator)
            if denominator > 1e-8
            else 0.0
        )
        self._previous_ball_position = position

    def _shot_outcome(self, done):
        if not self.ball_released:
            return 0.0
        position = np.asarray(
            p.getBasePositionAndOrientation(
                self.ball_id, physicsClientId=self.client_id
            )[0],
            dtype=np.float64,
        )
        distance = float(np.linalg.norm(position - self.HOOP_POSITION))
        self.min_hoop_distance = min(self.min_hoop_distance, distance)
        inner_radius = self.HOOP_RADIUS - self.BALL_RADIUS
        crossed_downward = (
            self._previous_ball_position is not None
            and self._previous_ball_position[2] >= self.HOOP_POSITION[2]
            and position[2] < self.HOOP_POSITION[2]
        )
        if crossed_downward:
            previous = self._previous_ball_position
            crossing_fraction = (self.HOOP_POSITION[2] - previous[2]) / (
                position[2] - previous[2]
            )
            crossing_xy = previous[:2] + crossing_fraction * (
                position[:2] - previous[:2]
            )
            radial_distance = float(
                np.linalg.norm(crossing_xy - self.HOOP_POSITION[:2])
            )
        else:
            radial_distance = float("inf")
        if crossed_downward and radial_distance <= inner_radius:
            self.made = True
        self._previous_ball_position = position
        if self.made and not self._made_rewarded:
            self._made_rewarded = True
            return 1.0
        if done and not self.made:
            distance_credit = max(0.0, 1.0 - self.min_hoop_distance / 2.0)
            alignment_credit = max(0.0, self.release_alignment)
            return 0.5 * distance_credit + 0.25 * alignment_credit
        return 0.0

    def reset(self, seed=None, options=None, motion_idx=None):
        if motion_idx not in (None, self.SHOOT_MOTION_IDX):
            raise ValueError("basketball shoot environment only supports motion_idx=1")
        observation, info = super().reset(
            seed=seed, options=options, motion_idx=self.SHOOT_MOTION_IDX
        )
        self._create_ball_and_hoop()
        self.ball_released = False
        self.made = False
        self._made_rewarded = False
        self.min_hoop_distance = float("inf")
        self.release_speed = 0.0
        self.release_angle_degrees = 0.0
        self.release_alignment = 0.0
        self._previous_ball_position = self._hand_position()
        self._hold_ball()
        return observation, {**info, **self._basketball_info(0.0)}

    def _basketball_info(self, shot_outcome_reward):
        return {
            "ball_released": self.ball_released,
            "made": self.made,
            "release_speed": self.release_speed,
            "release_angle_degrees": self.release_angle_degrees,
            "release_alignment": self.release_alignment,
            "min_hoop_distance": self.min_hoop_distance,
            "weighted_shot_outcome_reward": (
                self.SHOT_OUTCOME_WEIGHT * shot_outcome_reward
            ),
            "shot_outcome_reward": shot_outcome_reward,
        }

    def step(self, action):
        if not self.ball_released:
            self._hold_ball()
        observation, imitation_reward, terminated, truncated, info = super().step(action)
        if not self.ball_released and self.frame_idx >= self.RELEASE_FRAME:
            self._hold_ball()
            self._release_ball()
        shot_outcome_reward = self._shot_outcome(terminated or truncated)
        info.update(self._basketball_info(shot_outcome_reward))
        return (
            observation,
            imitation_reward + self.SHOT_OUTCOME_WEIGHT * shot_outcome_reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-1.5, -0.2, 2.7),
            distance=7.5,
            yaw=0.0,
            pitch=-8.0,
            roll=0.0,
            upAxisIndex=2,
        )
        projection = p.computeProjectionMatrixFOV(
            fov=45.0, aspect=4.0 / 3.0, nearVal=0.1, farVal=20.0
        )
        _, _, rgba, _, _ = p.getCameraImage(
            640,
            480,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client_id,
        )
        return np.asarray(rgba, dtype=np.uint8)[..., :3]
