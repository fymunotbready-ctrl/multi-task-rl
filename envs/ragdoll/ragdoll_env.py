from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from reward import select_pose_similarity_reward


class RagdollEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    _SIMULATION_HZ = 240
    _CONTROL_SUBSTEPS = 4
    _BASE_HEIGHT = 3.55
    _ACTION_OFFSET_SCALE = 0.25
    _PD_FORCE = 200.0

    def __init__(
        self,
        render_mode=None,
        max_steps=1000,
        reference_motion=None,
        reference_motions=None,
        reward_backend="auto",
        push_count=0,
        push_frame_range=(20, 90),
        push_force_range=(0.0, 0.0),
        push_duration_range=(1, 1),
        action_noise_std=0.0,
        control_strength=1.0,
        include_reference_observation=False,
    ):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"unsupported render_mode: {render_mode}")
        if push_count < 0:
            raise ValueError("push_count must be non-negative")
        if action_noise_std < 0.0:
            raise ValueError("action_noise_std must be non-negative")
        if control_strength <= 0.0:
            raise ValueError("control_strength must be positive")
        if push_frame_range[0] < 0 or push_frame_range[1] < push_frame_range[0]:
            raise ValueError("push_frame_range must be ordered and non-negative")
        if push_force_range[0] < 0.0 or push_force_range[1] < push_force_range[0]:
            raise ValueError("push_force_range must be ordered and non-negative")
        if push_duration_range[0] < 1 or push_duration_range[1] < push_duration_range[0]:
            raise ValueError("push_duration_range must be ordered and positive")

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.reference_motions = self._load_references(
            reference_motion, reference_motions
        )
        self.motion_idx = 0
        self.reference_motion = (
            self.reference_motions[0] if self.reference_motions else None
        )
        self.frame_idx = 0
        self.push_count = push_count
        self.push_frame_range = push_frame_range
        self.push_force_range = push_force_range
        self.push_duration_range = push_duration_range
        self.action_noise_std = action_noise_std
        self.control_strength = control_strength
        self.include_reference_observation = include_reference_observation
        self._pushes = []
        self._active_push = np.zeros(3, dtype=np.float32)
        self._pose_similarity_reward, self.reward_backend = (
            select_pose_similarity_reward(reward_backend)
        )
        mode = p.GUI if render_mode == "human" else p.DIRECT
        self.client_id = p.connect(mode)
        if self.client_id < 0:
            raise RuntimeError("failed to connect to PyBullet")

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client_id
        )
        self._load_world()
        self._joint_specs = self._controllable_joints()
        action_dim = sum(dof for _, dof in self._joint_specs)
        for motion in self.reference_motions:
            if motion.ndim != 2:
                raise ValueError("reference motions must be 2D frame arrays")
            if motion.shape[1] != action_dim:
                raise ValueError(
                    f"reference frames need {action_dim} joint angles, "
                    f"got {motion.shape[1]}"
                )

        observation_dim = self._get_obs().shape[0]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32,
        )

    @classmethod
    def _load_references(cls, reference_motion, reference_motions):
        if reference_motion is not None and reference_motions is not None:
            raise ValueError("provide reference_motion or reference_motions, not both")
        if reference_motions is None:
            reference_motions = () if reference_motion is None else (reference_motion,)
        return tuple(cls._load_reference(motion) for motion in reference_motions)

    @staticmethod
    def _load_reference(reference_motion):
        if isinstance(reference_motion, (str, Path)):
            reference_motion = np.load(reference_motion)
        return np.asarray(reference_motion, dtype=np.float32)

    def _load_world(self):
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / self._SIMULATION_HZ, physicsClientId=self.client_id)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        base_orientation = p.getQuaternionFromEuler((np.pi / 2.0, 0.0, 0.0))
        self.humanoid_id = p.loadURDF(
            "humanoid/humanoid.urdf",
            basePosition=(0.0, 0.0, self._BASE_HEIGHT),
            baseOrientation=base_orientation,
            useFixedBase=False,
            physicsClientId=self.client_id,
        )
        p.changeDynamics(
            self.plane_id, -1, lateralFriction=1.0, physicsClientId=self.client_id
        )
        for link_index in range(
            -1, p.getNumJoints(self.humanoid_id, physicsClientId=self.client_id)
        ):
            p.changeDynamics(
                self.humanoid_id,
                link_index,
                lateralFriction=1.0,
                physicsClientId=self.client_id,
            )
        self.steps = 0
        self.frame_idx = 0

    def _controllable_joints(self):
        dofs = {
            p.JOINT_REVOLUTE: 1,
            p.JOINT_PRISMATIC: 1,
            p.JOINT_SPHERICAL: 3,
            p.JOINT_PLANAR: 3,
        }
        joints = []
        for joint_index in range(
            p.getNumJoints(self.humanoid_id, physicsClientId=self.client_id)
        ):
            joint_type = p.getJointInfo(
                self.humanoid_id, joint_index, physicsClientId=self.client_id
            )[2]
            if joint_type in dofs:
                joints.append((joint_index, dofs[joint_type]))
        return joints

    @staticmethod
    def _rotation_vector_to_quaternion(rotation):
        angle = float(np.linalg.norm(rotation))
        if angle < 1e-8:
            return (0.0, 0.0, 0.0, 1.0)
        axis = rotation / angle
        return p.getQuaternionFromAxisAngle(axis.tolist(), angle)

    def _set_joint_pose(self, angles):
        offset = 0
        for joint_index, dof in self._joint_specs:
            target = angles[offset : offset + dof]
            if dof == 1:
                p.resetJointState(
                    self.humanoid_id,
                    joint_index,
                    targetValue=float(target[0]),
                    targetVelocity=0.0,
                    physicsClientId=self.client_id,
                )
            else:
                p.resetJointStateMultiDof(
                    self.humanoid_id,
                    joint_index,
                    targetValue=self._rotation_vector_to_quaternion(target),
                    targetVelocity=[0.0] * dof,
                    physicsClientId=self.client_id,
                )
            offset += dof

    def _disable_motors(self):
        for joint_index, dof in self._joint_specs:
            if dof == 1:
                p.setJointMotorControl2(
                    self.humanoid_id,
                    joint_index,
                    p.TORQUE_CONTROL,
                    force=0.0,
                    physicsClientId=self.client_id,
                )
            else:
                p.setJointMotorControlMultiDof(
                    self.humanoid_id,
                    joint_index,
                    p.TORQUE_CONTROL,
                    force=[0.0] * dof,
                    physicsClientId=self.client_id,
                )

    def _apply_torques(self, action):
        offset = 0
        for joint_index, dof in self._joint_specs:
            torque = action[offset : offset + dof]
            if dof == 1:
                p.setJointMotorControl2(
                    self.humanoid_id,
                    joint_index,
                    p.TORQUE_CONTROL,
                    force=float(torque[0]),
                    physicsClientId=self.client_id,
                )
            else:
                p.setJointMotorControlMultiDof(
                    self.humanoid_id,
                    joint_index,
                    p.TORQUE_CONTROL,
                    force=torque.tolist(),
                    physicsClientId=self.client_id,
                )
            offset += dof

    def _apply_pd_targets(self, target_angles):
        offset = 0
        for joint_index, dof in self._joint_specs:
            target = target_angles[offset : offset + dof]
            if dof == 1:
                p.setJointMotorControl2(
                    self.humanoid_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=float(target[0]),
                    force=self._PD_FORCE * self.control_strength,
                    positionGain=0.4,
                    velocityGain=1.0,
                    physicsClientId=self.client_id,
                )
            else:
                p.setJointMotorControlMultiDof(
                    self.humanoid_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=self._rotation_vector_to_quaternion(target),
                    targetVelocity=[0.0] * dof,
                    force=[self._PD_FORCE * self.control_strength] * dof,
                    positionGain=0.4,
                    velocityGain=1.0,
                    physicsClientId=self.client_id,
                )
            offset += dof

    def get_joint_angles(self):
        angles = []
        for joint_index, dof in self._joint_specs:
            position = p.getJointStateMultiDof(
                self.humanoid_id, joint_index, physicsClientId=self.client_id
            )[0]
            if dof == 1:
                angles.append(position[0])
            else:
                axis, angle = p.getAxisAngleFromQuaternion(position)
                angles.extend(np.asarray(axis) * angle)
        return np.asarray(angles, dtype=np.float32)

    def _sample_pushes(self):
        pushes = []
        frame_low, frame_high = self.push_frame_range
        force_low, force_high = self.push_force_range
        duration_low, duration_high = self.push_duration_range
        for _ in range(self.push_count):
            start = int(self.np_random.integers(frame_low, frame_high + 1))
            duration = int(self.np_random.integers(duration_low, duration_high + 1))
            magnitude = float(self.np_random.uniform(force_low, force_high))
            angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            force = np.array(
                [magnitude * np.cos(angle), magnitude * np.sin(angle), 0.0],
                dtype=np.float32,
            )
            pushes.append((start, start + duration, force))
        return pushes

    def _apply_active_push(self):
        self._active_push.fill(0.0)
        for start, end, force in self._pushes:
            if start <= self.frame_idx < end:
                self._active_push += force
        if np.any(self._active_push):
            base_position = p.getBasePositionAndOrientation(
                self.humanoid_id, physicsClientId=self.client_id
            )[0]
            p.applyExternalForce(
                self.humanoid_id,
                -1,
                forceObj=self._active_push.tolist(),
                posObj=base_position,
                flags=p.WORLD_FRAME,
                physicsClientId=self.client_id,
            )

    def _get_obs(self):
        joint_positions = []
        joint_velocities = []
        for joint_index, _ in self._joint_specs:
            position, velocity, _, _ = p.getJointStateMultiDof(
                self.humanoid_id, joint_index, physicsClientId=self.client_id
            )
            joint_positions.extend(position)
            joint_velocities.extend(velocity)

        base_position, base_orientation = p.getBasePositionAndOrientation(
            self.humanoid_id, physicsClientId=self.client_id
        )
        observation = (
            joint_positions
            + joint_velocities
            + list(base_position)
            + list(base_orientation)
        )
        if self.reference_motion is not None and self.include_reference_observation:
            base_linear_velocity, base_angular_velocity = p.getBaseVelocity(
                self.humanoid_id, physicsClientId=self.client_id
            )
            reference_index = min(self.frame_idx, len(self.reference_motion) - 1)
            denominator = max(len(self.reference_motion) - 1, 1)
            observation.extend(base_linear_velocity)
            observation.extend(base_angular_velocity)
            observation.extend(self.reference_motion[reference_index])
            observation.append(reference_index / denominator)
        if len(self.reference_motions) > 1:
            observation.extend(
                np.eye(len(self.reference_motions), dtype=np.float32)[self.motion_idx]
            )
        return np.asarray(observation, dtype=np.float32)

    def _fallen(self):
        base_position, base_orientation = p.getBasePositionAndOrientation(
            self.humanoid_id, physicsClientId=self.client_id
        )
        up_z = p.getMatrixFromQuaternion(base_orientation)[7]
        return base_position[2] < 1.8 or up_z < 0.5

    def reset(self, seed=None, options=None, motion_idx=None):
        super().reset(seed=seed)
        self._load_world()
        self._joint_specs = self._controllable_joints()
        if self.reference_motions:
            if motion_idx is None and options is not None:
                motion_idx = options.get("motion_idx")
            if motion_idx is None:
                motion_idx = int(self.np_random.integers(len(self.reference_motions)))
            if not 0 <= motion_idx < len(self.reference_motions):
                raise ValueError(
                    f"motion_idx must be in [0, {len(self.reference_motions) - 1}]"
                )
            self.motion_idx = motion_idx
            self.reference_motion = self.reference_motions[motion_idx]
        if self.reference_motion is None:
            self._disable_motors()
        else:
            position_noise = self.np_random.uniform(-0.01, 0.01, size=2)
            tilt_noise = self.np_random.uniform(-0.01, 0.01, size=2)
            p.resetBasePositionAndOrientation(
                self.humanoid_id,
                (
                    float(position_noise[0]),
                    float(position_noise[1]),
                    self._BASE_HEIGHT,
                ),
                p.getQuaternionFromEuler(
                    (np.pi / 2.0 + tilt_noise[0], tilt_noise[1], 0.0)
                ),
                physicsClientId=self.client_id,
            )
            self._set_joint_pose(self.reference_motion[0])
            self._apply_pd_targets(self.reference_motion[0])
            self._pushes = self._sample_pushes()
            self._active_push.fill(0.0)
        info = {"motion_idx": self.motion_idx} if self.reference_motion is not None else {}
        return self._get_obs(), info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"expected action shape {self.action_space.shape}, got {action.shape}"
            )

        if self.reference_motion is None:
            self._apply_torques(action)
        else:
            reference = self.reference_motion[self.frame_idx]
            noisy_action = action
            if self.action_noise_std:
                noisy_action = np.clip(
                    action
                    + self.np_random.normal(
                        0.0, self.action_noise_std, size=action.shape
                    ),
                    -1.0,
                    1.0,
                )
            target = reference + self._ACTION_OFFSET_SCALE * noisy_action
            self._apply_pd_targets(target)

        for _ in range(self._CONTROL_SUBSTEPS):
            self._apply_active_push()
            p.stepSimulation(physicsClientId=self.client_id)
        self.steps += 1

        if self.reference_motion is None:
            truncated = self.steps >= self.max_steps
            return self._get_obs(), 0.0, False, truncated, {}

        current_angles = self.get_joint_angles()
        reward = self._pose_similarity_reward(current_angles, reference)
        self.frame_idx += 1
        fallen = self._fallen()
        completed = self.frame_idx >= len(self.reference_motion) and not fallen
        terminated = fallen or completed
        info = {
            "completed": completed,
            "fallen": fallen,
            "frame_idx": self.frame_idx,
            "imitation_reward": reward,
            "push_force": self._active_push.copy(),
            "motion_idx": self.motion_idx,
        }
        return self._get_obs(), reward, terminated, False, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(0.0, 0.0, 2.0),
            distance=7.0,
            yaw=0.0,
            pitch=-5.0,
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

    def close(self):
        if self.client_id >= 0 and p.isConnected(self.client_id):
            p.disconnect(self.client_id)
            self.client_id = -1
