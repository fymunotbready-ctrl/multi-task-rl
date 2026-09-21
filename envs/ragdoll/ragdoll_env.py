import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class RagdollEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 240}

    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError(f"unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.max_steps = max_steps
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

    def _load_world(self):
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / self.metadata["render_fps"], physicsClientId=self.client_id)
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.humanoid_id = p.loadURDF(
            "humanoid/humanoid.urdf",
            basePosition=(0.0, 0.0, 1.4),
            useFixedBase=False,
            physicsClientId=self.client_id,
        )
        self.steps = 0

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
        return np.asarray(
            joint_positions
            + joint_velocities
            + list(base_position)
            + list(base_orientation),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_world()
        self._joint_specs = self._controllable_joints()
        self._disable_motors()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"expected action shape {self.action_space.shape}, got {action.shape}"
            )

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

        p.stepSimulation(physicsClientId=self.client_id)
        self.steps += 1
        truncated = self.steps >= self.max_steps
        return self._get_obs(), 0.0, False, truncated, {}

    def close(self):
        if self.client_id >= 0 and p.isConnected(self.client_id):
            p.disconnect(self.client_id)
            self.client_id = -1
