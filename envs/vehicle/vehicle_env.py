import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class VehicleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    MAX_STEPS = 300
    CONTROL_SUBSTEPS = 4
    WHEEL_JOINTS = (2, 3, 5, 7)
    STEERING_JOINTS = (4, 6)

    def __init__(
        self,
        render_mode=None,
        rear_friction=0.3,
        front_friction=1.2,
        motor_force=20.0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.rear_friction = rear_friction
        self.front_friction = front_friction
        self.motor_force = motor_force
        self.client = p.connect(p.DIRECT)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(18,), dtype=np.float32
        )
        self.car_id = None
        self.steps = 0

    def _load_world(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        yaw = float(self.np_random.uniform(-math.pi, math.pi))
        orientation = p.getQuaternionFromEuler((0, 0, yaw))
        self.car_id = p.loadURDF(
            "racecar/racecar.urdf",
            (0, 0, 0.2),
            orientation,
            physicsClientId=self.client,
        )
        for joint in range(p.getNumJoints(self.car_id, physicsClientId=self.client)):
            p.setJointMotorControl2(
                self.car_id,
                joint,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0,
                physicsClientId=self.client,
            )
        for joint in self.WHEEL_JOINTS[:2]:
            p.changeDynamics(
                self.car_id,
                joint,
                lateralFriction=self.rear_friction,
                physicsClientId=self.client,
            )
        for joint in self.WHEEL_JOINTS[2:]:
            p.changeDynamics(
                self.car_id,
                joint,
                lateralFriction=self.front_friction,
                physicsClientId=self.client,
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self._load_world()
        for _ in range(12):
            p.stepSimulation(physicsClientId=self.client)
        return self._observation(), {}

    def _kinematics(self):
        position, orientation = p.getBasePositionAndOrientation(
            self.car_id, physicsClientId=self.client
        )
        linear, angular = p.getBaseVelocity(self.car_id, physicsClientId=self.client)
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        local_velocity = rotation.T @ np.asarray(linear)
        forward_speed = float(local_velocity[0])
        lateral_speed = float(local_velocity[1])
        speed = float(np.linalg.norm(np.asarray(linear)[:2]))
        slip_angle = math.atan2(abs(lateral_speed), abs(forward_speed) + 1e-6)
        up_z = float(rotation[2, 2])
        return (
            np.asarray(position),
            np.asarray(orientation),
            np.asarray(linear),
            np.asarray(angular),
            forward_speed,
            lateral_speed,
            speed,
            slip_angle,
            up_z,
        )

    def _observation(self):
        position, orientation, linear, angular, forward, lateral, speed, slip, up_z = (
            self._kinematics()
        )
        return np.concatenate(
            (
                position,
                orientation,
                linear,
                angular,
                (forward, lateral, speed, slip, up_z),
            )
        ).astype(np.float32)

    def _apply_action(self, action):
        steering_command = float(np.clip(action[0], -1.0, 1.0))
        steering = steering_command * 0.5
        throttle = float(np.clip(action[1], -1.0, 1.0))
        target_velocity = throttle * 40.0
        for joint in self.WHEEL_JOINTS:
            p.setJointMotorControl2(
                self.car_id,
                joint,
                p.VELOCITY_CONTROL,
                targetVelocity=target_velocity,
                force=self.motor_force,
                physicsClientId=self.client,
            )
        for joint in self.STEERING_JOINTS:
            p.setJointMotorControl2(
                self.car_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=steering,
                force=20.0,
                physicsClientId=self.client,
            )
        return steering_command, throttle

    def step(self, action):
        self.steps += 1
        steering, throttle = self._apply_action(action)
        for _ in range(self.CONTROL_SUBSTEPS):
            p.applyExternalTorque(
                self.car_id,
                -1,
                (0, 0, 3.0 * steering * abs(throttle)),
                p.LINK_FRAME,
                physicsClientId=self.client,
            )
            p.stepSimulation(physicsClientId=self.client)

        observation = self._observation()
        _, _, _, angular, forward, lateral, speed, slip, up_z = self._kinematics()
        speed_factor = min(speed / 8.0, 1.0)
        slip_reward = speed_factor * math.sin(2.0 * slip)
        spin_ratio = min(abs(float(angular[2])) / (speed + 0.5), 2.0)
        spin_penalty = 0.05 * spin_ratio
        low_speed_penalty = 0.02 if speed < 1.0 else 0.0
        flipped = up_z < 0.5
        crashed = flipped or observation[2] < -0.1
        crash_penalty = 5.0 if crashed else 0.0
        reward = slip_reward - spin_penalty - low_speed_penalty - crash_penalty
        truncated = self.steps >= self.MAX_STEPS
        info = {
            "slip_angle": slip,
            "slip_angle_degrees": math.degrees(slip),
            "speed": speed,
            "forward_speed": forward,
            "lateral_speed": lateral,
            "slip_reward": slip_reward,
            "spin_penalty": spin_penalty,
            "flipped": bool(flipped),
            "safe": bool(not crashed),
        }
        return observation, float(reward), bool(crashed), truncated, info

    def render(self):
        position, _ = p.getBasePositionAndOrientation(
            self.car_id, physicsClientId=self.client
        )
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=position,
            distance=6.0,
            yaw=45,
            pitch=-35,
            roll=0,
            upAxisIndex=2,
        )
        projection = p.computeProjectionMatrixFOV(60, 4.0 / 3.0, 0.1, 50.0)
        _, _, rgba, _, _ = p.getCameraImage(
            640,
            480,
            view,
            projection,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client,
        )
        return np.asarray(rgba, dtype=np.uint8)[..., :3]

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)
