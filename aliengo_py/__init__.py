# aliengo_py/__init__.py
import numpy as np
from typing import List
from . import aliengo_py

Robot = aliengo_py.Robot
AliengoCommand = aliengo_py.AliengoCommand
AliengoState = aliengo_py.AliengoState

ORBIT_JOINT_ORDER = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]

SDK_JOINT_ORDER = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


# fmt: off
default_joint_pos = np.array(
    [
        # 0.1, -0.1, 0.1, -0.1,
        # 0.8, 0.8, 0.8, 0.8,
        # -1.5, -1.5, -1.5, -1.5,

        0.2, -0.2, 0.2, -0.2,
        1.0, 1.0, 1.0, 1.0,
        -1.8, -1.8, -1.8, -1.8,
    ],
)
# fmt on


def normalize(v: np.ndarray):
    return v / np.linalg.norm(v)


def mix(a: np.ndarray, b: np.ndarray, alpha: float):
    return a * (1 - alpha) + b * alpha

def wrap_to_pi(a: np.ndarray):
    # wrap to -pi to pi
    a = np.mod(a + np.pi, 2 * np.pi) - np.pi
    return a

def orbit_to_sdk(joints: np.ndarray):
    return np.flip(joints.reshape(3, 2, 2), axis=2).transpose(1, 2, 0).reshape(-1)

def sdk_to_orbit(joints: np.ndarray):
    return np.flip(joints.reshape(2, 2, 3), axis=1).transpose(2, 0, 1).reshape(-1)

class CommandManager:

    command_dim: int

    def __init__(self, robot: Robot) -> None:
        self._robot = robot
        self.command = np.zeros(self.command_dim)

    def update(self):
        pass


class JoyStickFlat(CommandManager):

    command_dim = 4

    max_speed_x = 1.0
    max_speed_y = 0.5
    max_angvel = 1.0

    def update(self):
        robot_state = self._robot.get_state()
        self.command[0] = mix(self.command[0], robot_state.lxy[1] * self.max_speed_x, 0.2)
        self.command[1] = mix(
            self.command[1], -robot_state.lxy[0] * self.max_speed_y, 0.2
        )
        self.command[2] = mix(
            self.command[2], -robot_state.rxy[0] * self.max_angvel, 0.2
        )
        self.command[3] = 0.0


class FixedCommandForce(CommandManager):
    command_dim = 10

    setpoint_pos_b = np.array([1.0, 0.0, 0.0])
    yaw_diff = 0.0

    kp = 5.0
    kd = 3.0
    virtual_mass = 10.0

    def update(self):
        self.command[:2] = self.setpoint_pos_b[:2]
        self.command[2] = self.yaw_diff
        self.command[3:5] = self.kp * self.setpoint_pos_b[:2]
        self.command[5:8] = self.kd
        self.command[8] = self.kp * self.yaw_diff
        self.command[9] = self.virtual_mass
        
class JoyStickForce_xy_yaw(FixedCommandForce):
    max_pos_x_b = 1.0
    max_pos_y_b = 1.0
    max_angvel = 0.7

    def __init__(self, robot: aliengo_py.Robot) -> None:
        super().__init__(robot)
        self._init_yaw = self._robot.get_state().rpy[2]
        self._target_yaw = 0.0
        self._angvel = 0.0
    
    def update(self):
        robot_state = self._robot.get_state()

        self._angvel = mix(self._angvel, -robot_state.rxy[0] * self.max_angvel, 0.2)
        self._target_yaw += self._angvel * 0.02

        self.yaw_diff = wrap_to_pi(self._target_yaw + self._init_yaw - robot_state.rpy[2])
        self.yaw_diff = 0
        self.setpoint_pos_b[0] = mix(self.setpoint_pos_b[0], robot_state.lxy[1] * self.max_pos_x_b, 0.2)
        self.setpoint_pos_b[1] = mix(self.setpoint_pos_b[1], -robot_state.lxy[0] * self.max_pos_y_b, 0.2)
        
        super().update()
    
class JoyStickForce_xy_kp(FixedCommandForce):
    max_pos_x_b = 1.0
    max_pos_y_b = 1.0
    max_kp = 10.0
    min_kp = 2.0

    def update(self):
        robot_state = self._robot.get_state()
        
        self.kp = (robot_state.rxy[1] + 1) / 2 * (self.max_kp - self.min_kp) + self.min_kp
        
        self.setpoint_pos_b[0] = mix(self.setpoint_pos_b[0], robot_state.lxy[1] * self.max_pos_x_b, 0.2)
        self.setpoint_pos_b[1] = mix(self.setpoint_pos_b[1], -robot_state.lxy[0] * self.max_pos_y_b, 0.2)
        
        super().update()


class ActionManager:

    action_dim: int

    def __init__(self, robot: Robot) -> None:
        self._robot = robot
        self.robot_cmd = AliengoCommand()
        self.action = np.zeros(self.action_dim)

    def update(self):
        pass

    def step(self, action: np.ndarray) -> None:
        raise NotImplementedError


class JointPositionAction(ActionManager):

    action_dim = 12

    kp: float = 60
    kd: float = 2

    def __init__(
        self,
        robot: Robot,
        clip_joint_targets: float = 1.6,
        alpha: float = 0.9,
        action_scaling: float = 0.5,
    ) -> None:
        super().__init__(robot)

        self.robot_cmd.jpos_des = default_joint_pos.tolist()
        self.robot_cmd.jvel_des = [0.0] * 12
        self.robot_cmd.kp = [self.kp] * 12
        self.robot_cmd.kd = [self.kd] * 12
        self.robot_cmd.tau_ff = [0.0] * 12

        self.clip_joint_targets = clip_joint_targets
        self.alpha = alpha
        self.action_scaling = action_scaling

    def step(self, action_sim: np.ndarray) -> np.ndarray:
        action_sim = action_sim.clip(-6, 6)
        self.action = mix(self.action, action_sim, self.alpha)
        jpos_target = (self.action * self.action_scaling).clip(
            -self.clip_joint_targets, self.clip_joint_targets
        ) + default_joint_pos
        self.robot_cmd.jpos_des = orbit_to_sdk(jpos_target).tolist()
        # print(self.robot_cmd.jpos_des)
        self._robot.set_command(self.robot_cmd)
        return action_sim


class EnvBase:

    dt: float = 0.02

    def __init__(
        self,
        robot: Robot,
        command_manager: CommandManager,
        action_manager: ActionManager,
    ) -> None:
        self.robot = robot
        self.robot_state = AliengoState()

        self.command_manager = command_manager
        self.action_manager = action_manager

    def reset(self):
        self.update()
        self.command_manager.update()
        return self.compute_obs()

    def apply_action(self, action: np.ndarray):
        raise NotImplementedError
        action = self.action_manager.step(action)
        self.action_buf[:, 1:] = self.action_buf[:, :-1]
        self.action_buf[:, 0] = action

        self.update()
        self.command_manager.update()
        return self.compute_obs()

    def update(self):
        """Update the environment state buffers."""
        raise NotImplementedError

    def compute_obs(self):
        """Compute the observation from the environment state buffers."""
        raise NotImplementedError


class FlatEnv(EnvBase):

    smoothing_length: int = 5
    smoothing_ratio: float = 0.4

    action_buf_steps = 3

    def __init__(
        self,
        robot: Robot,
        command_manager: CommandManager,
        action_manager: ActionManager,
    ) -> None:
        super().__init__(
            robot=robot,
            command_manager=command_manager,
            action_manager=action_manager,
        )

        # obs
        self.jpos_sdk = np.zeros(12)
        self.jvel_sdk = np.zeros(12)

        self.jpos_sim = np.zeros(12)
        self.jvel_sim = np.zeros(12)

        self.action_buf = np.zeros((action_manager.action_dim, self.action_buf_steps))

        # self.rpy = np.zeros(3)
        # self.angvel_history = np.zeros((3, self.smoothing_length))
        # self.angvel = np.zeros(3)
        self.projected_gravity_history = np.zeros((3, self.smoothing_length))
        self.projected_gravity = np.zeros(3)

    def update(self):
        self.robot_state = self.robot.get_state()

        self.jpos_sdk[:] = self.robot_state.jpos
        self.jvel_sdk[:] = self.robot_state.jvel

        self.jpos_sim[:] = sdk_to_orbit(self.jpos_sdk)
        self.jvel_sim[:] = sdk_to_orbit(self.jvel_sdk)

        # self.prev_rpy = self.rpy
        # self.rpy[:] = self.robot_state.rpy

        # self.angvel_history[:] = np.roll(self.angvel_history, 1, axis=1)
        # self.angvel_history[:, 0] = self.robot_state.gyro
        # self.angvel = mix(
        #     self.angvel, self.angvel_history.mean(axis=1), self.smoothing_ratio
        # )

        self.projected_gravity_history[:] = np.roll(
            self.projected_gravity_history, 1, axis=1
        )
        self.projected_gravity_history[:, 0] = self.robot_state.projected_gravity
        self.projected_gravity[:] = normalize(self.projected_gravity_history.mean(1))

        # TODO: add latency measurements
        # self.latency = (datetime.datetime.now() - self._robot.timestamp).total_seconds()
        # self.timestamp = time.perf_counter()

    def compute_obs(self):
        self.update()
        self.command_manager.update()

        obs = [
            self.command_manager.command,
            self.projected_gravity,
            self.jpos_sim,
            self.jvel_sim,
            self.action_buf[:, : self.action_buf_steps].reshape(-1),
        ]
        obs = np.concatenate(obs, dtype=np.float32)
        return obs
    
    def apply_action(self, action_sim: np.ndarray = None):
        if action_sim is not None:
            self.action_manager.step(action_sim)
            self.action_buf[:, 1:] = self.action_buf[:, :-1]
            self.action_buf[:, 0] = action_sim
