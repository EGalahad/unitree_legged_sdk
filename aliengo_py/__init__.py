# robot_interface.pyi
import numpy as np
from typing import List


default_joint_pos = np.array(
    [
        0.2,
        -0.2,
        0.2,
        -0.2,
        0.8,
        0.8,
        0.8,
        0.8,
        -1.5,
        -1.5,
        -1.5,
        -1.5,
    ],
)


class _AliengoState:
    jpos: List[float]
    jvel: List[float]
    jtorque: List[float]
    quat: List[float]
    rpy: List[float]
    gyro: List[float]
    projected_gravity: List[float]
    lxy: List[float]
    rxy: List[float]
    control_mode: int


class _AliengoCommand:
    jpos_des: List[float]
    jvel_des: List[float]
    kp: List[float]
    kd: List[float]
    tau_ff: List[float]


class AliengoRobot:
    def __init__(self, control_freq: int) -> None: ...
    def start_control(self) -> None: ...
    def get_state(self) -> _AliengoState: ...
    def set_command(self, command: _AliengoCommand) -> None: ...


def normalize(v: np.ndarray):
    return v / np.linalg.norm(v)


def mix(a: np.ndarray, b: np.ndarray, alpha: float):
    return a * (1 - alpha) + b * alpha


class CommandManager:

    command_dim: int

    def __init__(self, robot: AliengoRobot) -> None:
        self._robot = robot
        self.command = np.zeros(self.command_dim)

    def update(self):
        pass


class JoyStickFlat(CommandManager):

    command_dim = 4

    max_speed = 1.0
    max_angvel = 1.0

    def update(self):
        robot_state = self._robot.get_state()
        self.command[0] = mix(self.command[0], robot_state.lxy[1] * self.max_speed, 0.2)
        self.command[1] = mix(
            self.command[1], -robot_state.lxy[0] * self.max_speed, 0.2
        )
        self.command[2] = mix(
            self.command[2], -robot_state.rxy[0] * self.max_angvel, 0.2
        )
        self.command[3] = 0.0


class ActionManager:

    action_dim: int

    def __init__(self, robot: AliengoRobot) -> None:
        self._robot = robot
        self.robot_cmd = _AliengoCommand()
        self.action = np.zeros(self.action_dim)

    def update(self):
        pass

    def step(self, action: np.ndarray) -> None:
        raise NotImplementedError


class JointPositionAction(ActionManager):

    action_dim = 12

    def __init__(
        self,
        robot: AliengoRobot,
        clip_joint_targets: float = 1.6,
        alpha: float = 0.8,
        action_scaling: float = 0.5,
    ) -> None:
        super().__init__(robot)

        self.robot_cmd.jpos_des = default_joint_pos.tolist()
        self.robot_cmd.jvel_des = [0.0] * 12
        self.robot_cmd.kp = [60.0] * 12
        self.robot_cmd.kd = [2.0] * 12
        self.robot_cmd.tau_ff = [0.0] * 12

        self.clip_joint_targets = clip_joint_targets
        self.alpha = alpha
        self.action_scaling = action_scaling

    def step(self, action: np.ndarray) -> np.ndarray:
        action = action.clip(-6, 6)
        self.action = mix(self.action, action, self.alpha)
        jpos_target = (self.action * self.action_scaling).clip(
            -self.clip_joint_targets, self.clip_joint_targets
        ) + default_joint_pos
        self.robot_cmd.jpos_des = jpos_target.tolist()
        self._robot.set_command(self.robot_cmd)
        return action


class EnvBase:

    dt: float = 0.02

    kp = 60.0
    kd = 2.0

    def __init__(
        self,
        robot: AliengoRobot,
        command_manager: CommandManager,
        action_manager: ActionManager,
    ) -> None:
        self.robot = robot
        self.robot_state = _AliengoState()

        self.command_manager = command_manager
        self.action_manager = action_manager
        self.action_buf = np.zeros((action_manager.action_dim, 4))

    def reset(self):
        self.update()
        self.command_manager.update()
        return self._compute_obs()

    def step(self, action: np.ndarray):
        action = self.action_manager.step(action)
        self.action_buf[:, 1:] = self.action_buf[:, :-1]
        self.action_buf[:, 0] = action

        self.update()
        self.command_manager.update()
        return self._compute_obs()

    def update(self):
        """Update the environment state buffers."""
        raise NotImplementedError

    def _compute_obs(self):
        """Compute the observation from the environment state buffers."""
        raise NotImplementedError


class FlatEnv(EnvBase):

    smoothing_length: int = 5
    smoothing_ratio: float = 0.4

    action_buf_steps = 3

    def __init__(
        self,
        robot: AliengoRobot,
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

        # self.rpy = np.zeros(3)
        # self.angvel_history = np.zeros((3, self.smoothing_length))
        # self.angvel = np.zeros(3)
        self.projected_gravity_history = np.zeros((3, self.smoothing_length))
        self.projected_gravity = np.zeros(3)

    def update(self):
        self.robot_state = self.robot.get_state()

        self.jpos_sdk[:] = self.robot_state.jpos
        self.jvel_sdk[:] = self.robot_state.jvel

        self.jpos_sim[:] = self.sdk_to_orbit(self.jpos_sdk)
        self.jvel_sim[:] = self.sdk_to_orbit(self.jvel_sdk)

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

    def _compute_obs(self):

        obs = [
            self.command_manager.command,
            self.projected_gravity,
            self.jpos_sim,
            self.jvel_sim,
            self.action_buf[:, : self.action_buf_steps].reshape(-1),
        ]
        obs = np.concatenate(obs, dtype=np.float32)
        return obs

    @staticmethod
    def orbit_to_sdk(joints: np.ndarray):
        return np.flip(joints.reshape(3, 2, 2), axis=2).transpose(1, 2, 0).reshape(-1)

    @staticmethod
    def sdk_to_orbit(joints: np.ndarray):
        return np.flip(joints.reshape(2, 2, 3), axis=1).transpose(2, 0, 1).reshape(-1)
