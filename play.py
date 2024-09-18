import os
import sys

root_path = os.path.dirname(os.path.abspath(__file__))
print(root_path)
sys.path.append(os.path.join(root_path, "lib/python/arm64"))
import aliengo_py
from example_py.example_stand import stand

from typing import Optional

import time
import datetime
import numpy as np
import math
import torch
import itertools
import argparse

from scipy.spatial.transform import Rotation as R
from tensordict import TensorDict

from setproctitle import setproctitle

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

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


def normalize(v: np.ndarray):
    return v / np.linalg.norm(v)


class FlatEnv:

    smoothing_length: int = 5
    smoothing_ratio: float = 0.4

    max_speed = 1.0
    max_angvel = 1.0

    action_scaling = 0.5

    kp = 60.0
    kd = 2.0

    control_freq = 500

    clip_joint_targets = 1.6
    alpha = 0.8

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

    def __init__(self):
        self._robot = aliengo_py.Robot(self.control_freq)
        self._robot.start_control()

        self.dt = 0.02
        self.latency = 0.0
        self.robot_state = aliengo_py.AliengoState()
        self.robot_cmd = aliengo_py.AliengoCommand()

        self.robot_cmd.jpos_des = self.default_joint_pos.tolist()
        self.robot_cmd.jvel_des = [0.0] * 12
        self.robot_cmd.kp = [self.kp] * 12
        self.robot_cmd.kd = [self.kd] * 12
        self.robot_cmd.tau_ff = [0.0] * 12

        self.start_t = time.perf_counter()
        self.timestamp = time.perf_counter()
        self.step_count = 0

        # cmd
        self.lxy = np.zeros(2)
        self.rxy = np.zeros(2)
        self.command = np.zeros(4)  # linvel_xy, angvel_z, aux_input

        # obs
        self.jpos_sdk = np.zeros(12)
        self.jvel_sdk = np.zeros(12)

        self.jpos_sim = np.zeros(12)
        self.jvel_sim = np.zeros(12)

        self.rpy = np.zeros(3)
        self.angvel_history = np.zeros((3, self.smoothing_length))
        self.angvel = np.zeros(3)
        self.projected_gravity_history = np.zeros((3, self.smoothing_length))
        self.projected_gravity = np.zeros(3)

        # action
        self.action_buf_steps = 3
        self.action_buf = np.zeros((12, 4))
        self.last_action = np.zeros(12)

    def reset(self):
        self.start_t = time.perf_counter()
        self.update()
        return self._compute_obs()

    def update(self):
        self.robot_state = self._robot.get_state()

        self.prev_rpy = self.rpy
        self.rpy[:] = self.robot_state.rpy

        self.jpos_sdk[:] = self.robot_state.jpos
        self.jvel_sdk[:] = self.robot_state.jvel

        self.jpos_sim[:] = self.sdk_to_orbit(self.jpos_sdk)
        self.jvel_sim[:] = self.sdk_to_orbit(self.jvel_sdk)

        self.angvel_history[:] = np.roll(self.angvel_history, 1, axis=1)
        self.angvel_history[:, 0] = self.robot_state.gyro
        self.angvel = mix(
            self.angvel, self.angvel_history.mean(axis=1), self.smoothing_ratio
        )

        self.projected_gravity_history[:] = np.roll(
            self.projected_gravity_history, 1, axis=1
        )
        self.projected_gravity_history[:, 0] = self.robot_state.projected_gravity
        self.projected_gravity[:] = normalize(self.projected_gravity_history.mean(1))

        self.lxy = mix(self.lxy, np.array(self.robot_state.lxy), 0.5)
        self.rxy = mix(self.rxy, np.array(self.robot_state.rxy), 0.5)
        print("lxy", self.lxy)

        # TODO: add latency measurements
        # self.latency = (datetime.datetime.now() - self._robot.timestamp).total_seconds()
        # self.timestamp = time.perf_counter()

        self.command[0] = mix(self.command[0], self.lxy[1] * self.max_speed, 0.2)
        self.command[1] = mix(self.command[1], -self.lxy[0] * self.max_speed, 0.2)
        self.command[2] = -self.rxy[0] * self.max_angvel
        self.command[3] = 0.0

        # print("command", self.command)

    def step(self, action: Optional[np.ndarray] = None):
        if action is not None:
            self.action_buf[:, 1:] = self.action_buf[:, :-1]
            self.action_buf[:, 0] = action.clip(-6, 6)

            self.last_action = mix(self.last_action, self.action_buf[:, 0], self.alpha)
            jpos_target = (self.last_action * self.action_scaling).clip(
                -self.clip_joint_targets, self.clip_joint_targets
            ) + self.default_joint_pos

            self.robot_cmd.jpos_des = self.orbit_to_sdk(jpos_target)
            self._robot.set_command(self.robot_cmd)

        self.update()
        self.step_count += 1
        obs = self._compute_obs()

        return obs

    def _compute_obs(self):

        obs = [
            self.command,
            # angvel,
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

    def process_action(self, action: np.ndarray):
        return self.orbit_to_sdk(action * 0.5 + self.default_joint_pos)

    def process_action_inv(self, jpos_sdk: np.ndarray):
        return (self.sdk_to_orbit(jpos_sdk) - self.default_joint_pos) / 0.5


def mix(a, b, alpha):
    return a * (1 - alpha) + alpha * b


from torchrl.envs.utils import set_exploration_type, ExplorationType


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-l", "--log", action="store_true", default=False)
    # args = parser.parse_args()

    # timestr = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    setproctitle("play_aliengo")

    env = FlatEnv()
    print("Environment created")

    path = "policy-alienflat-397.pt"
    policy = torch.load(path)
    policy.module[0].set_missing_tolerance(True)
    # policy = lambda td: torch.zeros(12)

    stand(
        robot=env._robot,
        kp=env.kp,
        kd=env.kd,
        completion_time=5,
        default_joint_pos=env.orbit_to_sdk(env.default_joint_pos),
    )
    print("Robot is now in standing position. Press Enter to exit...")
    input()

    obs = env.reset()
    obs = env._compute_obs()
    print(obs.shape)
    print(policy)
    # policy.module.pop(0)

    policy_freq = 50
    dt = 1 / policy_freq

    try:
        td = TensorDict(
            {
                "policy": torch.as_tensor(obs),
                "is_init": torch.tensor(1, dtype=bool),
                "context_adapt_hx": torch.zeros(128),
            },
            [],
        ).unsqueeze(0)
        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            for i in itertools.count():
                start = time.perf_counter()
                try:
                    policy(td)
                except Exception as e:
                    print(e)
                    breakpoint()
                action = td["action"].squeeze(0).cpu().numpy()

                obs = torch.as_tensor(env.step(action))
                td["next", "policy"] = obs.unsqueeze(0)
                td["next", "is_init"] = torch.tensor([0], dtype=bool)

                # if i % 25 == 0:
                #     print(env.projected_gravity)
                # print(env.command)
                # print(robot.jpos_sdk.reshape(4, 3))
                # print(robot.sdk_to_orbit(robot.jpos_sdk).reshape(3, 4))

                td = td["next"]

                elapsed = time.perf_counter() - start
                # print(f"{i}: {elapsed:.4f}s")
                time.sleep(max(0, dt - elapsed))

    except KeyboardInterrupt:
        print("End")


if __name__ == "__main__":
    main()
