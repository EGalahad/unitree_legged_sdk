import os
import sys

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


from torchrl.envs.utils import set_exploration_type, ExplorationType


def main():
    setproctitle("play_aliengo")
    control_freq=500
    robot = aliengo_py.Robot(control_freq)
    robot.start_control()
    command_manager = aliengo_py.JoyStickFlat(robot)
    action_manager = aliengo_py.JointPositionAction(robot)

    env = aliengo_py.FlatEnv(
        robot=robot,
        command_manager=command_manager,
        action_manager=action_manager,       
    )
    print("Environment created")

    # path = "policy-alienflat-397.pt"
    path = "policy-alienflat-681.pt"
    action_manager.kp = 40.0
    action_manager.robot_cmd.kp = [40.0] * 12
    policy = torch.load(path)
    policy.module[0].set_missing_tolerance(True)
    # policy = lambda td: torch.zeros(12)

    stand(
        robot=env.robot,
        kp=action_manager.kp,
        kd=action_manager.kd,
        completion_time=5,
        default_joint_pos=aliengo_py.orbit_to_sdk(aliengo_py.default_joint_pos),
    )
    print("Robot is now in standing position. Press Enter to exit...")
    input()

    obs = env.reset()
    obs = env.compute_obs()
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

                obs = torch.as_tensor(env.compute_obs())
                td["next", "policy"] = obs.unsqueeze(0)
                td["next", "is_init"] = torch.tensor([0], dtype=bool)

                td = td["next"]

                try:
                    policy(td)
                except Exception as e:
                    print(e)
                    breakpoint()
                action = td["action"].squeeze(0).cpu().numpy()

                env.apply_action(action)

                elapsed = time.perf_counter() - start
                # print(f"{i}: {elapsed:.4f}s")
                freq = 1 / max(elapsed, dt)
                if i % 20 == 0:
                    print("command:", command_manager.command)
                    print(f"freq: {freq:.2f}Hz")
                time.sleep(max(0, dt - elapsed))

    except KeyboardInterrupt:
        print("End")


if __name__ == "__main__":
    main()
