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
import h5py
import argparse

from scipy.spatial.transform import Rotation as R
from tensordict import TensorDict

from setproctitle import setproctitle

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


from torchrl.envs.utils import set_exploration_type, ExplorationType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("-l", "--log", action="store_true", default=False)
    args = parser.parse_args()

    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.log:
        os.makedirs("logs", exist_ok=True)
        log_file = h5py.File(f"logs/{timestr}.h5py", "a")
    else:
        log_file = None
    
    setproctitle("play_aliengo")
    control_freq=200

    robot = aliengo_py.Robot(control_freq)
    robot.start_control()

    command_manager = aliengo_py.JoyStickFlat(robot)
    action_manager = aliengo_py.JointPositionAction(robot, alpha=1.0)

    env = aliengo_py.FlatEnv(
        robot=robot,
        command_manager=command_manager,
        action_manager=action_manager,
        log_file=log_file   
    )
    print("Environment created")

    # path = "policy-alienflat-397.pt"
    path = "policy-alienflat-681.pt"
    action_manager.robot_cmd.kp = [40.0] * 12
    action_manager.robot_cmd.kd = [2.0] * 12
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
    # for i in itertools.count():
    #     state = robot.get_state()
    #     rpy = R.from_euler("xyz", state.rpy)

    #     gravity_rpy = rpy.inv().apply(np.array([0., 0., -1]))
    #     jpos_sdk = np.array(state.jpos)
    #     print(state.projected_gravity, gravity_rpy)
    #     # print(jpos_sdk.reshape(4, 3))
    #     time.sleep(0.1)
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
                action = td["action"].squeeze(0).numpy()
                env.apply_action(action)

                elapsed = time.perf_counter() - start
                # print(f"{i}: {elapsed:.4f}s")
                freq = 1 / max(elapsed, dt)
                if i % 20 == 0:
                    # print("command:", command_manager.command)
                    print(f"freq: {freq:.2f}Hz")
                    print(env.jpos_sim.reshape(3, 4))
                    print(action.reshape(3, 4))
                time.sleep(max(0, dt - elapsed))

    except KeyboardInterrupt:
        print("End")


if __name__ == "__main__":
    main()
