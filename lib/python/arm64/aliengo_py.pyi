# robot_interface.pyi

from typing import List

class Robot:
    def __init__(self, control_freq: int) -> None: ...
    def start_control(self) -> None: ...
    def get_state(self) -> AliengoState: ...
    def set_command(self, command: AliengoCommand) -> None: ...

class AliengoState:
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

class AliengoCommand:
    jpos_des: List[float]
    jvel_des: List[float]
    kp: List[float]
    kd: List[float]
    tau_ff: List[float]
