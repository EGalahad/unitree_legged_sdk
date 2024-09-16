#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <boost/bind.hpp>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mutex>

#include "unitree_legged_sdk/unitree_joystick.h"
#include "unitree_legged_sdk/unitree_legged_sdk.h"

namespace py = pybind11;

constexpr uint16_t TARGET_PORT = 8007;
constexpr uint16_t LOCAL_PORT = 8082;
constexpr char TARGET_IP[] = "192.168.123.10";  // target IP address

const int LOW_CMD_LENGTH = 610;
const int LOW_STATE_LENGTH = 771;

namespace UNITREE_LEGGED_SDK {

class AliengoState {
   public:
    std::array<float, 12> jpos{};
    std::array<float, 12> jvel{};
    std::array<float, 12> jtorque{};
    std::array<float, 4> quat{};
    std::array<float, 3> rpy{};
    std::array<float, 3> gyro{};
    std::array<float, 3> projected_gravity{0, 0, -1};
    std::array<float, 2> lxy{};
    std::array<float, 2> rxy{};
    int control_mode = 0;  // 0: high-level, 1: low-level
};

class AliengoCommand {
   public:
    std::array<float, 12> jpos_des{0.0};
    std::array<float, 12> jvel_des{0.0};
    std::array<float, 12> kp{0.0};
    std::array<float, 12> kd{2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                             2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    std::array<float, 12> tau_ff{0.0};
};

class AliengoInterface {
   public:
    AliengoInterface(int control_freq = 500, int control_mode = 1)
        : udp(LOCAL_PORT, TARGET_IP, TARGET_PORT, LOW_CMD_LENGTH,
              LOW_STATE_LENGTH) {
        udp.InitCmdData(low_cmd);
        low_cmd.levelFlag = LOWLEVEL;

        udp.InitCmdData(high_cmd);
        // high_cmd.levelFlag = HIGHLEVEL;
        high_cmd.mode = 0;
        high_cmd.rpy[0] = 0;
        high_cmd.rpy[1] = 0;
        high_cmd.rpy[2] = 0;
        high_cmd.gaitType = 0;
        high_cmd.speedLevel = 0;
        high_cmd.dFootRaiseHeight = 0;
        high_cmd.dBodyHeight = 0;

        robot_state.control_mode = control_mode;
        udp.switchLevel(control_mode ? LOWLEVEL : HIGHLEVEL);
        dt = 1.0 / control_freq;
    }

    void start_control() {
        std::cout << "start control" << std::endl;
        InitEnvironment();
        std::cout << "init done" << std::endl;

        loop_control = new LoopFunc(
            "control", dt, boost::bind(&AliengoInterface::control_loop, this));
        loop_recv = new LoopFunc(
            "udp_recv", dt, 3, boost::bind(&AliengoInterface::udp_recv, this));
        loop_send = new LoopFunc(
            "udp_send", dt, 3, boost::bind(&AliengoInterface::udp_send, this));

        loop_control->start();
        loop_recv->start();
        loop_send->start();
    }

    AliengoState get_robot_state() {
        // std::lock_guard<std::mutex> lock(state_mutex);
        return robot_state;
    }

    void set_robot_command(const AliengoCommand& command) {
        // std::lock_guard<std::mutex> lock(cmd_mutex);
        robot_command = command;
    }

   private:
    void udp_recv() { udp.Recv(); }
    void udp_send() { udp.Send(); }
    void control_loop() {
        std::cout << "in control loop" << std::endl;
        if (robot_state.control_mode == 1) {
            std::cout << "in low level mode" << std::endl;
            udp.GetRecv(low_state);
            update_state_low();
            std::cout << "lxy: " << robot_state.lxy[0] << " "
                      << robot_state.lxy[1] << std::endl;
            update_command();
            udp.SetSend(low_cmd);
        } else {
            std::cout << "in high level mode" << std::endl;
            // update_state_high();
            udp.GetRecv(low_state);
            
            xRockerBtnDataStruct joystick_data;
            memcpy(&joystick_data, &low_state.wirelessRemote,
                   sizeof(xRockerBtnDataStruct));
            robot_state.lxy = {joystick_data.lx, joystick_data.ly};
            robot_state.rxy = {joystick_data.rx, joystick_data.ry};

            std::cout << "low state: " << std::endl;
            // cout lxy
            std::cout << "lxy: " << robot_state.lxy[0] << " "
                      << robot_state.lxy[1] << std::endl;

            // cout L1 state
            if (joystick_data.btn.components.L1)
                std::cout << "l1 pressed." << std::endl;

            udp.GetRecv(high_state);
            memcpy(&joystick_data, &high_state.wirelessRemote,
                   sizeof(xRockerBtnDataStruct));
            robot_state.lxy = {joystick_data.lx, joystick_data.ly};
            robot_state.rxy = {joystick_data.rx, joystick_data.ry};

            std::cout << "high state: " << std::endl;
            // cout lxy
            std::cout << "lxy: " << robot_state.lxy[0] << " "
                      << robot_state.lxy[1] << std::endl;

            // cout L1 state
            if (joystick_data.btn.components.L1)
                std::cout << "l1 pressed." << std::endl;

            udp.SetSend(high_cmd);
        }
    }

    void update_state_low() {
        // std::lock_guard<std::mutex> lock(state_mutex);

        // IMU data
        robot_state.quat = {
            low_state.imu.quaternion[0], low_state.imu.quaternion[1],
            low_state.imu.quaternion[2], low_state.imu.quaternion[3]};
        robot_state.rpy = {low_state.imu.rpy[0], low_state.imu.rpy[1],
                           low_state.imu.rpy[2]};
        robot_state.gyro = {low_state.imu.gyroscope[0],
                            low_state.imu.gyroscope[1],
                            low_state.imu.gyroscope[2]};
        update_projected_gravity();

        // Joint states
        for (int i = 0; i < 12; ++i) {
            robot_state.jpos[i] = low_state.motorState[i].q;
            robot_state.jvel[i] = low_state.motorState[i].dq;
            robot_state.jtorque[i] = low_state.motorState[i].tauEst;
        }
        // std::cout << "jpos: ";
        // for (int i = 0; i < 12; ++i) {
        //     std::cout << robot_state.jpos[i] << " ";
        // }
        // std::cout << std::endl;

        // Joystick data
        xRockerBtnDataStruct joystick_data;
        memcpy(&joystick_data, &low_state.wirelessRemote,
               sizeof(xRockerBtnDataStruct));
        robot_state.lxy = {joystick_data.lx, joystick_data.ly};
        robot_state.rxy = {joystick_data.rx, joystick_data.ry};

        // cout lxy
        std::cout << "lxy: " << robot_state.lxy[0] << " " << robot_state.lxy[1]
                  << std::endl;

        // cout L1 state
        if (joystick_data.btn.components.L1)
            std::cout << "l1 pressed." << std::endl;

        // Check for mode switch (e.g., when L1 is pressed)
        // TODO: maybe need to store last L1 state, only change mode on press
        // if (joystick_data.btn.components.L1) {
        //     robot_state.control_mode =
        //         1 - robot_state.control_mode;  // Toggle between 0 and 1
        //     udp.switchLevel(robot_state.control_mode ? LOWLEVEL : HIGHLEVEL);
        //     std::cout << "Switch control mode: " << robot_state.control_mode
        //               << std::endl;
        // }
    }

    void update_state_high() {
        // Joystick data
        xRockerBtnDataStruct joystick_data;
        memcpy(&joystick_data, &high_state.wirelessRemote,
               sizeof(xRockerBtnDataStruct));
        robot_state.lxy = {joystick_data.lx, joystick_data.ly};
        robot_state.rxy = {joystick_data.rx, joystick_data.ry};

        // cout lxy
        std::cout << "lxy: " << robot_state.lxy[0] << " " << robot_state.lxy[1]
                  << std::endl;

        // cout L1 state
        if (joystick_data.btn.components.L1)
            std::cout << "l1 pressed." << std::endl;

        // Check for mode switch (e.g., when L1 is pressed)
        // TODO: maybe need to store last L1 state, only change mode on press
        if (joystick_data.btn.components.L1) {
            robot_state.control_mode =
                1 - robot_state.control_mode;  // Toggle between 0 and 1
            udp.switchLevel(robot_state.control_mode ? LOWLEVEL : HIGHLEVEL);
            std::cout << "Switch control mode: " << robot_state.control_mode
                      << std::endl;
        }
    }

    void update_command() {
        // std::lock_guard<std::mutex> lock(cmd_mutex);
        for (int i = 0; i < 12; ++i) {
            low_cmd.motorCmd[i].q = robot_command.jpos_des[i];
            low_cmd.motorCmd[i].dq = robot_command.jvel_des[i];
            low_cmd.motorCmd[i].Kp = robot_command.kp[i];
            low_cmd.motorCmd[i].Kd = robot_command.kd[i];
            low_cmd.motorCmd[i].tau = robot_command.tau_ff[i];
        }
    }

    void update_projected_gravity() {
        float w = robot_state.quat[0], x = -robot_state.quat[1],
              y = -robot_state.quat[2], z = -robot_state.quat[3];
        float x2 = x * x, y2 = y * y, z2 = z * z, w2 = w * w;
        robot_state.projected_gravity[0] = 2 * (x * z - w * y);
        robot_state.projected_gravity[1] = 2 * (y * z + w * x);
        robot_state.projected_gravity[2] = w2 - x2 - y2 + z2;
    }

    static uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
        uint32_t xbit = 0;
        uint32_t data = 0;
        uint32_t CRC32 = 0xFFFFFFFF;
        const uint32_t dwPolynomial = 0x04c11db7;

        for (uint32_t i = 0; i < len; i++) {
            xbit = 1 << 31;
            data = ptr[i];
            for (uint32_t bits = 0; bits < 32; bits++) {
                if (CRC32 & 0x80000000) {
                    CRC32 = (CRC32 << 1) ^ dwPolynomial;
                } else {
                    CRC32 <<= 1;
                }
                if (data & xbit) {
                    CRC32 ^= dwPolynomial;
                }
                xbit >>= 1;
            }
        }
        return CRC32;
    }

    UDP udp;
    float dt;  // 0.001 ~ 0.01

    LowCmd low_cmd;
    LowState low_state;
    HighCmd high_cmd;
    HighState high_state;

    AliengoState robot_state;
    AliengoCommand robot_command;
    std::mutex state_mutex, cmd_mutex;

    LoopFunc* loop_control = nullptr;
    LoopFunc* loop_recv = nullptr;
    LoopFunc* loop_send = nullptr;
};

}  // namespace UNITREE_LEGGED_SDK

PYBIND11_MODULE(aliengo_py, m) {
    py::class_<UNITREE_LEGGED_SDK::AliengoInterface>(m, "Robot")
        .def(py::init<int, int>())
        .def("start_control",
             &UNITREE_LEGGED_SDK::AliengoInterface::start_control)
        .def("get_state",
             &UNITREE_LEGGED_SDK::AliengoInterface::get_robot_state)
        .def("set_command",
             &UNITREE_LEGGED_SDK::AliengoInterface::set_robot_command);

    py::class_<UNITREE_LEGGED_SDK::AliengoState>(m, "AliengoState")
        .def(py::init<>())
        .def_readwrite("jpos", &UNITREE_LEGGED_SDK::AliengoState::jpos)
        .def_readwrite("jvel", &UNITREE_LEGGED_SDK::AliengoState::jvel)
        .def_readwrite("jtorque", &UNITREE_LEGGED_SDK::AliengoState::jtorque)
        .def_readwrite("quat", &UNITREE_LEGGED_SDK::AliengoState::quat)
        .def_readwrite("rpy", &UNITREE_LEGGED_SDK::AliengoState::rpy)
        .def_readwrite("gyro", &UNITREE_LEGGED_SDK::AliengoState::gyro)
        .def_readwrite("projected_gravity",
                       &UNITREE_LEGGED_SDK::AliengoState::projected_gravity)
        .def_readwrite("lxy", &UNITREE_LEGGED_SDK::AliengoState::lxy)
        .def_readwrite("rxy", &UNITREE_LEGGED_SDK::AliengoState::rxy)
        .def_readwrite("control_mode",
                       &UNITREE_LEGGED_SDK::AliengoState::control_mode);

    py::class_<UNITREE_LEGGED_SDK::AliengoCommand>(m, "AliengoCommand")
        .def(py::init<>())
        .def_readwrite("jpos_des",
                       &UNITREE_LEGGED_SDK::AliengoCommand::jpos_des)
        .def_readwrite("jvel_des",
                       &UNITREE_LEGGED_SDK::AliengoCommand::jvel_des)
        .def_readwrite("kp", &UNITREE_LEGGED_SDK::AliengoCommand::kp)
        .def_readwrite("kd", &UNITREE_LEGGED_SDK::AliengoCommand::kd)
        .def_readwrite("tau_ff", &UNITREE_LEGGED_SDK::AliengoCommand::tau_ff);
}
