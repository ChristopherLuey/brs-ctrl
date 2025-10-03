import time
import numpy as np
from tqdm import tqdm
import rclpy
from brs_ctrl.joylo import JoyLoController
from brs_ctrl.joylo.joylo_arms import R1ProJoyLoArmPositionController
from brs_ctrl.joylo.joycon import R1ProJoyConInterface
from brs_ctrl.robot_interface import R1ProInterface
from brs_ctrl.robot_interface.grippers.galaxea_g1 import GalaxeaR1ProGripper


neutral_left_arm_qs = np.zeros(7)
neutral_right_arm_qs = np.zeros(7)


if __name__ == "__main__":
    joylo_arms = R1ProJoyLoArmPositionController(
        left_motor_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        right_motor_ids=[9, 10, 11, 12, 13, 14, 15, 16, 17],
        motors_port="/dev/tty_joylo_r1pro",
        left_arm_joint_signs=[-1, -1, -1, 1, -1, 1, 1],
        right_arm_joint_signs=[1, -1, -1, 1, -1, 1, 1],
        left_slave_motor_ids=[1, 3],
        left_master_motor_ids=[0, 2],
        right_slave_motor_ids=[10, 12],
        right_master_motor_ids=[9, 11],
        left_arm_joint_reset_positions=neutral_left_arm_qs,
        right_arm_joint_reset_positions=neutral_right_arm_qs,
        multithread_read_joints=True,
        baudrate=3000000,
    )
    joycon = R1ProJoyConInterface()
    joylo = JoyLoController(joycon=joycon, joylo_arms=joylo_arms)

    time.sleep(3)

    rclpy.init()
    node = R1ProInterface(
        left_gripper=GalaxeaR1ProGripper(
            left_or_right="left",
            gripper_close_stroke=0.0,
            gripper_open_stroke=100.0,
        ),
        right_gripper=GalaxeaR1ProGripper(
            left_or_right="right",
            gripper_close_stroke=0.0,
            gripper_open_stroke=100.0,
        ),
    )

    alpha = 0.8
    left_joylo_q = None
    right_joylo_q = None

    pbar = tqdm()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            try:
                assert node.last_joint_position is not None
            except Exception:
                print("No jon states yet")
                continue

            joylo_arms_q = joylo_arms.q
            left_joylo_q = (
                joylo_arms_q["left"]
                if left_joylo_q is None
                else (1 - alpha) * left_joylo_q + alpha * joylo_arms_q["left"]
            )
            right_joylo_q = (
                joylo_arms_q["right"]
                if right_joylo_q is None
                else (1 - alpha) * right_joylo_q + alpha * joylo_arms_q["right"]
            )
            curr_torso_qs = node.last_joint_position["torso"]
            joycon_action = joycon.act(curr_torso_qs)
            robot_torso_cmd = np.zeros((4,))
            robot_torso_cmd[:] = joycon_action["torso_cmd"][:]

            node.control(
                arm_cmd={"left": left_joylo_q, "right": right_joylo_q},
                torso_cmd=robot_torso_cmd,
                gripper_cmd={
                    "left": joycon_action["gripper_cmd"]["left"],
                    "right": joycon_action["gripper_cmd"]["right"],
                },
                base_cmd=joycon_action["mobile_base_cmd"],
            )
            time.sleep(0.01)
            pbar.update(1)

    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()
        joylo.close()
        pbar.close()
