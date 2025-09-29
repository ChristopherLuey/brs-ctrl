import os
import time
import pybullet as pb
import numpy as np
from pathlib import Path

from brs_ctrl.asset_root import ASSET_ROOT
from brs_ctrl.joylo import JoyLoController
from brs_ctrl.joylo.joylo_arms import R1ProJoyLoArmPositionController
from brs_ctrl.joylo.joycon import R1ProJoyConInterface


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
    )
    joycon = R1ProJoyConInterface()
    joylo = JoyLoController(joycon=joycon, joylo_arms=joylo_arms)

    pb_client = pb.connect(pb.GUI)
    pb.setGravity(0, 0, -9.8)
    # load robot
    robot = pb.loadURDF(
        os.path.join(ASSET_ROOT, "robot/r1_pro/r1_pro.urdf"),
        [0, 0, 0],
        useFixedBase=True,
    )

    # reset all joints to 0
    for i in range(pb.getNumJoints(robot)):
        pb.resetJointState(robot, i, 0)

    torso_joint_idxs = [13, 14, 15, 16]
    left_arm_joint_idxs = [22, 23, 24, 25, 26, 27, 28]
    right_arm_joint_idxs = [34, 35, 36, 37, 38, 39, 40]

    curr_torso_qs = np.array([0.0, 0.0, 0.0, 0.0])

    while pb.isConnected():
        joylo_actions = joylo.act(curr_torso_qs)

        for i, q in zip(torso_joint_idxs, joylo_actions["torso_cmd"]):
            pb.resetJointState(robot, i, q)
        for i, q in enumerate(joylo_actions["arm_cmd"]["left"]):
            pb.resetJointState(robot, left_arm_joint_idxs[i], q)
        for i, q in enumerate(joylo_actions["arm_cmd"]["right"]):
            pb.resetJointState(robot, right_arm_joint_idxs[i], q)

        pb.stepSimulation()

        curr_torso_qs = []
        for idx in torso_joint_idxs:
            curr_torso_qs.append(pb.getJointState(robot, idx)[0])
        curr_torso_qs = np.array(curr_torso_qs)
        time.sleep(0.05)
