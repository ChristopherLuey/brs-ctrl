import re
import os
import numpy as np
import pybullet as pb

from brs_ctrl.asset_root import ASSET_ROOT


class R1ProKinematics:
    """Kinematics model for R1Pro robot using PyBullet."""

    torso_joint_high = np.array([1.8326, 2.5307, 1.5708, 3.0543])
    torso_joint_low = np.array([-1.1345, -2.7925, -1.8326, -3.0543])
    left_arm_joint_high = np.array(
        [1.3090, 3.1416, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708]
    )
    left_arm_joint_low = np.array(
        [-4.4506, -0.1745, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708]
    )
    right_arm_joint_high = np.array(
        [1.3090, 0.1745, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708]
    )
    right_arm_joint_low = np.array(
        [-4.4506, -3.1416, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708]
    )

    def __init__(self):
        urdf_path = os.path.join(ASSET_ROOT, "robot", "r1_pro", "r1_pro.urdf")

        self._pb_client_id = pb.connect(pb.DIRECT)
        self._pb_robot_id = pb.loadURDF(
            str(urdf_path),
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=self._pb_client_id,
        )
        pb.resetBasePositionAndOrientation(
            self._pb_robot_id,
            [0, 0, 0],
            [0, 0, 0, 1],
            physicsClientId=self._pb_client_id,
        )

        self._pb_num_joints = pb.getNumJoints(
            self._pb_robot_id, physicsClientId=self._pb_client_id
        )
        for i in range(self._pb_num_joints):
            pb.resetJointState(
                self._pb_robot_id, i, 0, physicsClientId=self._pb_client_id
            )

        self._left_arm_joint_idxs = [[] for _ in range(7)]
        self._right_arm_joint_idxs = [[] for _ in range(7)]
        self._torso_joint_idxs = [[] for _ in range(4)]

        self._link_name_to_index = {
            pb.getBodyInfo(self._pb_robot_id, physicsClientId=self._pb_client_id)[
                0
            ].decode("UTF-8"): -1,
        }

        left_joint_pattern = re.compile(r"left_arm_joint[1-7]")
        right_joint_pattern = re.compile(r"right_arm_joint[1-7]")
        torso_joint_pattern = re.compile(r"torso_joint[1-4]")

        for _id in range(
            pb.getNumJoints(self._pb_robot_id, physicsClientId=self._pb_client_id)
        ):
            joint_info = pb.getJointInfo(
                self._pb_robot_id, _id, physicsClientId=self._pb_client_id
            )
            joint_name = joint_info[1].decode("UTF-8")
            if left_joint_pattern.match(joint_name):
                idx = int(joint_name[-1]) - 1
                self._left_arm_joint_idxs[idx].append(_id)
            elif right_joint_pattern.match(joint_name):
                idx = int(joint_name[-1]) - 1
                self._right_arm_joint_idxs[idx].append(_id)
            elif torso_joint_pattern.match(joint_name):
                idx = int(joint_name[-1]) - 1
                self._torso_joint_idxs[idx].append(_id)
            link_name = joint_info[12].decode("UTF-8")
            self._link_name_to_index[link_name] = _id
        self._left_arm_joint_idxs = [idx[0] for idx in self._left_arm_joint_idxs]
        self._right_arm_joint_idxs = [idx[0] for idx in self._right_arm_joint_idxs]
        self._torso_joint_idxs = [idx[0] for idx in self._torso_joint_idxs]

        self._odom2base_link = None

    @property
    def T_odom2base(self):
        """Get the transformation matrix from T265 odometry frame to robot base_link."""
        if self._odom2base_link is None:
            odom_frame_idx = self._link_name_to_index["t265_pose_tracking_frame"]
            ls = pb.getLinkState(
                self._pb_robot_id, odom_frame_idx, physicsClientId=self._pb_client_id
            )
            odom_position, odom_quaternion = np.array(ls[0]), ls[1]
            odom_rotation_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    odom_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            self._odom2base_link = np.eye(4)
            self._odom2base_link[:3, :3] = odom_rotation_matrix
            self._odom2base_link[:3, 3] = odom_position
        return self._odom2base_link

    def get_link_poses_in_base_link(
        self,
        *,
        left_eef_link_name: str = "left_gripper_link",
        right_eef_link_name: str = "right_gripper_link",
        curr_left_arm_joint: np.ndarray,
        curr_right_arm_joint: np.ndarray,
        curr_torso_joint: np.ndarray,
        return_matrix: bool = True,
    ):
        """
        Get the pose of end effectors in the base link frame.
        """
        assert (
            curr_left_arm_joint.ndim == 1 and len(curr_left_arm_joint) == 7
        ), "Must provide 7 left arm joint angles."
        assert np.all(self.left_arm_joint_low <= curr_left_arm_joint) and np.all(
            curr_left_arm_joint <= self.left_arm_joint_high
        ), "Left arm joint angles out of range."

        assert (
            curr_right_arm_joint.ndim == 1 and len(curr_right_arm_joint) == 7
        ), "Must provide 7 right arm joint angles."
        assert np.all(self.right_arm_joint_low <= curr_right_arm_joint) and np.all(
            curr_right_arm_joint <= self.right_arm_joint_high
        ), "Right arm joint angles out of range."

        assert (
            curr_torso_joint.ndim == 1 and len(curr_torso_joint) == 4
        ), "Must provide 4 torso joint angles."
        assert np.all(self.torso_joint_low <= curr_torso_joint) and np.all(
            curr_torso_joint <= self.torso_joint_high
        ), "Torso joint angles out of range."

        for idx, q in zip(self._left_arm_joint_idxs, curr_left_arm_joint):
            pb.resetJointState(
                self._pb_robot_id, idx, q, physicsClientId=self._pb_client_id
            )

        for idx, q in zip(self._right_arm_joint_idxs, curr_right_arm_joint):
            pb.resetJointState(
                self._pb_robot_id, idx, q, physicsClientId=self._pb_client_id
            )

        for idx, q in zip(self._torso_joint_idxs, curr_torso_joint):
            pb.resetJointState(
                self._pb_robot_id, idx, q, physicsClientId=self._pb_client_id
            )

        # get link poses
        left_eef_link_idx = self._link_name_to_index[left_eef_link_name]
        right_eef_link_idx = self._link_name_to_index[right_eef_link_name]

        left_eef_ls, right_eef_ls = pb.getLinkStates(
            self._pb_robot_id,
            [left_eef_link_idx, right_eef_link_idx],
            physicsClientId=self._pb_client_id,
        )
        left_eef_position, left_eef_quaternion = (
            np.array(left_eef_ls[0]),
            left_eef_ls[1],
        )
        right_eef_position, right_eef_quaternion = (
            np.array(right_eef_ls[0]),
            right_eef_ls[1],
        )

        if not return_matrix:
            return {
                "left_eef": (left_eef_position, left_eef_quaternion),
                "right_eef": (right_eef_position, right_eef_quaternion),
            }
        else:
            left_eef_rotation_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    left_eef_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            T_left_eef = np.eye(4)
            T_left_eef[:3, :3] = left_eef_rotation_matrix
            T_left_eef[:3, 3] = left_eef_position

            right_eef_rotation_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    right_eef_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            T_right_eef = np.eye(4)
            T_right_eef[:3, :3] = right_eef_rotation_matrix
            T_right_eef[:3, 3] = right_eef_position

            return {
                "left_eef": T_left_eef,
                "right_eef": T_right_eef,
            }
