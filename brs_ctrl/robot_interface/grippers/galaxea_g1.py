from typing import Any, Union, Literal

try:
    import rospy
except ImportError as e:
    print(f"Failed to import ROS related modules, GalaxeaR1Gripper won't work.")
    print(e)

try:
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
except ImportError as e:
    print(f"Failed to import ROS2 related modules, GalaxeaR1Gripper won't work.")
    print(e)


import numpy as np
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState

import brs_ctrl.utils as U
from brs_ctrl.robot_interface.grippers.base import R1BaseGripper, R1ProBaseGripper


class GalaxeaR1Gripper(R1BaseGripper):
    def __init__(
        self,
        left_or_right: Literal["left", "right"],
        gripper_position_control_topic: str = "/motion_control/position_control_gripper_{left_or_right}",
        gripper_feedback_topic: str = "/hdas/feedback_gripper_{left_or_right}",
        gripper_close_stroke: float = 10.0,
        gripper_open_stroke: float = 90.0,
        publisher_queue_size: int = 1,
        state_buffer_size: int = 1000,
    ):
        assert left_or_right in ["left", "right"]
        self._gripper_position_control_topic = gripper_position_control_topic.format(
            left_or_right=left_or_right
        )
        self._gripper_feedback_topic = gripper_feedback_topic.format(
            left_or_right=left_or_right
        )
        self._gripper_close_stroke = gripper_close_stroke
        self._gripper_open_stroke = gripper_open_stroke
        self._publisher_queue_size = publisher_queue_size
        self._state_buffer_size = state_buffer_size

        self._gripper_position_control_pub = None
        self._gripper_feedback_sub = None
        self._gripper_state_buffer = None

    def init_hook(self):
        self._gripper_position_control_pub = rospy.Publisher(
            self._gripper_position_control_topic,
            Float32,
            queue_size=self._publisher_queue_size,
            latch=True,
        )
        self._gripper_feedback_sub = rospy.Subscriber(
            self._gripper_feedback_topic,
            JointState,
            self._gripper_state_callback,
        )

    def act(self, action: Union[float, np.ndarray]):
        assert isinstance(action, float) or isinstance(
            action, int
        ), "GalaxeaG1Gripper only support a single number as action"
        stroke = self._gripper_close_stroke + (1 - action) * (
            self._gripper_open_stroke - self._gripper_close_stroke
        )
        gripper_msg = Float32()
        gripper_msg.data = max(min(stroke, 100), 0)
        self._gripper_position_control_pub.publish(gripper_msg)

    def get_state(self, data: JointState) -> Any:
        new_state = {
            "gripper_position": np.array([data.position[0]]),
            "gripper_velocity": np.array([data.velocity[0]]),
            "gripper_effort": np.array([data.effort[0]]),
            "seq": np.array([data.header.seq]),
            "stamp": np.array(
                [data.header.stamp.secs + data.header.stamp.nsecs * 1e-9]
            ),
        }
        return new_state

    def close(self):
        pass  # ros topic publisher closed outside

    def _gripper_state_callback(self, data: JointState):
        new_state = self.get_state(data)
        if self._gripper_state_buffer is None:
            self._gripper_state_buffer = new_state
            return
        self._gripper_state_buffer = U.any_concat(
            [
                self._gripper_state_buffer,
                new_state,
            ],
            dim=0,
        )
        self._gripper_state_buffer = U.any_slice(
            self._gripper_state_buffer, np.s_[-self._state_buffer_size :]
        )

    @property
    def state_buffer(self):
        return self._gripper_state_buffer


class GalaxeaR1ProGripper(R1ProBaseGripper):
    def __init__(
        self,
        left_or_right: Literal["left", "right"],
        gripper_position_control_topic: str = "/motion_target/target_position_gripper_{left_or_right}",
        gripper_feedback_topic: str = "/hdas/feedback_gripper_{left_or_right}",
        gripper_close_stroke: float = 10.0,
        gripper_open_stroke: float = 90.0,
        publisher_queue_size: int = 1,
        state_buffer_size: int = 1000,
    ):
        assert left_or_right in ["left", "right"]
        self._gripper_position_control_topic = gripper_position_control_topic.format(
            left_or_right=left_or_right
        )
        self._gripper_feedback_topic = gripper_feedback_topic.format(
            left_or_right=left_or_right
        )
        self._gripper_close_stroke = gripper_close_stroke
        self._gripper_open_stroke = gripper_open_stroke
        self._publisher_queue_size = publisher_queue_size
        self._state_buffer_size = state_buffer_size

        self._node: Optional[Node] = None
        self._gripper_position_control_pub = None
        self._gripper_feedback_sub = None
        self._gripper_state_buffer = None

    def init_hook(self, node: Node):
        """
        Initialize ROS 2 publishers/subscribers using the provided Node.
        Usage: gripper.init_hook(self)  # from within your Node subclass
        """
        self._node = node

        # "Latched" in ROS 2 => TRANSIENT_LOCAL durability
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, self._publisher_queue_size),
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._gripper_position_control_pub = self._node.create_publisher(
            JointState, self._gripper_position_control_topic, cmd_qos
        )

        # Sensor QoS for feedback
        self._gripper_feedback_sub = self._node.create_subscription(
            JointState,
            self._gripper_feedback_topic,
            self._gripper_state_callback,
            qos_profile_sensor_data,
        )

    def act(self, action: Union[float, np.ndarray]):
        assert isinstance(
            action, (float, int)
        ), "ParallelGripper expects a single scalar action"
        stroke = self._gripper_close_stroke + (1 - float(action)) * (
            self._gripper_open_stroke - self._gripper_close_stroke
        )
        gripper_msg = JointState()
        # keep within [0,100]
        gripper_msg.position = [float(max(min(stroke, 100.0), 0.0))]
        self._gripper_position_control_pub.publish(gripper_msg)

    def get_state(self, data: JointState) -> Any:
        # ROS 2 header has sec/nanosec; there is no seq
        stamp = data.header.stamp.sec + data.header.stamp.nanosec * 1e-9

        # Guard in case arrays are empty
        p = data.position[0] if len(data.position) > 0 else np.nan
        v = data.velocity[0] if len(data.velocity) > 0 else np.nan
        e = data.effort[0] if len(data.effort) > 0 else np.nan

        new_state = {
            "gripper_position": np.array([p], dtype=float),
            "gripper_velocity": np.array([v], dtype=float),
            "gripper_effort": np.array([e], dtype=float),
            "seq": np.array(
                [data.header.stamp.nanosec]
            ),  # placeholder; ROS 2 has no seq
            "stamp": np.array([stamp], dtype=float),
        }
        return new_state

    def close(self):
        # Pub/sub objects are owned by the Node; let the parent Node handle destroy/shutdown.
        self._gripper_position_control_pub = None
        self._gripper_feedback_sub = None
        self._node = None

    def _gripper_state_callback(self, data: JointState):
        new_state = self.get_state(data)
        if self._gripper_state_buffer is None:
            self._gripper_state_buffer = new_state
            return
        self._gripper_state_buffer = U.any_concat(
            [self._gripper_state_buffer, new_state], dim=0
        )
        self._gripper_state_buffer = U.any_slice(
            self._gripper_state_buffer, np.s_[-self._state_buffer_size :]
        )

    @property
    def state_buffer(self):
        return self._gripper_state_buffer
