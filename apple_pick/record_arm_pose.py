#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


DEFAULT_TOPIC = "/vx300s/joint_states"
DEFAULT_JOINT_ORDER = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]


class PoseRecorder(Node):
    def __init__(self, topic: str, joint_order: list[str], timeout_sec: float):
        super().__init__("record_arm_pose")
        self._topic = topic
        self._joint_order = joint_order
        self._timeout_sec = timeout_sec
        self._latest_msg: JointState | None = None
        self.subscription = self.create_subscription(JointState, topic, self._callback, 10)

    def _callback(self, msg: JointState):
        self._latest_msg = msg

    def wait_for_pose(self) -> JointState:
        deadline = time.monotonic() + self._timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_msg is not None:
                return self._latest_msg
        raise TimeoutError(f"No message received on {self._topic} within {self._timeout_sec}s")

    def extract_joint_positions(self, msg: JointState) -> list[float]:
        if not msg.name:
            raise RuntimeError("JointState message contains no joint names")

        positions = []
        for name in self._joint_order:
            if name not in msg.name:
                raise RuntimeError(f"Joint '{name}' not found in {msg.name}")
            idx = msg.name.index(name)
            positions.append(float(msg.position[idx]))
        return positions


def parse_args():
    parser = argparse.ArgumentParser(description="Record the current arm joint positions from ROS2")
    parser.add_argument("--name", default="saved_pose", help="Name to label the saved pose")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="Joint state topic")
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for a joint state message")
    parser.add_argument("--joint-order", nargs="+", default=DEFAULT_JOINT_ORDER, help="Joint names in the order used by your MoveIt planner")
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init(args=None)
    node = PoseRecorder(args.topic, args.joint_order, args.timeout)

    try:
        msg = node.wait_for_pose()
        positions = node.extract_joint_positions(msg)
    except Exception as exc:
        print(f"Error: {exc}")
        rclpy.shutdown()
        sys.exit(1)

    print(f"Pose name: {args.name}")
    print("Joint names:", args.joint_order)
    print("Joint positions:", positions)
    print()
    print("Copy this into your script:")
    print(f"    return self.go_to_joint_positions({positions}, \"{args.name}\")")
    print()
    print("Or save it as a YAML-like snippet:")
    print(f"{args.name}: {positions}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
