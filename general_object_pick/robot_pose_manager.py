#!/usr/bin/env python3
"""MoveIt observation-pose manager for the general object picking UI."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

import rclpy
import zmq
from builtin_interfaces.msg import Duration as DurationMsg
from controller_manager_msgs.srv import SwitchController
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from interbotix_xs_msgs.msg import JointSingleCommand
from interbotix_xs_msgs.srv import OperatingModes
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints,
    DisplayTrajectory,
    JointConstraint,
    MoveItErrorCodes,
    RobotState,
)
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_TOPIC = "/vx300s/joint_states"
JOINT_ORDER = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]
COMMAND_ENDPOINT = "tcp://127.0.0.1:5561"
STATUS_ENDPOINT = "tcp://127.0.0.1:5562"
POSE_FILE = Path(__file__).resolve().parent / "config" / "observation_poses.json"
BASE_FRAME = "vx300s/base_link"
END_EFFECTOR_LINK = "vx300s/ee_gripper_link"
OPEN_PWM = 350.0
CLOSE_PWM = -650.0
OPEN_HOLD_SECONDS = 1.25
CLOSE_HOLD_SECONDS = 1.3
GRIPPER_SETTLE_SECONDS = 0.5
ARM_SETTLE_SECONDS = 2.5
PREGRASP_Z_LIFT_M = 0.05
APPROACH_DISTANCE_M = 0.03
# On this physical VX300S, left_finger feedback decreases while the PWM open
# command is applied and increases while the close command is applied.  These
# values were confirmed from the real hardware instead of inferred from the
# MoveIt named-state values.
OPEN_FINGER_MAX_M = 0.025
CLOSED_FINGER_MIN_M = 0.030
FINGER_MOVEMENT_MIN_M = 0.002


class RobotPoseManager(Node):
    def __init__(self) -> None:
        super().__init__("general_object_pick_robot_manager")
        self._lock = threading.Lock()
        self._latest_joints: list[float] | None = None
        self._latest_finger_position: float | None = None
        self._latest_joint_time = 0.0
        self._planned_name: str | None = None
        self._planned_trajectory = None
        self._plan_start_joints: list[float] | None = None
        self._pick_trajectories: list = []
        self._pick_start_joints: list[float] | None = None
        self._pick_pregrasp_joints: list[float] | None = None
        self._pick_grasp_joints: list[float] | None = None
        self._pick_target: dict | None = None
        self._holding_object = False
        self._gripper_state = "unknown"
        self._active_goal_handle = None
        self._active_command: str | None = None
        self._cancel_requested = threading.Event()
        self._last_result = "Robot manager connected."
        self._result_sequence = 1
        self._result_history = deque(maxlen=30)
        self._result_history.append(
            {
                "sequence": self._result_sequence,
                "time": time.strftime("%H:%M:%S"),
                "message": self._last_result,
            }
        )

        self.create_subscription(JointState, JOINT_TOPIC, self._joint_callback, 20)
        self._move_group = ActionClient(self, MoveGroup, "/move_action")
        self._execute_trajectory = ActionClient(
            self, ExecuteTrajectory, "/execute_trajectory"
        )
        self._arm_controller = ActionClient(
            self,
            FollowJointTrajectory,
            "/vx300s/arm_controller/follow_joint_trajectory",
        )
        self._gripper_controller = ActionClient(
            self,
            FollowJointTrajectory,
            "/vx300s/gripper_controller/follow_joint_trajectory",
        )
        self._display_publisher = self.create_publisher(
            DisplayTrajectory, "/display_planned_path", 10
        )
        self._compute_ik = self.create_client(GetPositionIK, "/compute_ik")
        self._set_operating_modes = self.create_client(
            OperatingModes, "/vx300s/set_operating_modes"
        )
        self._switch_controllers = self.create_client(
            SwitchController, "/vx300s/controller_manager/switch_controller"
        )
        self._gripper_pwm = self.create_publisher(
            JointSingleCommand, "/vx300s/commands/joint_single", 10
        )
        self._ensure_pose_file()
        self._gripper_position_controller_disabled = False

    def _joint_callback(self, message: JointState) -> None:
        positions = {name: value for name, value in zip(message.name, message.position)}
        if not all(name in positions for name in JOINT_ORDER):
            return
        with self._lock:
            self._latest_joints = [float(positions[name]) for name in JOINT_ORDER]
            if "left_finger" in positions:
                self._latest_finger_position = float(positions["left_finger"])
            self._latest_joint_time = time.monotonic()

    @staticmethod
    def _ensure_pose_file() -> None:
        POSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not POSE_FILE.exists():
            RobotPoseManager._write_pose_data(
                {"schema_version": 1, "joint_order": JOINT_ORDER, "poses": {}}
            )

    @staticmethod
    def _read_pose_data() -> dict:
        with POSE_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if data.get("joint_order") != JOINT_ORDER or not isinstance(data.get("poses"), dict):
            raise ValueError("Observation pose file has an incompatible format")
        return data

    @staticmethod
    def _write_pose_data(data: dict) -> None:
        temporary = POSE_FILE.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
        os.replace(temporary, POSE_FILE)

    def current_joints(self) -> list[float] | None:
        with self._lock:
            if self._latest_joints is None or time.monotonic() - self._latest_joint_time > 1.0:
                return None
            return list(self._latest_joints)

    def current_finger_position(self) -> float | None:
        with self._lock:
            if (
                self._latest_finger_position is None
                or time.monotonic() - self._latest_joint_time > 1.0
            ):
                return None
            return float(self._latest_finger_position)

    def status(self) -> dict:
        with self._lock:
            active_command = self._active_command
            last_result = self._last_result
            result_sequence = self._result_sequence
            planned_name = self._planned_name
            pick_plan_ready = bool(self._pick_trajectories)
            holding_object = self._holding_object
            gripper_state = self._gripper_state
            result_history = list(self._result_history)
        try:
            poses = sorted(self._read_pose_data()["poses"])
        except Exception:
            poses = []
        return {
            "moveit_ready": self._move_group.server_is_ready()
            and self._execute_trajectory.server_is_ready(),
            "joint_state_ready": self.current_joints() is not None,
            "arm_controller_ready": self._arm_controller.server_is_ready(),
            "gripper_ready": self._set_operating_modes.service_is_ready(),
            "active_command": active_command,
            "last_result": last_result,
            "result_sequence": result_sequence,
            "planned_name": planned_name,
            "pick_plan_ready": pick_plan_ready,
            "holding_object": holding_object,
            "gripper_state": gripper_state,
            "finger_position_m": self.current_finger_position(),
            "result_history": result_history,
            "poses": poses,
        }

    def save_pose(self, name: str) -> dict:
        name = name.strip()
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in name):
            return {"success": False, "message": "Pose name may contain letters, numbers, _ and - only."}
        joints = self.current_joints()
        if joints is None:
            return {"success": False, "message": f"No recent joint state on {JOINT_TOPIC}."}
        data = self._read_pose_data()
        data["poses"][name] = {
            "joint_positions": joints,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self._write_pose_data(data)
        self._invalidate_plan()
        message = f"Saved observation pose '{name}'."
        self._set_result(message)
        return {"success": True, "message": message, "poses": sorted(data["poses"])}

    def plan_pose(self, name: str) -> dict:
        if self._active_command is not None:
            return {"success": False, "message": "Another robot command is active."}
        data = self._read_pose_data()
        pose = data["poses"].get(name)
        if pose is None:
            return {"success": False, "message": f"Unknown observation pose '{name}'."}
        start_joints = self.current_joints()
        if start_joints is None:
            return {"success": False, "message": "Current joint state is unavailable."}
        if not self._move_group.wait_for_server(timeout_sec=2.0):
            return {"success": False, "message": "MoveGroup action server is unavailable."}

        self._invalidate_plan()
        self._active_command = "plan"
        try:
            goal = MoveGroup.Goal()
            goal.request.group_name = "interbotix_arm"
            goal.request.num_planning_attempts = 5
            goal.request.allowed_planning_time = 5.0
            goal.request.max_velocity_scaling_factor = 0.10
            goal.request.max_acceleration_scaling_factor = 0.10
            goal.request.goal_constraints = [
                self._joint_constraints(pose["joint_positions"])
            ]
            goal.request.start_state.is_diff = True
            goal.planning_options.plan_only = True

            goal_handle = self._await(self._move_group.send_goal_async(goal), 10.0)
            if goal_handle is None or not goal_handle.accepted:
                raise RuntimeError("MoveGroup rejected the planning request.")
            self._active_goal_handle = goal_handle
            result_wrapper = self._await(goal_handle.get_result_async(), 35.0)
            self._active_goal_handle = None
            if result_wrapper is None:
                raise RuntimeError("MoveIt planning timed out.")
            result = result_wrapper.result
            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                raise RuntimeError(f"MoveIt planning failed with code {result.error_code.val}.")

            self._planned_name = name
            self._planned_trajectory = result.planned_trajectory
            self._plan_start_joints = start_joints
            display = DisplayTrajectory()
            display.trajectory_start = result.trajectory_start
            display.trajectory = [result.planned_trajectory]
            self._display_publisher.publish(display)
            message = f"Plan succeeded for '{name}'. Review the trajectory in RViz."
            self._set_result(message)
            return {"success": True, "message": message, "planned_name": name}
        except Exception as error:
            self._invalidate_plan()
            message = str(error)
            self._set_result(message)
            return {"success": False, "message": message}
        finally:
            self._active_command = None

    def begin_execute(self, name: str) -> dict:
        if self._active_command is not None:
            return {"success": False, "message": "Another robot command is active."}
        if name != self._planned_name or self._planned_trajectory is None:
            return {"success": False, "message": "Run Plan Only for this pose first."}
        current = self.current_joints()
        if current is None or self._plan_start_joints is None:
            return {"success": False, "message": "Current joint state is unavailable."}
        if max(abs(a - b) for a, b in zip(current, self._plan_start_joints)) > 0.05:
            self._invalidate_plan()
            return {"success": False, "message": "Robot moved after planning. Run Plan Only again."}
        self._active_command = "execute"
        threading.Thread(target=self._execute_worker, args=(name,), daemon=True).start()
        return {"success": True, "message": f"Execution accepted for '{name}'."}

    def plan_pick(self, target: dict, target_label: str, profile: str) -> dict:
        """Plan the proven apple approach without moving any hardware."""
        if self._active_command is not None:
            return {"success": False, "message": "Another robot command is active."}
        if profile != "apple":
            return {
                "success": False,
                "message": "Only the tested apple grasp profile is enabled in this version.",
            }
        if target_label.strip().lower() != "apple":
            return {
                "success": False,
                "message": "Select target object 'apple' for the apple grasp profile.",
            }
        try:
            x, y, z = (float(target[key]) for key in ("x", "y", "z"))
        except (KeyError, TypeError, ValueError):
            return {"success": False, "message": "The target coordinate is invalid."}
        if not self._within_workspace(x, y, z):
            return {"success": False, "message": "Target is outside the configured workspace."}
        start_joints = self.current_joints()
        if start_joints is None:
            return {"success": False, "message": "Current joint state is unavailable."}
        if not self._move_group.wait_for_server(timeout_sec=2.0):
            return {"success": False, "message": "MoveGroup action server is unavailable."}
        if not self._compute_ik.wait_for_service(timeout_sec=2.0):
            return {"success": False, "message": "MoveIt IK service is unavailable."}

        distance = math.hypot(x, y)
        if distance < 0.10:
            return {"success": False, "message": "Target is too close to the robot base."}
        yaw = math.atan2(y, x)
        pre = self._make_pose(x, y, z + PREGRASP_Z_LIFT_M, yaw)
        grasp = self._make_pose(
            x + APPROACH_DISTANCE_M * x / distance,
            y + APPROACH_DISTANCE_M * y / distance,
            z + PREGRASP_Z_LIFT_M,
            yaw,
        )
        if not self._within_workspace(
            grasp.pose.position.x, grasp.pose.position.y, grasp.pose.position.z
        ):
            return {"success": False, "message": "Calculated grasp point is outside the workspace."}

        self._active_command = "plan_pick"
        self._invalidate_plan()
        try:
            pre_joints = self._solve_ik(pre, start_joints, "pregrasp")
            pre_result = self._plan_joint_segment(start_joints, pre_joints, "pregrasp")
            pre_end = self._trajectory_endpoint(pre_result.planned_trajectory)

            grasp_joints = self._solve_ik(grasp, pre_end, "grasp")
            approach_result = self._plan_joint_segment(pre_end, grasp_joints, "approach")
            grasp_end = self._trajectory_endpoint(approach_result.planned_trajectory)
            retreat_result = self._plan_joint_segment(grasp_end, pre_joints, "retreat")
            retreat_end = self._trajectory_endpoint(retreat_result.planned_trajectory)
            return_result = self._plan_joint_segment(
                retreat_end, start_joints, "return to start"
            )

            self._pick_trajectories = [
                pre_result.planned_trajectory,
                approach_result.planned_trajectory,
                retreat_result.planned_trajectory,
                return_result.planned_trajectory,
            ]
            self._pick_start_joints = start_joints
            self._pick_pregrasp_joints = list(pre_joints)
            self._pick_grasp_joints = list(grasp_joints)
            self._pick_target = {"x": x, "y": y, "z": z, "label": target_label}
            display = DisplayTrajectory()
            display.trajectory_start = pre_result.trajectory_start
            display.trajectory = list(self._pick_trajectories)
            self._display_publisher.publish(display)
            message = (
                "Straight-pick plan succeeded: pregrasp, 3 cm approach, retreat, and "
                "return to the original start pose. Review all four segments in RViz."
            )
            self._set_result(message)
            return {"success": True, "message": message, "pick_plan_ready": True}
        except Exception as error:
            self._invalidate_pick_plan()
            message = str(error)
            self._set_result(message)
            return {"success": False, "message": message}
        finally:
            self._active_command = None

    def begin_pick(self, current_target: dict) -> dict:
        if self._active_command is not None:
            return {"success": False, "message": "Another robot command is active."}
        if (
            len(self._pick_trajectories) != 4
            or self._pick_start_joints is None
            or self._pick_pregrasp_joints is None
            or self._pick_grasp_joints is None
        ):
            return {"success": False, "message": "Run Plan Straight Pick first."}
        try:
            current_xyz = [float(current_target[key]) for key in ("x", "y", "z")]
            planned_xyz = [float(self._pick_target[key]) for key in ("x", "y", "z")]
        except (KeyError, TypeError, ValueError):
            return {"success": False, "message": "The current target is unavailable. Plan again."}
        if max(abs(a - b) for a, b in zip(current_xyz, planned_xyz)) > 0.015:
            self._invalidate_pick_plan()
            return {"success": False, "message": "Target moved after planning. Plan the pick again."}
        current = self.current_joints()
        if current is None:
            return {"success": False, "message": "Current joint state is unavailable."}
        if max(abs(a - b) for a, b in zip(current, self._pick_start_joints)) > 0.05:
            self._invalidate_pick_plan()
            return {"success": False, "message": "Robot moved after planning. Plan the pick again."}
        self._cancel_requested.clear()
        self._active_command = "pick"
        threading.Thread(target=self._pick_worker, daemon=True).start()
        return {"success": True, "message": "Straight-pick execution accepted."}

    def _pick_worker(self) -> None:
        try:
            pregrasp_joints = list(self._pick_pregrasp_joints)
            grasp_joints = list(self._pick_grasp_joints)
            start_joints = list(self._pick_start_joints)
            self._set_result("Pick step 1/8: opening the gripper before arm movement.")
            self._command_gripper(OPEN_PWM, OPEN_HOLD_SECONDS, "open")
            self._set_result("Pick step 2/8: moving to the pregrasp pose.")
            self._runtime_plan_and_execute(pregrasp_joints, "pregrasp")
            self._set_result("Pick step 3/8: holding at pregrasp for 2.5 seconds.")
            self._wait_with_cancel(ARM_SETTLE_SECONDS)
            self._set_result("Pick step 4/8: approaching the object by 3 cm.")
            self._runtime_plan_and_execute(grasp_joints, "approach")
            self._set_result("Pick step 5/8: holding at the grasp point for 2.5 seconds.")
            self._wait_with_cancel(ARM_SETTLE_SECONDS)
            self._set_result("Pick step 6/8: closing the gripper.")
            self._command_gripper(CLOSE_PWM, CLOSE_HOLD_SECONDS, "close")
            self._holding_object = True
            self._set_result("Pick step 7/8: retreating to the pregrasp pose.")
            self._runtime_plan_and_execute(pregrasp_joints, "retreat")
            self._set_result("Pick step 8/8: returning to the original start pose.")
            self._runtime_plan_and_execute(start_joints, "return to start")
            self._set_result(
                "Straight pick completed. The arm returned to its original start pose and "
                "the gripper remains closed."
            )
        except Exception as error:
            self._set_result(str(error))
        finally:
            self._active_command = None
            self._invalidate_pick_plan()

    def begin_gripper_command(self, action: str) -> dict:
        if self._active_command is not None:
            return {"success": False, "message": "Another robot command is active."}
        if action not in {"open", "close"}:
            return {"success": False, "message": "Unknown gripper command."}
        self._cancel_requested.clear()
        self._active_command = f"gripper_{action}"
        threading.Thread(
            target=self._gripper_worker, args=(action,), daemon=True
        ).start()
        return {"success": True, "message": f"Gripper {action} command accepted."}

    def _gripper_worker(self, action: str) -> None:
        try:
            if action == "open":
                self._command_gripper(OPEN_PWM, OPEN_HOLD_SECONDS, "open")
                self._set_result("Gripper opened at the current arm position.")
            else:
                self._command_gripper(CLOSE_PWM, CLOSE_HOLD_SECONDS, "close")
                self._set_result("Gripper closed at the current arm position.")
        except Exception as error:
            self._set_result(str(error))
        finally:
            self._active_command = None

    def _execute_worker(self, name: str) -> None:
        try:
            if not self._execute_trajectory.wait_for_server(timeout_sec=2.0):
                raise RuntimeError("MoveIt execute trajectory server is unavailable.")
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = self._planned_trajectory
            goal_handle = self._await(self._execute_trajectory.send_goal_async(goal), 10.0)
            if goal_handle is None or not goal_handle.accepted:
                raise RuntimeError("MoveIt rejected trajectory execution.")
            self._active_goal_handle = goal_handle
            result_wrapper = self._await(goal_handle.get_result_async(), 60.0)
            self._active_goal_handle = None
            if result_wrapper is None:
                raise RuntimeError("Trajectory execution timed out.")
            error_code = result_wrapper.result.error_code.val
            if error_code != MoveItErrorCodes.SUCCESS:
                raise RuntimeError(f"Trajectory execution failed with code {error_code}.")
            self._set_result(f"Execution completed for '{name}'.")
        except Exception as error:
            self._set_result(str(error))
        finally:
            self._active_command = None
            self._invalidate_plan()

    def cancel(self) -> dict:
        self._cancel_requested.set()
        goal_handle = self._active_goal_handle
        if goal_handle is None:
            return {"success": True, "message": "No active MoveIt goal."}
        result = self._await(goal_handle.cancel_goal_async(), 3.0)
        message = "Cancel request sent." if result is not None else "Cancel request timed out."
        self._set_result(message)
        return {"success": result is not None, "message": message}

    def _solve_ik(self, pose: PoseStamped, seed: list[float], label: str) -> list[float]:
        request = GetPositionIK.Request()
        request.ik_request.group_name = "interbotix_arm"
        request.ik_request.ik_link_name = END_EFFECTOR_LINK
        request.ik_request.pose_stamped = pose
        request.ik_request.robot_state = self._robot_state(seed)
        request.ik_request.timeout = DurationMsg(sec=2, nanosec=0)
        response = self._await(self._compute_ik.call_async(request), 4.0)
        if response is None:
            raise RuntimeError(f"IK timed out for {label}.")
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"IK failed for {label} with code {response.error_code.val}.")
        by_name = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        if not all(name in by_name for name in JOINT_ORDER):
            raise RuntimeError(f"IK response for {label} omitted an arm joint.")
        return [float(by_name[name]) for name in JOINT_ORDER]

    def _plan_joint_segment(self, start: list[float], goal_joints: list[float], label: str):
        goal = MoveGroup.Goal()
        goal.request.group_name = "interbotix_arm"
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.10
        goal.request.max_acceleration_scaling_factor = 0.10
        goal.request.start_state = self._robot_state(start)
        goal.request.goal_constraints = [self._joint_constraints(goal_joints)]
        goal.planning_options.plan_only = True
        handle = self._await(self._move_group.send_goal_async(goal), 10.0)
        if handle is None or not handle.accepted:
            raise RuntimeError(f"MoveGroup rejected the {label} plan.")
        self._active_goal_handle = handle
        wrapper = self._await(handle.get_result_async(), 35.0)
        self._active_goal_handle = None
        if wrapper is None:
            raise RuntimeError(f"MoveIt timed out while planning {label}.")
        if wrapper.result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(
                f"MoveIt failed to plan {label} with code {wrapper.result.error_code.val}."
            )
        return wrapper.result

    def _execute_segment(self, trajectory, label: str) -> None:
        self._raise_if_cancelled()
        if not self._execute_trajectory.wait_for_server(timeout_sec=2.0):
            raise RuntimeError("MoveIt execute trajectory server is unavailable.")
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        handle = self._await(self._execute_trajectory.send_goal_async(goal), 10.0)
        if handle is None or not handle.accepted:
            raise RuntimeError(f"MoveIt rejected {label} execution.")
        self._active_goal_handle = handle
        wrapper = self._await(handle.get_result_async(), 60.0)
        self._active_goal_handle = None
        if wrapper is None:
            raise RuntimeError(f"Execution timed out during {label}.")
        error_code = wrapper.result.error_code.val
        if error_code != MoveItErrorCodes.SUCCESS:
            joint_error = self._trajectory_endpoint_error(trajectory)
            if error_code == MoveItErrorCodes.CONTROL_FAILED and joint_error <= 0.025:
                self._set_result(
                    f"Controller reported -4 during {label}, but the arm reached the "
                    f"planned endpoint (max joint error {joint_error:.4f} rad). Continuing."
                )
                return
            raise RuntimeError(
                f"Execution failed during {label} with code {error_code}; "
                f"max endpoint joint error was {joint_error:.4f} rad."
            )

    def _runtime_plan_and_execute(self, target_joints: list[float], label: str) -> None:
        """Use MoveGroup plan+execute, matching the proven legacy apple workflow."""
        self._raise_if_cancelled()
        if self.current_joints() is None:
            raise RuntimeError(f"Current joint state is unavailable before {label}.")
        if not self._move_group.wait_for_server(timeout_sec=2.0):
            raise RuntimeError("MoveGroup action server is unavailable.")

        goal = MoveGroup.Goal()
        goal.request.group_name = "interbotix_arm"
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.10
        goal.request.max_acceleration_scaling_factor = 0.10
        goal.request.goal_constraints = [self._joint_constraints(target_joints)]
        goal.request.start_state.is_diff = True
        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = False

        handle = self._await(self._move_group.send_goal_async(goal), 10.0)
        if handle is None or not handle.accepted:
            raise RuntimeError(f"MoveGroup rejected {label} execution.")
        self._active_goal_handle = handle
        wrapper = self._await(handle.get_result_async(), 75.0)
        self._active_goal_handle = None
        if wrapper is None:
            raise RuntimeError(f"MoveGroup timed out during {label}.")

        error_code = wrapper.result.error_code.val
        if error_code == MoveItErrorCodes.SUCCESS:
            return
        joint_error = self._joint_target_error(target_joints)
        if error_code == MoveItErrorCodes.CONTROL_FAILED and joint_error <= 0.025:
            self._set_result(
                f"MoveGroup reported -4 during {label}, but the arm reached the target "
                f"(max joint error {joint_error:.4f} rad). Continuing."
            )
            return
        raise RuntimeError(
            f"MoveGroup failed during {label} with code {error_code}; "
            f"max target joint error was {joint_error:.4f} rad."
        )

    def _joint_target_error(self, target_joints: list[float]) -> float:
        current = self.current_joints()
        if current is None:
            return float("inf")
        return max(abs(actual - goal) for actual, goal in zip(current, target_joints))

    def _trajectory_endpoint_error(self, trajectory) -> float:
        expected = self._trajectory_endpoint(trajectory)
        current = self.current_joints()
        if current is None:
            return float("inf")
        return max(abs(actual - goal) for actual, goal in zip(current, expected))

    def _set_gripper_pwm_mode(self) -> None:
        if not self._gripper_position_controller_disabled:
            if not self._switch_controllers.wait_for_service(timeout_sec=2.0):
                raise RuntimeError("Controller Manager switch service is unavailable.")
            switch = SwitchController.Request()
            switch.deactivate_controllers = ["gripper_controller"]
            switch.strictness = SwitchController.Request.BEST_EFFORT
            switch.activate_asap = True
            switch.timeout = DurationMsg(sec=2, nanosec=0)
            switch_response = self._await(
                self._switch_controllers.call_async(switch), 4.0
            )
            if switch_response is None or not switch_response.ok:
                raise RuntimeError(
                    "Could not disable the position gripper controller before PWM control."
                )
            self._gripper_position_controller_disabled = True
            self._set_result(
                "Position gripper controller disabled; UI now has exclusive PWM control."
            )
        if not self._set_operating_modes.wait_for_service(timeout_sec=2.0):
            raise RuntimeError("Gripper operating-mode service is unavailable.")
        request = OperatingModes.Request()
        request.cmd_type = "single"
        request.name = "gripper"
        request.mode = "pwm"
        request.profile_type = "velocity"
        response = self._await(self._set_operating_modes.call_async(request), 4.0)
        if response is None:
            raise RuntimeError("Timed out while setting gripper PWM mode.")

    def _command_gripper(self, pwm: float, duration: float, action: str) -> None:
        before = self.current_finger_position()
        if before is None:
            raise RuntimeError("Gripper joint feedback is unavailable.")
        self._set_gripper_pwm_mode()
        command = JointSingleCommand(name="gripper", cmd=float(pwm))
        stop = JointSingleCommand(name="gripper", cmd=0.0)
        deadline = time.monotonic() + duration
        try:
            while time.monotonic() < deadline:
                self._raise_if_cancelled()
                self._gripper_pwm.publish(command)
                time.sleep(0.0005)
        finally:
            self._gripper_pwm.publish(stop)
        time.sleep(GRIPPER_SETTLE_SECONDS)
        self._raise_if_cancelled()
        after = self.current_finger_position()
        if after is None:
            raise RuntimeError("Gripper joint feedback was lost after the PWM command.")
        movement = after - before
        if (
            action == "open"
            and after > OPEN_FINGER_MAX_M
            and movement > -FINGER_MOVEMENT_MIN_M
        ):
            raise RuntimeError(
                "Open PWM was sent, but no opening movement was measured: "
                f"{before:.4f} -> {after:.4f} m."
            )
        if (
            action == "close"
            and after < CLOSED_FINGER_MIN_M
            and movement < FINGER_MOVEMENT_MIN_M
        ):
            raise RuntimeError(
                "Close PWM was sent, but no closing movement was measured: "
                f"{before:.4f} -> {after:.4f} m."
            )
        self._gripper_state = "open" if action == "open" else "closed"
        if action == "open":
            self._holding_object = False
        self._set_result(
            f"Gripper {action} verified from left_finger feedback: "
            f"{before:.4f} -> {after:.4f} m."
        )

    def _raise_if_cancelled(self) -> None:
        if self._cancel_requested.is_set():
            raise RuntimeError("Robot workflow was cancelled by the operator.")

    def _wait_with_cancel(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._raise_if_cancelled()
            time.sleep(max(0.0, min(0.1, deadline - time.monotonic())))
        self._raise_if_cancelled()

    @staticmethod
    def _robot_state(joints: list[float]) -> RobotState:
        state = RobotState()
        state.is_diff = False
        state.joint_state.name = list(JOINT_ORDER)
        state.joint_state.position = [float(value) for value in joints]
        return state

    @staticmethod
    def _trajectory_endpoint(trajectory) -> list[float]:
        points = trajectory.joint_trajectory.points
        names = trajectory.joint_trajectory.joint_names
        if not points:
            raise RuntimeError("MoveIt returned an empty trajectory.")
        by_name = dict(zip(names, points[-1].positions))
        if not all(name in by_name for name in JOINT_ORDER):
            raise RuntimeError("MoveIt trajectory omitted an arm joint.")
        return [float(by_name[name]) for name in JOINT_ORDER]

    @staticmethod
    def _make_pose(x: float, y: float, z: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = BASE_FRAME
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose

    @staticmethod
    def _within_workspace(x: float, y: float, z: float) -> bool:
        return 0.15 <= x <= 0.65 and -0.40 <= y <= 0.40 and z >= -0.05

    @staticmethod
    def _joint_constraints(joints: list[float]) -> Constraints:
        constraints = Constraints()
        for joint_name, position in zip(JOINT_ORDER, joints):
            constraint = JointConstraint()
            constraint.joint_name = joint_name
            constraint.position = float(position)
            constraint.tolerance_above = 0.01745
            constraint.tolerance_below = 0.01745
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        return constraints

    @staticmethod
    def _await(future, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if future.done():
                return future.result()
            time.sleep(0.01)
        return None

    def _invalidate_plan(self) -> None:
        self._planned_name = None
        self._planned_trajectory = None
        self._plan_start_joints = None
        self._invalidate_pick_plan()

    def _invalidate_pick_plan(self) -> None:
        self._pick_trajectories = []
        self._pick_start_joints = None
        self._pick_pregrasp_joints = None
        self._pick_grasp_joints = None
        self._pick_target = None

    def _set_result(self, message: str) -> None:
        with self._lock:
            self._last_result = message
            self._result_sequence += 1
            self._result_history.append(
                {
                    "sequence": self._result_sequence,
                    "time": time.strftime("%H:%M:%S"),
                    "message": message,
                }
            )


def main() -> int:
    os.environ["MOVEIT_CAMERA_BROADCAST"] = "false"
    rclpy.init()
    node = RobotPoseManager()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    context = zmq.Context()
    command_socket = context.socket(zmq.REP)
    command_socket.setsockopt(zmq.LINGER, 0)
    command_socket.bind(COMMAND_ENDPOINT)
    status_socket = context.socket(zmq.PUB)
    status_socket.setsockopt(zmq.LINGER, 0)
    status_socket.bind(STATUS_ENDPOINT)
    poller = zmq.Poller()
    poller.register(command_socket, zmq.POLLIN)
    last_status_time = 0.0

    try:
        while rclpy.ok():
            events = dict(poller.poll(100))
            if command_socket in events:
                request = command_socket.recv_json()
                command = request.get("command")
                try:
                    if command == "save_pose":
                        response = node.save_pose(str(request.get("name", "")))
                    elif command == "plan_pose":
                        response = node.plan_pose(str(request.get("name", "")))
                    elif command == "execute_pose":
                        response = node.begin_execute(str(request.get("name", "")))
                    elif command == "plan_pick":
                        response = node.plan_pick(
                            request.get("target", {}),
                            str(request.get("target_label", "")),
                            str(request.get("profile", "")),
                        )
                    elif command == "execute_pick":
                        response = node.begin_pick(request.get("target", {}))
                    elif command == "release_object":
                        response = node.begin_gripper_command("open")
                    elif command == "open_gripper":
                        response = node.begin_gripper_command("open")
                    elif command == "close_gripper":
                        response = node.begin_gripper_command("close")
                    elif command == "cancel":
                        response = node.cancel()
                    elif command == "status":
                        response = {"success": True, **node.status()}
                    else:
                        response = {"success": False, "message": "Unknown command."}
                except Exception as error:
                    response = {"success": False, "message": str(error)}
                command_socket.send_json(response)

            now = time.monotonic()
            if now - last_status_time >= 0.5:
                status_socket.send_json(node.status())
                last_status_time = now
    except KeyboardInterrupt:
        pass
    finally:
        node.cancel()
        command_socket.close(linger=0)
        status_socket.close(linger=0)
        context.term()
        executor.shutdown()
        executor_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
