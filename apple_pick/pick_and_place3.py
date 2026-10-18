#!/usr/bin/env python3
from copy import deepcopy
from pathlib import Path
import os
import sys
import time
import math

import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger
from threading import Thread
from tf2_ros import Buffer, TransformListener

from geometry_msgs.msg import PoseStamped


def _add_workspace_python_paths():
    """Allow direct execution without first sourcing the local ROS overlay."""
    workspace_root = Path(__file__).resolve().parents[1]
    candidate_paths = [
        workspace_root / "ros2_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules",
        workspace_root / "ros2_ws/src/interbotix_ros_toolboxes/interbotix_common_toolbox/interbotix_common_modules",
        workspace_root / "ros2_ws/build/interbotix_xs_msgs/rosidl_generator_py",
    ]

    for path in candidate_paths:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


_add_workspace_python_paths()

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

OPEN_FINGER_LIMIT_M = 0.06
OPEN_PWM = 350.0
CLOSE_PWM = -700.0
OPEN_HOLD_SECONDS = 1.25
CLOSE_HOLD_SECONDS = 1.3
PWM_PUBLISH_HZ = 2000.0
ARM_SETTLE_TIME = 2.5
GRIPPER_WAIT_TIME = 0.5


def hold_gripper_pwm(bot, pwm, seconds):
    """Continuously command raw gripper PWM for the full requested duration."""
    period = 1.0 / PWM_PUBLISH_HZ
    deadline = time.monotonic() + seconds

    try:
        while time.monotonic() < deadline and rclpy.ok():
            bot.core.robot_write_joint_command("gripper", pwm)
            time.sleep(period)
    finally:
        bot.core.robot_write_joint_command("gripper", 0.0)


def within_workspace(x, y, z):
    """Use the original target limits for every commanded Cartesian pose."""
    return 0.15 <= x <= 0.65 and -0.40 <= y <= 0.40 and z >= -0.05


def build_pose(x, y, z, orientation_mode="neutral") -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = "vx300s/base_link"
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = float(z)

    if orientation_mode == "tilted":
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.707
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.707
    else:
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
    return pose


def discover_detected_target_ids(tf_buffer, target_prefix: str, max_ids: int = 20):
    detected_ids = []
    for candidate_id in range(1, max_ids + 1):
        frame_name = f"target_{target_prefix}_{candidate_id}"
        try:
            tf_buffer.lookup_transform("vx300s/base_link", frame_name, rclpy.time.Time())
            detected_ids.append(candidate_id)
        except Exception:
            continue
    return detected_ids


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
VLSERVO_SRC = WORKSPACE_ROOT / "ros2_ws" / "src" / "vlservo"
if VLSERVO_SRC.is_dir() and str(VLSERVO_SRC) not in sys.path:
    sys.path.insert(0, str(VLSERVO_SRC))

from VLServo.moveit_marker_bridge import MoveItMarkerBridge

class MoveItPlanner:
    def __init__(self):
        self._logger = get_logger("moveit_planner")
        self.moveit_bridge = MoveItMarkerBridge(
            robot_name="vx300s",
            planning_group="interbotix_arm",
            eef_link="vx300s/ee_gripper_link",
            joint_order=["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"],
            target_frame="vx300s/base_link",
            camera_frame="camera_link",
            velocity_scaling=0.2,
            acceleration_scaling=0.2,
        )
        self._logger.info("✅ MoveIt bridge 已啟動！")

    def plan_and_execute(self, pose_goal: PoseStamped, velocity_scaling=0.1, acceleration_scaling=0.1):
        self.moveit_bridge.velocity_scaling = float(velocity_scaling)
        self.moveit_bridge.acceleration_scaling = float(acceleration_scaling)

        ik_response = self.moveit_bridge._compute_ik(pose_goal)
        if ik_response is None:
            return False
        if ik_response.error_code.val != 1:
            return False

        joint_positions = self.moveit_bridge._build_joint_vector(ik_response.solution.joint_state)
        if joint_positions is None:
            return False

        self.moveit_bridge._last_joint_positions = list(joint_positions)
        self.moveit_bridge._last_pose_target = pose_goal

        result = self.moveit_bridge._send_move_group_goal(joint_positions, plan_only=False)
        return result.success

    def go_to_joint_positions(self, joints, label="目標位置"):
        result = self.moveit_bridge._send_move_group_goal(joints, plan_only=False)
        return result.success

    def go_to_home_pose(self):
        return self.go_to_joint_positions([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "Home 姿勢")

    def go_to_sleep_pose(self):
        sleep_joints = [0.0046, -1.6153, 1.5509, -0.0015, 0.2654, -0.0015]
        return self.go_to_joint_positions(sleep_joints, "Sleep 姿勢")

    def go_to_place_pose(self):
        place_joints = [1.5140390396118164, -0.015339808538556099, 1.1842331886291504, -0.0015339808305725455, -1.1274758577346802, 0.015339808538556099]
        return self.go_to_joint_positions(place_joints, "Place 姿勢")

    def return_to_safe_pose(self):
        self._logger.warning("⚠️ 啟動安全回收流程：先回 Home，再回 Sleep")
        if not self.go_to_home_pose():
            self._logger.error("❌ 回 Home 姿勢失敗！")
        time.sleep(0.5)
        if not self.go_to_sleep_pose():
            self._logger.error("❌ 回 Sleep 姿勢失敗！")
            return False
        return True

    def destroy(self):
        if hasattr(self.moveit_bridge, 'destroy'):
            self.moveit_bridge.destroy()

def main(args=None):
    # hand_eye.launch.py is the sole publisher of camera_link for this task.
    # MoveItMarkerBridge otherwise publishes a different default static TF.
    os.environ["MOVEIT_CAMERA_BROADCAST"] = "false"
    moveit_planner = MoveItPlanner()
    node = Node("pick_and_place_controller")
    logger = node.get_logger()

    print("初始化手臂 API (配置實體夾爪)...")
    bot = InterbotixManipulatorXS(
        robot_model="vx300s",
        group_name="arm",
        gripper_name="gripper",
        gripper_pressure=0.45,
    )
    bot.core.robot_set_operating_modes("single", "gripper", "pwm")
    bot.gripper.left_finger_upper_limit = max(
        bot.gripper.left_finger_upper_limit,
        OPEN_FINGER_LIMIT_M,
    )

    executor_thread = Thread(target=rclpy.spin, args=[node], daemon=True)
    executor_thread.start()

    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node, spin_thread=False)

    # ==========================================
    # 步驟 1：分別搜尋熟蘋果與青蘋果
    # ==========================================
    logger.info("▶️ 開始步驟 1: 搜尋場景中的蘋果 TF...")
    
    ripe_ids = []
    green_ids = []
    
    while rclpy.ok():
        ripe_ids = discover_detected_target_ids(tf_buffer, "ripe_apple")
        green_ids = discover_detected_target_ids(tf_buffer, "green_apple")
        
        # 若至少搜尋到其中一種目標，則跳出搜尋迴圈
        if ripe_ids or green_ids:
            break
        logger.info("⏳ 正在搜尋熟蘋果 (ripe_apple) 與青蘋果 (green_apple) 的 TF...", throttle_duration_sec=2.0)
        time.sleep(0.5)
        
    logger.info("✅ 結束步驟 1: 成功獲取目標 TF 狀態。")

    # ==========================================
    # 步驟 2：分別告知數量，並進行人工確認
    # ==========================================
    logger.info("▶️ 開始步驟 2: 輸出辨識數量...")
    print(f"\n========================================")
    print(f"📋 辨識結果統計：")
    print(f"🍎 熟蘋果數量: {len(ripe_ids)} 個 (ID: {ripe_ids})")
    print(f"🍏 青蘋果數量: {len(green_ids)} 個 (ID: {green_ids})")
    print(f"----------------------------------------")
    print(f"🎯 提示：系統將僅針對有編號的 {len(ripe_ids)} 個【熟蘋果】執行夾取，自動忽略青蘋果。")
    print(f"========================================")

    if not ripe_ids:
        logger.warning("⚠️ 未偵測到任何熟蘋果（可夾取目標），流程關閉。")
        moveit_planner.destroy()
        node.destroy_node()
        rclpy.shutdown()
        executor_thread.join(timeout=1.0)
        return

    input("👉 請按 [Enter] 鍵繼續熟蘋果自動抓取流程...")
    logger.info("✅ 結束步驟 2: 使用者已確認。")

    # ==========================================
    # 步驟 3：僅紀錄熟蘋果 (ripe_apple) 的位置
    # ==========================================
    logger.info("▶️ 開始步驟 3: 紀錄所有【熟蘋果】位置...")
    targets_data = []
    for tid in ripe_ids:
        target_frame_name = f'target_ripe_apple_{tid}'
        try:
            trans = tf_buffer.lookup_transform('vx300s/base_link', target_frame_name, rclpy.time.Time())
            targets_data.append({
                'id': tid,
                'x': trans.transform.translation.x,
                'y': trans.transform.translation.y,
                'z': trans.transform.translation.z
            })
        except Exception as e:
            logger.error(f"❌ 錯誤: 步驟 3 (紀錄熟蘋果 ID {tid} 的位置) 失敗: {e}")
    
    # 強制將熟蘋果依照 'id' 由小到大排序
    targets_data.sort(key=lambda item: item['id'])
    
    logger.info(f"✅ 結束步驟 3: 成功紀錄了 {len(targets_data)} 個熟蘋果的座標，已按 ID 排序完成。")

    # ==========================================
    # 依序號對熟蘋果重複執行夾取
    # ==========================================
    for idx, tgt in enumerate(targets_data):
        logger.info(f"\n🚀 正在處理第 {idx+1}/{len(targets_data)} 個熟蘋果 (Ripe ID: {tgt['id']})")
        
        x, y, z = tgt['x'], tgt['y'], tgt['z']

        x_offset = 0.0   # + up
        y_offset = -0.00 # - right
        z_offset = 0.00  # + front
        
        x += x_offset
        y += y_offset
        z += z_offset
        
        # 安全防護攔截
        if not within_workspace(x, y, z):
            logger.error(f"🛑 錯誤: 熟蘋果 ID {tgt['id']} 座標超出工作範圍！跳過此目標。")
            continue

        # 計算姿態與位移
        yaw = math.atan2(y, x)
        distance = math.sqrt(x**2 + y**2)
        z_lift = 0.05
        
        # 準備姿態的四元數
        qx, qy = 0.0, 0.0
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        # 到目標前方 2 公分
        back_off = 0.00
        pre_x = x - back_off * (x / distance)
        pre_y = y - back_off * (y / distance)
        pre_z = z + z_lift

        pre_grasp_pose = PoseStamped()
        pre_grasp_pose.header.frame_id = "vx300s/base_link"
        pre_grasp_pose.pose.position.x = pre_x
        pre_grasp_pose.pose.position.y = pre_y
        pre_grasp_pose.pose.position.z = pre_z
        pre_grasp_pose.pose.orientation.x = qx
        pre_grasp_pose.pose.orientation.y = qy
        pre_grasp_pose.pose.orientation.z = qz
        pre_grasp_pose.pose.orientation.w = qw

        # 水平向前移動
        forward_dist = 0.03
        grasp_x = pre_x + forward_dist * (x / distance)
        grasp_y = pre_y + forward_dist * (y / distance)

        if not within_workspace(pre_x, pre_y, pre_z) or not within_workspace(grasp_x, grasp_y, pre_z):
            logger.error(f"🛑 熟蘋果 ID {tgt['id']} 的預備點或夾取點超出工作範圍，跳過此目標。")
            continue
        
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = "vx300s/base_link"
        grasp_pose.pose.position.x = grasp_x
        grasp_pose.pose.position.y = grasp_y
        grasp_pose.pose.position.z = pre_z
        grasp_pose.pose.orientation = pre_grasp_pose.pose.orientation

        # ==========================================
        # 步驟 4 & 5：同時開啟夾爪並移動到預備位置
        # ==========================================
        logger.info("▶️ 開始步驟 4 & 5: 同時開啟夾爪並移動手臂...")

        open_gripper_errors = []

        def open_gripper_task():
            try:
                hold_gripper_pwm(bot, OPEN_PWM, OPEN_HOLD_SECONDS)
                time.sleep(GRIPPER_WAIT_TIME)
                logger.info("✅ 夾爪開啟完畢 (步驟 4 完成)。")
            except Exception as e:
                open_gripper_errors.append(e)
                logger.error(f"❌ 錯誤: 開啟夾爪失敗: {e}")

        gripper_thread = Thread(target=open_gripper_task)
        gripper_thread.start()

        if not moveit_planner.plan_and_execute(pre_grasp_pose, velocity_scaling=0.2):
            logger.error(f"❌ 錯誤: 手臂移動到預備位置失敗！")
            gripper_thread.join()
            moveit_planner.return_to_safe_pose()
            continue
        
        time.sleep(ARM_SETTLE_TIME)
        logger.info("✅ 已到達目標預備位置 (步驟 5 完成)。")

        gripper_thread.join()
        if open_gripper_errors:
            logger.error(f"❌ 夾爪未成功開啟，停止處理熟蘋果 ID {tgt['id']}。")
            moveit_planner.return_to_safe_pose()
            continue
        logger.info("✅ 結束步驟 4 & 5: 夾爪開啟與手臂移動已同步完成！")

        # ==========================================
        # 步驟 6：向前推移夾取
        # ==========================================
        logger.info("▶️ 開始步驟 6: 水平向前推移...")
        if not moveit_planner.plan_and_execute(grasp_pose, velocity_scaling=0.1):
            logger.error(f"❌ 錯誤: 步驟 6 (熟蘋果 ID {tgt['id']} 向前移動) 執行失敗！")
            moveit_planner.return_to_safe_pose()
            continue
        time.sleep(ARM_SETTLE_TIME)
        logger.info("✅ 結束步驟 6: 已到達夾取點。")

        # ==========================================
        # 步驟 7：閉合夾爪
        # ==========================================
        logger.info("▶️ 開始步驟 7: 閉合夾爪...")
        try:
            hold_gripper_pwm(bot, CLOSE_PWM, CLOSE_HOLD_SECONDS)
            time.sleep(GRIPPER_WAIT_TIME)
            logger.info("✅ 結束步驟 7: 夾爪閉合完畢 (目標抓取)。")
        except Exception as e:
            logger.error(f"❌ 錯誤: 步驟 7 (閉合夾爪) 失敗: {e}")
            moveit_planner.return_to_safe_pose()
            continue

        # ==========================================
        # 步驟 8：回位並移動到放置點
        # ==========================================
        logger.info("▶️ 開始步驟 8: 回位並移動到放置位置...")
        if not moveit_planner.go_to_sleep_pose():
            logger.error(f"❌ 錯誤: 步驟 8 (回位 Sleep) 執行失敗！")
            moveit_planner.return_to_safe_pose()
            continue
        time.sleep(3.0)
        
        if not moveit_planner.go_to_place_pose():
            logger.error(f"❌ 錯誤: 步驟 8 (移動到 Place 點) 執行失敗！")
            moveit_planner.return_to_safe_pose()
            continue
        time.sleep(ARM_SETTLE_TIME)
        logger.info("✅ 結束步驟 8: 已抵達放置位置。")

        # ==========================================
        # 步驟 9：開爪釋放
        # ==========================================
        logger.info("▶️ 開始步驟 9: 開爪釋放目標...")
        try:
            hold_gripper_pwm(bot, OPEN_PWM, OPEN_HOLD_SECONDS)
            time.sleep(1.0)
            logger.info("✅ 結束步驟 9: 熟蘋果已釋放。")
        except Exception as e:
            logger.error(f"❌ 錯誤: 步驟 9 (開爪釋放) 失敗: {e}")
            moveit_planner.return_to_safe_pose()
            continue

        # ==========================================
        # 步驟 10：關爪
        # ==========================================
        logger.info("▶️ 開始步驟 10: 關閉夾爪...")
        try:
            hold_gripper_pwm(bot, CLOSE_PWM, CLOSE_HOLD_SECONDS)
            time.sleep(GRIPPER_WAIT_TIME)
            logger.info("✅ 結束步驟 10: 夾爪已關閉。")
        except Exception as e:
            logger.error(f"❌ 錯誤: 步驟 10 (關閉夾爪) 失敗: {e}")
            moveit_planner.return_to_safe_pose()
            continue

        # ==========================================
        # 步驟 11：回位準備下一次作業
        # ==========================================
        logger.info("▶️ 開始步驟 11: 最後回位 (Sleep)...")
        if not moveit_planner.go_to_sleep_pose():
            logger.error(f"❌ 錯誤: 步驟 11 (最終回位) 執行失敗！")
            moveit_planner.return_to_safe_pose()
            continue
        logger.info("✅ 結束步驟 11: 已回到 Sleep 姿勢。")
        time.sleep(3.0)
        
        logger.info(f"🎉 成功完成第 {idx+1} 個熟蘋果 (Ripe ID: {tgt['id']}) 的抓放作業！\n")

    # 🏁 安全清理並退出
    logger.info("👋 所有熟蘋果均已處理完畢，正在關閉系統...")
    
    if 'tf_listener' in locals():
        del tf_listener
        
    moveit_planner.destroy()
    node.destroy_node()
    
    rclpy.shutdown()
    executor_thread.join(timeout=1.0)
    print("🏁 任務完成，程式已安全結束。")


if __name__ == "__main__":
    from rclpy.executors import ExternalShutdownException
    
    try:
        main()
    except ExternalShutdownException:
        pass
    except KeyboardInterrupt:
        print("\n🛑 使用者強制中斷程式。")
