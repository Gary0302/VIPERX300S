#!/usr/bin/env python3
from pathlib import Path
import sys
import time


def _add_workspace_python_paths():
    """Allow direct execution without first sourcing the local ROS overlay."""
    workspace_root = Path(__file__).resolve().parents[1]
    candidate_paths = [
        workspace_root / 'ros2_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules',
        workspace_root / 'ros2_ws/src/interbotix_ros_toolboxes/interbotix_common_toolbox/interbotix_common_modules',
        workspace_root / 'ros2_ws/build/interbotix_xs_msgs/rosidl_generator_py',
    ]

    for path in candidate_paths:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


_add_workspace_python_paths()

import rclpy
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

OPEN_FINGER_LIMIT_M = 0.06
OPEN_PWM = 350.0
CLOSE_PWM = -700.0
OPEN_HOLD_SECONDS = 1.0
CLOSE_HOLD_SECONDS = 1.0
PWM_PUBLISH_HZ = 2000.0


def print_gripper_status(bot, label):
    finger_position = bot.gripper.get_finger_position()
    print(
        f"{label}: left_finger={finger_position:.4f} m, "
        f"estimated_opening={finger_position * 2.0:.4f} m, "
        f"limit=[{bot.gripper.left_finger_lower_limit:.4f}, "
        f"{bot.gripper.left_finger_upper_limit:.4f}] m"
    )


def hold_gripper_pwm(bot, pwm, seconds):
    """Continuously command raw gripper PWM for the full requested duration."""
    period = 1.0 / PWM_PUBLISH_HZ
    deadline = time.monotonic() + seconds

    while time.monotonic() < deadline and rclpy.ok():
        bot.core.robot_write_joint_command('gripper', pwm)
        time.sleep(period)

    bot.core.robot_write_joint_command('gripper', 0.0)


def main():
    # 初始化 rclpy
    rclpy.init()
    
    print("初始化手臂 API...")
    # 初始化 bot，這會在底層自動將夾爪配置為預設的 PWM 模式
    bot = InterbotixManipulatorXS(
        robot_model='vx300s', 
        group_name='arm', 
        gripper_name='gripper',
        gripper_pressure=0.45,
    )

    bot.core.robot_set_operating_modes('single', 'gripper', 'pwm')
    bot.gripper.left_finger_upper_limit = max(
        bot.gripper.left_finger_upper_limit,
        OPEN_FINGER_LIMIT_M,
    )
    
    print("使用 PWM 模式【順暢張開】夾爪...")
    print_gripper_status(bot, "開爪前")
    hold_gripper_pwm(bot, OPEN_PWM, OPEN_HOLD_SECONDS)
    print_gripper_status(bot, "開爪後")

    input("確認夾爪張開幅度後按 Enter 繼續閉合，或 Ctrl+C 停止...")

    
    print("使用 PWM 模式【順暢閉合/抓取】...")
    # close() 會使用預設的適當力道閉合。
    # 如果你想精準控制夾緊的力道（PWM 大小），可以使用：
    # bot.gripper.grasp(pressure=0.3, delay=2.0)
    # pressure 範圍是 0.0 ~ 1.0（對應最小到最大 PWM），值越小動作越溫和，不容易傷到物體
    hold_gripper_pwm(bot, CLOSE_PWM, CLOSE_HOLD_SECONDS)
    print_gripper_status(bot, "閉爪後")
    
    print("清理並關閉...")
    rclpy.shutdown()

if __name__ == '__main__':
    main()
