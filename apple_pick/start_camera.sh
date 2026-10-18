#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
exec ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true initial_reset:=true "$@"
