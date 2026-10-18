# ViperX-300S 校正與硬體問題排除指南

本文件只記錄目前蘋果辨識抓取流程使用的校正方式。完整啟動順序請看 [蘋果抓取操作指南](APPLE_PICK_GUIDE.md)。容器內的專案固定掛載在 `/workspace`，不依賴原作者的主機路徑。

## 1. 目前使用的手眼校正

[hand_eye.launch.py](hand_eye.launch.py) 發布 `vx300s/ee_gripper_link → camera_link`：

```text
x=-0.06228  y=0.01783  z=0.02963
qx=0.03568  qy=0.02170  qz=0.01704  qw=0.99898
```

這組數值已用目前的 ViperX-300S、夾爪與 RealSense 安裝方式完成實機抓取。只要相機或固定架的位置、角度發生變化，就必須重新校正。

## 2. 重新進行手眼校正

準備 Tag36h11、ID 0 的 AprilTag。`tags.yaml` 目前設定標籤邊長為 `0.16 m`，若使用其他尺寸，必須先修改 [tags.yaml](tags.yaml) 的 `size`。

1. 在啟動 Docker 前接妥相機、U2D2 USB 與手臂電源。從專案根目錄執行：

   ```bash
   ls -l /dev/ttyUSB0 /dev/video0
   cd docker
   ./run.sh
   ```

2. 每個新容器終端機都確認使用相同的 Domain ID：

   ```bash
   export ROS_DOMAIN_ID=42
   echo "$ROS_DOMAIN_ID"
   ```

3. 分別在三個終端機啟動相機、MoveIt 2 與 AprilTag。校正期間不要啟動 `hand_eye.launch.py`。

   ```bash
   bash /workspace/apple_pick/start_camera.sh
   ```

   ```bash
   ros2 launch interbotix_xsarm_moveit xsarm_moveit.launch.py robot_model:=vx300s hardware_type:=actual
   ```

   ```bash
   ros2 run apriltag_ros apriltag_node --ros-args \
     -r image_rect:=/camera/camera/color/image_raw \
     -r camera_info:=/camera/camera/color/camera_info \
     --params-file /workspace/apple_pick/tags.yaml
   ```

4. 確認能取得手臂與標籤 TF：

   ```bash
   ros2 run tf2_ros tf2_echo vx300s/base_link vx300s/ee_gripper_link
   ros2 run tf2_ros tf2_echo camera_link tag36h11:0
   ```

5. 在第四個終端機啟動校正程式：

   ```bash
   python3 /workspace/apple_pick/hand_eye_calibration.py
   ```

6. 用 MoveIt 2 將手臂移到至少 5～10 個不同姿態。每次等待手臂完全停止且相機看得到 AprilTag，再回到校正終端機按 `Enter` 記錄。樣本必須包含明顯的手腕旋轉、傾斜與側擺，不能只有平移。

7. 輸入 `c` 計算。程式會輸出 `static_transform_publisher` 指令，其中依序包含 `x y z qx qy qz qw`。

8. 將七個新數值填入 [hand_eye.launch.py](hand_eye.launch.py) 的 `arguments`，重新啟動該 launch 檔後再驗證目標位置。先用一顆熟蘋果低速測試，確認 TF 與實物位置一致後再進行完整抓放。

## 3. 手臂原點與馬達設定

自行拆裝關節後，若 RViz2 姿態與實體手臂不一致，可使用 Dynamixel Wizard 2.0 檢查：

1. 關閉所有馬達 Torque。
2. 將手臂手動擺到機構定義的零度姿態。
3. Address 20 的 Homing Offset 可依下式計算：

   ```text
   新 Offset = 舊 Offset + (2048 - 實體擺正時的 Present Position)
   ```

若補償量超出馬達允許範圍，應先檢查關節是否在組裝時相差半圈，不要用過大的 Offset 強行補償。Homing Offset 寫入馬達 EEPROM 後，不會被 ROS 2 設定檔覆蓋。

馬達位置極限及 Drive Mode 由下列檔案載入：

```text
ros2_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/
interbotix_xsarm_control/config/vx300s.yaml
```

修改 ROS 工作區內的 YAML 後，要在容器內重新建置：

```bash
cd /workspace/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## 4. 常見問題

### 找不到 `/dev/ttyUSB0`

在主機確認 U2D2 已接上，並關閉可能占用連接埠的 Dynamixel Wizard：

```bash
ls -l /dev/ttyUSB*
```

若設備是在容器建立後才插入，從主機的 `docker/` 目錄執行 `./stop.sh`，接妥設備後再執行 `./run.sh`，讓 Compose 重新綁定裝置。

### 相機沒有影像

先確認 RealSense 使用 USB 3.x 連線，再檢查彩色影像頻率：

```bash
rs-enumerate-devices
ros2 topic hz /camera/camera/color/image_raw
```

正常情況約為 30 Hz。若 topic 有頻率但 RViz2 沒畫面，在 Image 顯示中選擇 `/camera/camera/color/image_raw`，並將 Reliability Policy 設為 `Best Effort`。

### YOLO 一直等待深度圖

```bash
ros2 topic list | grep aligned_depth
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
```

若沒有對齊深度 topic，回到相機終端機檢查啟動錯誤。每個終端機也必須使用相同的 `ROS_DOMAIN_ID`。

### MoveIt 2 可以規劃但手臂不動

確認啟動參數為 `hardware_type:=actual`，並檢查控制器與關節狀態：

```bash
ros2 control list_controllers -c /vx300s/controller_manager
ros2 topic hz /vx300s/joint_states
```

## 5. 權限注意事項

目前 Compose 使用 `privileged: true` 並明確掛載 USB 與影像裝置。一般情況不需要把整個 `/dev/bus/usb` 或所有 `/dev/video*` 改成 `777`。若另一台主機仍出現權限錯誤，應依該主機的 udev 規則、`dialout`／`plugdev` 群組與實際裝置名稱調整，避免永久開放所有裝置權限。
