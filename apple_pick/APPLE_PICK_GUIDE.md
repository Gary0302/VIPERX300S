# ViperX-300S 蘋果辨識與自動抓放操作指南

本頁以專題生最後版本的 `apple_pick/yolo_detector6.py` ＋ `apple_pick/pick_and_place3.py` 為正式流程。相關任務程式集中在 `apple_pick/`，模型權重放在 `models/`。偵測器以 `target_ripe_apple_ID`、`target_green_apple_ID` 發布 TF；抓放程式只處理熟蘋果。整理時沒有更動夾爪 PWM、動作位置或校正數值；目前已完成單顆熟蘋果的辨識、夾取、放置及回到 Sleep 姿勢實測。

## 0. 主機端準備

在 Git 倉庫根目錄執行；無論倉庫放在哪個使用者資料夾，`docker/` 與 `ros2_ws/` 的相對位置都要維持。

```bash
# 先接上手臂與 RealSense，確認 Docker 所需的裝置存在
ls -l /dev/ttyUSB0 /dev/video0
xhost +local:docker
cd docker
./build.sh                 # 第一次使用或 Dockerfile 變更時
./run.sh                   # 啟動容器；詢問是否進入時直接按 Enter
```

若只測試 RealSense 相機，先在主機確認相機：

```bash
lsusb | grep -i realsense
ls -l /dev/video*
```

`lsusb` 用來確認 USB 裝置是否被系統辨識，`/dev/video*` 是相機建立的影像裝置。RealSense 通常會建立多個 `/dev/videoN`，不一定只有 `/dev/video0`。

目前 `docker/docker-compose.yml` 將 `/dev/ttyUSB0`（手臂的 U2D2）列為必要裝置。因此即使只測相機，啟動這份容器時仍須接上 U2D2 的 USB；手臂馬達電源可以保持關閉，也不要啟動 MoveIt 2。若 `/dev/ttyUSB0` 不存在，`./run.sh` 可能無法建立容器。

Docker Compose 會把這個倉庫掛載到容器的 `/workspace`。`/workspace` 是容器內的固定路徑，不是原作者電腦的路徑。

第一次使用或修改 `ros2_ws/src` 後，先從主機進入容器並建置工作區：

```bash
docker exec -it viperx300s_robot bash
source /opt/ros/humble/setup.bash
cd /workspace/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## 1. 啟動順序

每個新開的**主機終端機**先執行：

```bash
docker exec -it viperx300s_robot bash
export ROS_DOMAIN_ID=42
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
```

進入容器時若看到 `ROS2 Humble + GPU + Interbotix ViperX-300S 環境已就緒!`，這只是 Docker 映像寫在 `/root/.bashrc` 的提示文字，也表示 ROS 2 與現有工作區已自動載入；重複執行上面的兩個 `source` 不會有問題。

查看目前終端機的 Domain ID：

```bash
echo "$ROS_DOMAIN_ID"
# 或
printenv ROS_DOMAIN_ID
```

本專案操作時預期顯示 `42`。`ROS_DOMAIN_ID` 必須在所有需要互相通訊的終端機一致，而且每個新終端機都要確認；若不是 `42`，先執行 `export ROS_DOMAIN_ID=42`，再執行任何 `ros2` 指令。若工作區尚未建置完成，先回到第 0 步；不要改用從別台電腦複製來的 `install/`。

1. **相機**

   ```bash
   bash /workspace/apple_pick/start_camera.sh
   ```

   只測相機時，執行到這一步即可，不需要啟動後面的手臂、手眼校正、辨識或抓放程式。另開一個已進入容器並載入相同 ROS 環境的終端機，可用以下指令確認影像是否持續發布：

   ```bash
   ros2 topic hz /camera/camera/color/image_raw
   ```

2. **手臂與 MoveIt 2**

   ```bash
   ros2 launch interbotix_xsarm_moveit xsarm_moveit.launch.py robot_model:=vx300s hardware_type:=actual
   ```

3. **手眼校正 TF**

   ```bash
   ros2 launch /workspace/apple_pick/hand_eye.launch.py
   ```

4. **蘋果辨識**

   ```bash
   python3 /workspace/apple_pick/yolo_detector6.py
   ```

5. **視覺檢查（選用）**

   ```bash
   rviz2
   ```

6. **熟蘋果抓放**：先在 RViz 確認目標 TF 與實物位置相符，也可用 `ros2 run tf2_ros tf2_echo vx300s/base_link target_ripe_apple_1` 查看目標座標。確認後再執行；程式辨識到目標後會要求按 Enter 才開始動作。

   ```bash
   python3 /workspace/apple_pick/pick_and_place3.py
   ```

## Docker 停止方式

- `docker compose stop`：只停止容器，保留容器本身。下次可用 `docker compose start`，或直接再執行 `./run.sh`。
- `docker compose down`：停止並移除 Compose 建立的容器與網路。下次執行 `./run.sh` 時會重新建立容器，但不會重新建置 Docker image。
- `./stop.sh`：目前內部執行的是 `docker compose down`。

這兩種方式都不會刪除 Git 專案檔案。一般的 `docker compose down` 也不會刪除 Docker image；只有另外加上 `--rmi` 才會移除 image，加上 `-v` 才會移除 Compose 管理的 volume。

## 此套硬體的校正設定

目前 [hand_eye.launch.py](hand_eye.launch.py) 發布 `vx300s/ee_gripper_link → camera_link`：

```text
x=-0.06228  y=0.01783  z=0.02963
qx=0.03568  qy=0.02170  qz=0.01704  qw=0.99898
```

專題生測試成功後，相機與夾爪的安裝沒有改動，因此以這組程式內的數值為準。舊版校正數值只供 Git 紀錄追溯，不應與目前的 launch 檔混用。若日後安裝位置變動，依 [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md) 重新校正。

## 對應與注意事項

- `apple_pick/yolo_detector6.py` 預設使用 `models/yolo11s.pt`，並將 YOLO 推論限制為 COCO class 47（apple），不會輸出模型中的其他物件類別。偵測到的蘋果再依顏色分成 `ripe_apple`、`green_apple` 或 `unknown`；抓放程式只處理 `ripe_apple`。換模型時可在此終端機設定 `YOLO_MODEL=models/新檔名.pt`，自訂類別 ID 另設 `APPLE_CLASS_ID`。
- `apple_pick/pick_and_place3.py` 尋找 `target_ripe_apple_1` 等帶編號的 TF，依序抓取熟蘋果。它會實際控制手臂與夾爪。
- 抓放程式已停用 MoveIt 橋接器內建的相機 TF，避免與第 3 步的手眼校正 TF 同時發布 `camera_link`。
- 目標、預備點及夾取點都會檢查原有工作範圍；夾爪若未成功開啟，該目標不會進入夾取步驟。
- 偵測器目前每張影像都依偵測順序重新編號，沒有跨影格追蹤；多顆蘋果時 ID 可能互換。首次實機測試建議先放一顆熟蘋果，確認座標與動作後再測多顆。
- 相機、深度、TF 或工作區尚未就緒時，先檢查 `ros2 topic list` 與 `ros2 run tf2_ros tf2_echo vx300s/base_link vx300s/ee_gripper_link`。
- 同一時間只啟動**一個** YOLO 偵測器；公開版本以 `yolo_detector6.py` 為正式流程。
- Docker 的 USB 裝置與 NVIDIA 設定仍依主機硬體而定；專案路徑可攜不代表硬體接線自動通用。
