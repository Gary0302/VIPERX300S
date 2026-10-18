# ViperX-300S 蘋果辨識與抓放

本專案使用 ROS 2 Humble、RealSense、YOLO11s、MoveIt 2 與 Interbotix ViperX-300S。辨識夾取任務的程式與操作文件集中在 `apple_pick/`，模型權重獨立放在 `models/`。開始操作請看 [蘋果辨識與自動抓放操作指南](apple_pick/APPLE_PICK_GUIDE.md)。

## 資料夾

- `apple_pick/`：相機啟動、YOLO 辨識、熟蘋果抓放、手眼校正、AprilTag 設定及輔助程式。相機節點本身由 ROS 的 `realsense2_camera` 套件提供，這裡的 `start_camera.sh` 負責啟動它。
- `models/`：YOLO 權重；正式流程使用 `yolo11s.pt`。
- `ros2_ws/src/`：ROS 原始碼；建置產物由使用者自行產生。
- `docker/`：唯一一套 Docker 建置及啟動設定。Docker image 在使用者電腦上建置，不包含在 Git 倉庫中。

整理前的備份與實驗腳本保留在本機 `archive/`，並由 `.gitignore` 排除，不會混入可執行版本。需要追溯已提交過的原始版本時可查看 Git 紀錄。

## 從 GitHub 下載後

以下主機端指令從下載後的**專案根目錄**執行，無須使用原作者電腦的路徑。需要 Ubuntu 22.04、Docker Compose、NVIDIA GPU／Container Toolkit、X11，以及已接妥的 ViperX-300S 與 RealSense。Compose 會把目前專案掛載成容器內固定的 `/workspace`。

```bash
git clone https://github.com/nthu114033639-lgtm/VIPERX300S-rebuild.git
cd VIPERX300S-rebuild
cd docker
./build.sh
./run.sh    # 詢問是否進入容器時直接按 Enter
```

首次使用或 ROS 原始碼變更後，從主機進入容器並建置工作區：

```bash
docker exec -it viperx300s_robot bash
source /opt/ros/humble/setup.bash
cd /workspace/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

之後按 [蘋果辨識與自動抓放操作指南](apple_pick/APPLE_PICK_GUIDE.md) 的順序啟動。`apple_pick/yolo_detector6.py` 預設載入 `models/yolo11s.pt`；從其他工作目錄呼叫也能找到權重。

若換成新的蘋果辨識模型，把 `.pt` 檔放進 `models/`，在辨識終端機指定路徑即可，無須修改程式：

```bash
YOLO_MODEL=models/new_apple.pt APPLE_CLASS_ID=0 python3 /workspace/apple_pick/yolo_detector6.py
```

`YOLO_MODEL` 的相對路徑以專案根目錄為準，也可使用容器內的絕對路徑。`APPLE_CLASS_ID` 預設為 COCO 蘋果類別 `47`；自訂模型若把蘋果放在其他類別 ID，需一起指定。程式仍會用影像顏色區分熟／青蘋果，新模型的類別與輸出也必須和這個流程相容。

## 硬體設定

`docker/docker-compose.yml` 目前指定 `/dev/ttyUSB0`、`/dev/video0` 至 `/dev/video11`、NVIDIA runtime 與 X11；其他電腦若裝置名稱或圖形環境不同，需要依實際主機調整。`apple_pick/hand_eye.launch.py` 是此套硬體的校正值，換相機、夾爪安裝位置或機械臂時必須依 [校正指南](apple_pick/CALIBRATION_GUIDE.md) 重新校正。首次實機執行前，先確認 TF、目標座標與夾爪活動空間。
