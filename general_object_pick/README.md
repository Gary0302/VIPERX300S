# ViperX-300S 通用物件辨識與夾取

這個資料夾是通用物件夾取系統。介面整合 RealSense、YOLO、深度座標、手眼 TF、MoveIt 2、觀察姿勢與夾爪控制。目前已完成蘋果的實機直進夾取與返回測試；瓶子、鑷子及其他物品仍需建立各自的夾取設定。

既有且獨立的蘋果流程保留在 `apple_pick/`，不會被本資料夾取代。系統架構、完成進度與後續規劃請看 [DESIGN.md](DESIGN.md)。

## 目前已完成功能

- [x] 自動掃描 `models/` 中的 `.pt`、`.pth`、`.onnx` 與 `.engine` 模型。
- [x] 從 UI 選擇 YOLO 模型並輸入目標類別。
- [x] 啟動 RealSense，或沿用已存在的相機節點。
- [x] 顯示辨識框、類別、信心分數、數量與 FPS。
- [x] 使用對齊深度與相機內參計算目標三維座標。
- [x] 使用手眼校正 TF 將座標轉換至 `vx300s/base_link`。
- [x] 顯示 Camera、Detector、TF、MoveIt 與 Gripper 狀態。
- [x] 儲存、規劃、預覽與執行觀察姿勢。
- [x] 手動張開與閉合夾爪，並檢查實機馬達位置回報。
- [x] 蘋果目標座標穩定檢查、MoveIt 預覽及二次確認。
- [x] 蘋果直進 3 公分夾取、退回並返回執行前姿勢。
- [x] 保留單顆模式，並在 UI 加入多顆蘋果逐輪重新辨識、夾取、返回與釋放模式。
- [x] 動作期間顯示逐步紀錄，發生錯誤時中止後續步驟。

目前只有 `apple (Existing reference)` 可以執行實體夾取。其他模型與類別可以辨識及預覽座標，但尚未校正前不會開放夾取。

`Batch Pick and Release` 已完成軟體流程，但尚未完成多顆蘋果的實機驗證。第一次測試請將 `Maximum batch picks` 設為 `1`，確認返回位置下方有安全容器，再逐步增加數量。

## 資料夾內容

```text
general_object_pick/
├── README.md                    # 本操作說明
├── DESIGN.md                    # 架構、進度與後續規劃
├── ui_prototype.py              # PyQt5 操作介面與流程管理
├── generic_yolo_detector.py     # YOLO、深度與 TF 座標節點
├── robot_pose_manager.py        # MoveIt、觀察姿勢與夾爪控制
└── config/
    └── observation_poses.json   # 已儲存的觀察姿勢
```

`__pycache__/` 是 Python 執行時自動產生的位元碼快取，不是備份，也不是必要原始碼。專案的 `.gitignore` 已忽略它；刪除後下次執行 Python 仍可能自動建立。

## 第一次下載與建置

主機環境為 Ubuntu 22.04，並需要 Docker Compose、NVIDIA Container Toolkit、X11、RealSense、U2D2 與 ViperX-300S。

```bash
git clone https://github.com/jerry12221/VIPERX300S-rebuild.git
cd VIPERX300S-rebuild/docker
./build.sh
./run.sh
```

第一次使用、ROS 原始碼變更或 `ros2_ws/install/` 不存在時，在容器內建置：

```bash
source /opt/ros/humble/setup.bash
cd /workspace/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

`docker/docker-compose.yml` 目前需要 `/dev/ttyUSB0`。如果曾執行 `docker compose down`，重新建立容器前先接上 U2D2 USB。

## 每次啟動

### 1. 啟動容器

在專案根目錄執行：

```bash
cd docker
./run.sh
```

之後若要從新終端機進入既有容器：

```bash
docker exec -it viperx300s_robot bash
```

每個容器終端機都先執行：

```bash
export ROS_DOMAIN_ID=42
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
echo "$ROS_DOMAIN_ID"
```

所有終端機的 Domain ID 都必須顯示 `42`。

### 2. 終端機 1：MoveIt 2 與 RViz

```bash
ros2 launch interbotix_xsarm_moveit xsarm_moveit.launch.py \
  robot_model:=vx300s hardware_type:=actual
```

RViz 用來調整觀察姿勢及檢查 MoveIt 規劃路徑。拖曳互動標記時也可能留下半透明的目標模型，判斷 UI 規劃結果時要同時查看 Event Log。

### 3. 終端機 2：手眼校正 TF

```bash
ros2 launch /workspace/apple_pick/hand_eye.launch.py
```

這會發布目前實機已驗證的轉換：

```text
vx300s/ee_gripper_link -> camera_link
```

如果相機固定架、相機角度或夾爪結構改變，必須重新進行手眼校正。

### 4. 終端機 3：啟動 UI

```bash
python3 /workspace/general_object_pick/ui_prototype.py
```

UI 會自行管理相機與辨識器，不需要另外啟動 RealSense 或 `apple_pick/yolo_detector6.py`。

## UI 操作流程

### 視覺與座標

1. `Backend` 選擇 `YOLO (Phase 1)`。
2. `Model` 選擇需要的模型，例如 `yolo11s.pt`。
3. `Target object` 預設為 `apple`，也可從下拉選單選擇 `bottle`、`tweezers`，或輸入模型支援的其他類別。
4. 按 `Start Vision`。
5. 等待 Camera 與 Detector 顯示 `Ready`。
6. 若手眼 TF 完整，TF 會顯示 `Ready`，座標系應為 `vx300s/base_link`。
7. `Preview Target` 可以把目前座標記錄到 Event Log。

一般 COCO 模型不一定包含 `tweezers`。輸入的類別名稱必須存在於所選模型內。

### 連接手臂

1. 確認 MoveIt 與手眼 TF 已在另外兩個終端機執行。
2. 按 `Start Robot`。
3. 等待 MoveIt 與 Gripper 顯示 `Ready`。

`Start Robot` 只會連接既有 MoveIt，不會啟動第二套 MoveIt，也不會立刻移動手臂。

### 儲存與返回觀察姿勢

1. 在 RViz 將手臂調整到適合觀察的位置。
2. 在 UI 輸入 `Pose name`，只使用英文字母、數字、底線或減號。
3. 按 `Save Current`，將六軸角度存入 `config/observation_poses.json`。
4. 選擇已儲存姿勢並按 `Plan Only`。
5. 在 RViz 檢查路徑。
6. `Execute Plan` 解鎖後，再次確認才執行。

若規劃後手臂移動超過約 `0.05 rad`，舊規劃會失效，必須重新規劃。規劃速度與加速度比例目前都是 `0.10`。

### 手動測試夾爪

先按 `Start Robot`，再使用：

- `Open Gripper`：在目前位置張開夾爪。
- `Close Gripper`：在目前位置閉合夾爪。
- `Last command`：顯示最後完成的命令與 `left_finger` 回報值。

夾爪沿用實機成功參數：

- 張開：PWM `+350`，持續 `1.0 s`，然後送出 `0`。
- 閉合：PWM `-500`，持續 `1.0 s`，然後送出 `0`。

參數定義在 `robot_pose_manager.py`：

- `OPEN_PWM`、`CLOSE_PWM` 控制 PWM 輸出強度。絕對值越大，輸出通常越強、速度也越快；絕對值越小則較弱、較慢。這台實機以正值張開、負值閉合，調整強度時不要改變正負方向。
- `OPEN_HOLD_SECONDS`、`CLOSE_HOLD_SECONDS` 控制 PWM 持續時間。若要調整張開或閉合的移動量，優先調整對應時間；時間越短通常移動越少，時間越長通常移動越多。

這是依 PWM 與時間控制的開迴路方式，無法指定精確的兩指開口。調整時一次只改一個參數並小幅測試，例如以 `0.1 s` 調整持續時間，避免夾爪撞擊端點或施力過大。

第一次操作時，UI 會停用 MoveIt 的位置型 `gripper_controller`，讓 UI 獨占 PWM 控制。這台實機測得張開時 `left_finger` 約由 `0.0346 m` 降至 `0.0165 m`，閉合時約由 `0.0165 m` 升至 `0.0346 m`。程式會檢查移動方向與變化量。

`left_finger` 是馬達編碼器換算出的回報值，不是兩指間距，也不是受力感測器。目前只能確認夾爪有移動，不能確認物品是否真的被夾住。

自動夾取時，如果物品阻擋手指，使關閉位置回報未通過驗證，UI 會顯示警告，但手臂仍會執行安全退回並返回原始起點。操作員仍應把物品視為可能已被夾住，並在安全位置手動張開夾爪。

### 執行蘋果直進夾取

1. `Target object` 選擇或輸入 `apple`。
2. `Grasp profile` 選擇 `apple (Existing reference)`。
3. 等待 Camera、Detector、TF、MoveIt 與 Gripper 都顯示 `Ready`。
4. 畫面只放一顆蘋果，等待至少 10 筆座標穩定，三軸變動需小於 1.5 公分。
5. 按 `Plan Straight Pick`。這一步只規劃，不會移動硬體。
6. 在 RViz 檢查目前位置到預備點、前進、退回及返回起點四段路徑。
7. `Confirm and Pick` 解鎖後，閱讀確認視窗並選擇 `Yes`。
8. 自動流程依序執行：
   - 張開夾爪。
   - 移動到目標上方 5 公分的預備點。
   - 停留 2.5 秒。
   - 朝物品方向水平前進 3 公分。
   - 停留 2.5 秒。
   - 閉合夾爪。
   - 退回預備點。
   - 返回按下夾取前的原始姿勢。
9. 返回後夾爪保持閉合。確認位置適合放置後，按 `Open Gripper` 放開物品。

自動流程開始後不要再手動按夾爪按鈕。目標超出 X `0.15–0.65 m`、Y `-0.40–0.40 m` 或 Z 低於 `-0.05 m` 時，系統會拒絕規劃。

### 執行多顆蘋果批次夾取與釋放

批次模式不會取代上面的單顆模式。`Single Pick (Keep Holding)` 保留原本抓取後保持夾爪閉合的流程；`Batch Pick and Release` 才會在每輪返回後自動張開夾爪。

1. 先使用 `Observation Pose` 將手臂移到相機適合觀察、且夾爪下方有安全容器的位置。批次流程會把每輪開始規劃時的關節位置當作返回與釋放位置。
2. `Target object` 選擇 `apple`，`Grasp profile` 選擇 `apple (Existing reference)`。
3. `Pick mode` 選擇 `Batch Pick and Release`。
4. 第一次實機測試把 `Maximum batch picks` 設為 `1`；確認後再設定為 `2` 或 `3`。
5. 等待 Camera、Detector、TF、MoveIt 與 Gripper 都顯示 `Ready`，再按 `Confirm and Start Batch`。
6. 確認視窗會再次提醒：返回位置下方必須有容器，而且釋放後的蘋果不能留在可夾取區域。

每一輪會：

- 取得畫面內所有有效蘋果座標。
- 先鎖定信心最高的候選，再以最近位置連續追蹤 10 筆；三軸變動需小於 1.5 公分。
- 規劃並執行張爪、pregrasp、前進 3 公分、關爪、retreat 及返回起點。
- 返回起點後停留 1 秒，再自動張開夾爪。釋放後維持張開，讓下一輪可以直接進入夾取準備；若批次只設定一顆，任務結束後也會保持張開。
- 丟棄上一輪座標，等待 1.5 秒並重新累積全新的辨識資料，再選下一顆。
- 達到 `Maximum batch picks`，或完成至少一顆後連續 5 秒找不到剩餘目標時結束。

目前使用三維位置的最近鄰方式維持單輪目標，不是永久物件 ID。已完成目標原始位置周圍 4 公分會從同一批次的候選中排除；若蘋果彼此重疊、被移動，或釋放後重新落入辨識區，應立即取消批次並改用單顆模式。

## 確認 TF 是否正常

UI 顯示下列內容時，可以使用基座座標規劃：

```text
TF - Ready
Coordinate frame: vx300s/base_link
```

若顯示 `camera_color_optical_frame`，代表相機與深度正常，但完整手臂 TF 尚未建立，不能直接用該座標控制手臂。

終端機可分別檢查：

```bash
ros2 run tf2_ros tf2_echo \
  vx300s/base_link vx300s/ee_gripper_link

ros2 run tf2_ros tf2_echo \
  vx300s/base_link camera_color_optical_frame
```

持續出現 Translation 與 Rotation 代表轉換鏈正常：

```text
vx300s/base_link
  -> vx300s/ee_gripper_link
  -> camera_link
  -> camera_color_optical_frame
```

## 常用檢查指令

```bash
ros2 node list
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
ros2 topic hz /vx300s/joint_states
ros2 topic echo /object_detector/status
```

正常測試時，相機約為 30 Hz，手臂關節狀態約為 100 Hz。

## 常見問題

### MoveIt 回報 `CONTROL_FAILED (-4)`

MoveIt 執行前會比較目前關節角度與路徑起點。UI 在每一段實體動作前，會依最新 `/vx300s/joint_states` 重新規劃並執行，以降低起點落差。如果仍然失敗：

1. 不要再次按 `Confirm and Pick`。
2. 查看 Event Log 的錯誤碼與最大關節誤差。
3. 重新規劃返回安全觀察姿勢。
4. 在 RViz 確認路徑後才執行。
5. 若控制器沒有回應，使用實體急停或關閉馬達電源。

### 夾爪回報方向

這台實機的 `left_finger` 在張開時數值下降，閉合時數值上升，與 MoveIt 預設姿勢的數值慣例相反。PWM 的方向沒有寫反；程式已依實機回報方向驗證。如果更換手臂、夾爪或控制設定，需重新測量開關方向與端點。

## 正常停止

1. 若夾爪持有物品，先移到安全位置，再按 `Open Gripper`。
2. 按 `Stop Vision / Cancel Motion`。
3. 關閉 UI。
4. 在手眼 TF 與 MoveIt 終端機分別按 `Ctrl+C`。
5. 輸入 `exit` 離開容器。
6. 主機執行 `docker compose stop` 可保留容器；`docker compose down` 會移除容器與 Compose 網路。

UI 的停止與取消功能不能取代實體急停。若手臂未停止，立即使用實體急停或關閉馬達電源。

## 目前限制

- MoveIt、RViz 與手眼 TF 仍需在外部終端機啟動。
- 實體夾取目前只開放蘋果設定。
- 多個同類物品出現時，目前選擇信心最高者，還沒有固定追蹤 ID。
- 沒有物品輪廓、方向或重心估計，細長器具不能直接使用方框中心夾取。
- 尚未建立可命名的放置姿勢與自動放置流程。
- 尚未使用馬達電流或受力資訊判斷是否真正夾住物品。

## 相關文件

- [系統設計、進度與後續規劃](DESIGN.md)
- [既有蘋果辨識與自動抓放指南](../apple_pick/APPLE_PICK_GUIDE.md)
- [手眼校正指南](../apple_pick/CALIBRATION_GUIDE.md)
