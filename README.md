# ViperX-300S 機械手臂：從 YOLO 換成 SAM 3

這是一支會「看」東西、再把東西夾起來的機械手臂。
以前它用 **YOLO** 當眼睛，現在換成 **SAM 3**。

> 這份說明寫給第一次接觸的人，國中生也能照著做。
> 看不懂的詞，先翻到最下面的「📖 小字典」。

> ⚠️ **SAM 3 版還沒在真的手臂上測過。** 哪些測過、哪些還沒，請看「[10. 目前進度和待辦](#10-目前進度和待辦)」。

---

## 1. 一分鐘看懂：YOLO 和 SAM 3 差在哪？

想像你請一個朋友幫你在桌上找蘋果：

| | YOLO（舊的眼睛） | SAM 3（新的眼睛） |
|---|---|---|
| 怎麼告訴它要找什麼？ | 只能找「考前背過」的東西（訓練過的類別） | **直接打字**，例如 `apple`、`red apple`、`bottle` |
| 找到後怎麼標示？ | 畫一個**方框** | 把物品的**輪廓整個塗出來**（遮罩） |
| 想找新東西？ | 要先拍很多照片、重新訓練 | 通常**換個詞**就好 |
| 速度 | 很快 | 比較慢，需要好一點的顯示卡（GPU） |

手臂要知道「蘋果在哪」，流程是這樣：

```
相機拍照 ──▶ SAM 3 找出蘋果的輪廓 ──▶ 取輪廓中間的點
        ──▶ 用深度相機量距離 ──▶ 換算成手臂的座標 ──▶ 手臂去夾
```

這次**只換了「找蘋果」那一步**，後面量距離、算座標、移動手臂的程式完全沒動。
所以原本調好的夾取動作可以照常使用。

---

## 2. 這次改了哪些檔案？

只改了很少的地方：

| 檔案 | 做了什麼 |
|---|---|
| `general_object_pick/sam3_detector.py` | **新檔案**。SAM 3 版的「眼睛」，大約 100 行 |
| `general_object_pick/generic_yolo_detector.py` | 把「用模型找東西」那一段拆成 `detect()`，讓 SAM 3 能直接替換；YOLO 的結果和以前一樣 |
| `general_object_pick/ui_prototype.py` | 操作畫面預設改用 SAM 3，舊的 YOLO 還能選 |
| `docker/Dockerfile` | 安裝能跑 SAM 3 的新版 `transformers`（5.0 以上） |
| `.gitignore` | 不讓 SAM 3 模型檔和登入金鑰被上傳到 GitHub |

SAM 3 的用法參考自 [VLMotion `feature/sam3-counting-3d`](https://github.com/hrc-pme/VLMotion/tree/feature/sam3-counting-3d)，用的是 Hugging Face 的 `Sam3Model` 和 `Sam3Processor`。
手臂程式來自 [VIPERX300S-rebuild](https://github.com/jerry12221/VIPERX300S-rebuild)，原作者的 Git 紀錄都有保留。

---

## 3. 開始前要準備什麼？

**硬體**

- 一台 Ubuntu 22.04 電腦，有 NVIDIA 顯示卡（建議顯示卡記憶體 8 GB 以上）
- ViperX-300S 機械手臂（用 U2D2 USB 接到電腦）
- Intel RealSense 深度相機
- **緊急停止開關放在手邊**，手臂動作不對時馬上按

**軟體**（電腦上要先裝好）

- Docker 和 Docker Compose
- NVIDIA Container Toolkit（讓 Docker 能用顯示卡）

**帳號**

- 一個免費的 [Hugging Face](https://huggingface.co) 帳號，用來下載 SAM 3

---

## 4. 第一次安裝（只要做一次）

### 步驟 1：下載這個專案

在**電腦的終端機**輸入：

```bash
git clone https://github.com/Gary0302/VIPERX300S.git
cd VIPERX300S
```

### 步驟 2：建立 Docker 環境

Docker 就像一個「裝好所有工具的箱子」，不會弄亂你的電腦。

```bash
cd docker
./build.sh      # 第一次會跑很久（可能 30 分鐘以上），等它跑完
./run.sh        # 問你要不要進去時，直接按 Enter
```

看到 `ROS2 Humble + GPU + Interbotix ViperX-300S 環境已就緒!` 就代表你已經在箱子裡了。

### 步驟 3：建置手臂程式

在**箱子裡（容器內）**輸入：

```bash
source /opt/ros/humble/setup.bash
cd /workspace/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### 步驟 4：拿到 SAM 3 模型

SAM 3 是 Meta 公司做的，要先申請才能下載（免費）。

1. 打開 <https://huggingface.co/facebook/sam3>，登入後按 **申請存取（Request access）**，等通過（通常很快）。
2. 打開 <https://huggingface.co/settings/tokens>，建立一個 **Read** 權限的 token（一串密碼），先複製起來。
3. 回到**容器內**，登入並下載模型（好幾 GB，請確認硬碟空間夠）：

```bash
hf auth login                     # 貼上剛剛的 token，按 Enter
hf download facebook/sam3 --local-dir /workspace/models/sam3
```

下載好的模型會放在 `models/sam3/`，程式會自動找到它。

> 🔒 token 就像密碼，**不要傳給別人，也不要貼到 GitHub**。
> 這個專案已經設定好不會上傳 `models/sam3/` 和 `.hf_cache/`。

---

## 5. 每次使用

你需要開 **3 個終端機視窗**，每一個都要先進到容器裡。

**每個視窗都先做這兩步：**

```bash
docker exec -it viperx300s_robot bash        # 在電腦上：進入容器
```

```bash
export ROS_DOMAIN_ID=42                       # 在容器內：讓 3 個視窗能互相溝通
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
```

> 3 個視窗的 `ROS_DOMAIN_ID` 一定要一樣，不然它們會「聽不到」彼此。

**視窗 1：啟動手臂的大腦（MoveIt）和 3D 畫面（RViz）**

```bash
ros2 launch interbotix_xsarm_moveit xsarm_moveit.launch.py robot_model:=vx300s hardware_type:=actual
```

**視窗 2：告訴電腦「相機裝在手臂哪裡」**

```bash
ros2 launch /workspace/apple_pick/hand_eye.launch.py
```

**視窗 3：打開操作畫面**

```bash
python3 /workspace/general_object_pick/ui_prototype.py
```

---

## 6. 在操作畫面上找東西、夾東西

1. **Backend** 選 `SAM 3 (Default)`（預設就是它）。
2. **Target object** 輸入你要找的東西，用英文：
   - `apple`：所有蘋果
   - `red_apple`：只找紅蘋果（程式會把 `_` 換成空格，變成 `red apple` 給 SAM 3）
   - `green_apple`：只找綠蘋果
   - 也可以試 `bottle`、`cup` 等其他東西，但**目前只有蘋果能真的夾**，其他東西只能看、不能夾
3. 按 **Start Vision**。第一次載入 SAM 3 要等一下。
4. 畫面上的蘋果會被**塗上綠色**，旁邊寫著名稱和分數（0～1，越接近 1 越有把握）。
5. 等 Camera、Detector、TF 都變成 `Ready`。
6. 按 **Start Robot** 連接手臂，等 MoveIt 和 Gripper 變成 `Ready`。
7. 桌上只放一顆蘋果，等座標穩定。
8. 按 **Plan Straight Pick**：**只規劃路線，手臂不會動**。到 RViz 看路線合不合理。
9. 確認安全後按 **Confirm and Pick**，再按 **Yes**，手臂就會去夾。

更詳細的操作（觀察姿勢、夾爪、一次夾很多顆）請看 [`general_object_pick/README.md`](general_object_pick/README.md)。

### 想改回舊的 YOLO？

在 **Backend** 選 `YOLO (Old)`，再從 **Model** 選 `printed_apple_yolo11s.pt`，其他步驟一樣。

---

## 7. 常見問題

| 狀況 | 原因和解決方法 |
|---|---|
| 出現 `401`、`gated repo`、`Access to model facebook/sam3 is restricted` | 還沒通過申請，或還沒登入。回到步驟 4 重做 |
| 出現 `cannot import name 'Sam3Model'` | `transformers` 太舊。重新執行 `./build.sh`，或在容器內執行 `pip3 install -U "transformers>=5.0.0"` |
| 出現 `CUDA out of memory` | 顯示卡記憶體不夠。關掉其他用到顯示卡的程式再試 |
| 畫面很卡、FPS 很低 | SAM 3 本來就比 YOLO 慢，這是正常的。手臂只在東西靜止時才夾，所以還是能用 |
| 找不到東西 | 換個說法試試，例如 `apple` 改成 `red apple`；或把光線弄亮一點 |
| 找到太多不對的東西 | 程式預設分數 0.5 以上才算數。可以在 `ui_prototype.py` 把 SAM 3 的 `detector_confidence = 0.5` 調高一點 |
| TF 一直是 `Waiting` | 視窗 1 或視窗 2 沒開，或 `ROS_DOMAIN_ID` 不一樣 |

想用別的地方的 SAM 3 模型（例如自己微調過的），可以這樣指定：

```bash
SAM3_MODEL=/workspace/models/my_sam3 python3 /workspace/general_object_pick/ui_prototype.py
```

---

## 8. 安全規則（很重要！）

- 手臂移動時，**手和臉不要靠近**。
- **緊急停止開關**一定要在伸手就按得到的地方。
- 每次夾之前都先用 **Plan** 看路線，確定沒問題才執行。
- 操作畫面上的停止按鈕**不能取代**實體的緊急停止開關。

---

## 9. 資料夾介紹

```text
VIPERX300S/
├── general_object_pick/   # 主要的操作畫面和辨識程式（SAM 3 在這裡）
├── apple_pick/            # 舊的蘋果流程（仍用 YOLO）、相機啟動、手眼校正
├── models/                # 模型檔；SAM 3 下載到 models/sam3/
├── ros2_ws/src/           # 手臂、相機、AprilTag 的 ROS 原始碼
└── docker/                # 建立和啟動 Docker 容器
```

- 舊的 `apple_pick/` 流程沒有改動，用法看 [蘋果辨識與自動抓放操作指南](apple_pick/APPLE_PICK_GUIDE.md)。
- 換了相機、夾爪或手臂的位置，要依 [校正指南](apple_pick/CALIBRATION_GUIDE.md) 重新校正。
- `docker/docker-compose.yml` 預設使用 `/dev/ttyUSB0` 和 `/dev/video0`～`/dev/video11`，別台電腦的裝置名稱不同時要修改。

---

## 10. 目前進度和待辦

### 已經確認過的

- 程式在**小型隨機 SAM 3 模型**加上**假的 ROS 模組**下跑得起來（`transformers` 5.0.0 和 5.23.0 都試過）。
- 輸入一張圖會吐出遮罩、方框、分數和中心點，而且格式跟 YOLO 一樣，後面算座標、移動手臂的程式可以直接用。
- 改回 `YOLO (Old)` 時，YOLO 的結果和改之前一樣。

### 還沒測過的（需要真的硬體）

- [ ] 下載真正的 `facebook/sam3` 權重（第 4 節步驟 4），確認載入沒有錯誤
- [ ] 開操作畫面，Target object 輸入 `apple`，按 Start Vision，確認：
  - [ ] 蘋果有被正確塗成綠色
  - [ ] 3D 座標合理（跟用尺量的差不多）
  - [ ] 記下 FPS（填到下面的表）
- [ ] 用真的蘋果調 SAM 3 的分數門檻（`ui_prototype.py` 的 `start_vision()` 裡 `detector_confidence = 0.5`）
  - [ ] **空桌子**時不會亂找到東西
  - [ ] 一顆、多顆、紅綠混在一起都找得到
- [ ] 桌上只放**一顆**蘋果，按 Plan Straight Pick，到 RViz 檢查路線，再 Confirm（**緊急停止開關放手邊**）

### 測完請把結果填在這裡

| 項目 | 結果 |
|---|---|
| 顯示卡型號 | （待填） |
| SAM 3 實際 FPS | （待填） |
| 選定的分數門檻 | （待填，目前預設 0.5） |
| 空桌子會不會誤判 | （待填） |
| 單顆蘋果夾取 | （待填：成功／失敗，原因） |

---

## 📖 小字典

| 詞 | 意思 |
|---|---|
| **YOLO** | 一種很快的物體辨識模型，只會找訓練過的東西，會畫方框 |
| **SAM 3** | Meta 做的「Segment Anything Model 3」。輸入文字就能找東西，還會畫出輪廓 |
| **遮罩（mask）** | 把物品的形狀一格一格塗出來的圖，比方框更準 |
| **分數（confidence）** | 模型覺得「這真的是蘋果」的把握程度，0 到 1 之間 |
| **深度相機** | 除了拍照，還能量出每個點離相機多遠 |
| **ROS 2** | 讓機器人各個程式互相傳訊息的系統 |
| **MoveIt** | 幫手臂規劃「怎麼動才不會撞到」的程式 |
| **RViz** | 在電腦上顯示手臂 3D 樣子和規劃路線的畫面 |
| **TF** | 記錄「相機在哪、手臂在哪」的座標關係 |
| **手眼校正** | 量出相機到底裝在手臂的哪個位置、朝哪個方向 |
| **Docker / 容器** | 一個裝好所有工具的箱子，不會弄亂你的電腦 |
| **GPU / 顯示卡** | 很會算圖的晶片，讓 SAM 3 跑得快 |
| **Hugging Face** | 一個放 AI 模型的網站，SAM 3 從這裡下載 |
| **token** | 登入 Hugging Face 用的一串密碼 |
