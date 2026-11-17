# 3D 列印蘋果資料蒐集、標註與訓練指南

這份文件涵蓋完整資料流程：使用現有 UI 從 ROS 2 原始彩色影像 topic 儲存 JPEG、整理照片、自動預標註、人工審查、建立 train/validation/test，以及微調 YOLO11。UI 畫面仍會顯示 YOLO 辨識框，但資料集保存的是未畫框原圖。

## 先理解三件事

- UI 的 `Start Auto Capture` 會自動訂閱 `/camera/camera/color/image_raw`，不需要再開一個終端機執行拍照程式。
- `Capture category` 只會寫進 session 紀錄，方便整理；它不會在拍攝時產生 YOLO 方框。拍完後使用本專案的 `auto_label_dataset.py` 與 `dataset_annotator.py` 完成標註。
- ROS 2 bag 是可選的完整訊息錄影，適合除錯及重播相機資料；單純建立 YOLO 資料集時，直接用 UI 存 JPEG 較省空間也較容易整理。

## 啟動方式

在主機啟動容器：

```bash
cd /home/jerry/jerry/viperx300s-VLA/docker
./run.sh
```

進入容器的新終端機都先設定環境：

```bash
docker exec -it viperx300s_robot bash
export ROS_DOMAIN_ID=42
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
```

只蒐集照片時不需要啟動 MoveIt、RViz、手眼 TF 或 Robot manager，只需啟動 UI：

```bash
python3 /workspace/general_object_pick/ui_prototype.py
```

如果同時要進行手臂夾取測試，才照主 README 的三終端機流程啟動 MoveIt、手眼 TF 與 UI。不要為了改變視角而直接拖動已上扭力的實體手臂；請移動蘋果、轉動蘋果、調整相機允許的安全位置，或使用已確認安全的觀察姿勢。

這次新增的是掛載中的 Python 程式，不需要重建 Docker；更新程式後關閉並重新啟動 UI 即可。

## UI 蒐集步驟

1. 按 `Start Vision`，等 Camera 顯示 `Ready`。UI 會啟動或沿用 RealSense topic。
2. 捲到 `6. Dataset Capture`。
3. `Session` 使用英文、數字、底線或減號，例如 `red_table_light_01`。每換一次場景、光線或拍攝日期，建議建立新 session。
4. 選擇 `Capture category`：
   - `red_apple`：只有紅蘋果。
   - `green_apple`：只有綠蘋果。
   - `mixed_apples`：紅綠同時出現。
   - `negative`：沒有蘋果的背景、手、夾爪或容易誤判的物品。
   - `apple`：暫時不區分顏色。
5. 設定 `Interval`。建議從 `0.5 s` 開始；移動很慢時可用 `1.0 s`，避免大量幾乎相同的照片。
6. 設定 `Maximum`，例如 `100`。
7. 按 `Start Auto Capture`。狀態會顯示本次新增數、session 總數及剩餘空間。
8. 緩慢改變蘋果的方向、位置、數量、遮擋、背景及光線。每個畫面停留一下，讓相機取得清楚影像。
9. 達到最大數量會自動停止，也可以按 `Stop Capture` 提前停止。
10. `Save Snapshot` 只保存一張，適合補拍特定困難畫面。

按 `Stop Vision / Cancel Motion` 或關閉 UI 時，正在執行的資料蒐集也會先停止。程式預設在磁碟可用空間低於 2 GB 時自動停止。

## 照片與紀錄位置

資料保存在：

```text
/workspace/datasets/printed_apple/raw/<session>/
├── images/
│   ├── frame_000001_....jpg
│   └── ...
└── session.json
```

主機上的對應位置是：

```text
/home/jerry/jerry/viperx300s-VLA/datasets/printed_apple/raw/<session>/
```

`session.json` 記錄 scene label、影像 topic、拍攝間隔、解析度及張數。原始照片預設不加入 Git，避免 repository 因大量影像變得過大；之後應備份到獨立硬碟、雲端或專用資料集儲存空間。

## 怎麼拍才有幫助

不要只連拍固定畫面。每一組都要刻意涵蓋：

- 蘋果正面、側面、斜面、蒂朝上與蒂朝旁邊。
- 畫面中央、四周、靠近與較遠的位置。
- 單顆、兩顆、三到四顆、部分互相遮擋。
- 明亮、稍暗、側光、陰影及可能出現的反光。
- 實際工作桌、容器、夾爪、手以及其他相似形狀物品。
- 完全沒有蘋果的 negative 場景。

第一版可先蒐集約 250–400 張有效且差異明顯的照片：

| 場景 | 建議張數 |
|---|---:|
| 紅蘋果 | 60–100 |
| 綠蘋果 | 60–100 |
| 紅綠混合與遮擋 | 80–120 |
| 無蘋果與容易誤判的負樣本 | 40–80 |

兩顆紅色加兩顆綠色 3D 列印蘋果足以開始第一版，尤其若系統只會操作這四顆實物。不過照片很多不等於物件種類很多；模型仍可能記住這四顆的列印紋路或缺陷。若未來要辨識其他批次、尺寸、材質或表面處理的列印蘋果，應再加入更多實體樣本。

為避免資料洩漏，不要把同一次連拍的相鄰影格逐張隨機拆到不同集合。目前 `build_training_dataset.py` 會先把連續影格組成短片段再切分；日後累積多個獨立 session 後，應優先改為整個 session 一起切分，測試結果才更接近新場景表現。

## 要標成 apple 還是紅綠兩類

如果眼前目標是先提升夾取成功率，可以全部標成單一 `apple` 類別；資料較容易累積，模型也較穩定。

如果後續流程確實要分別選取紅蘋果與綠蘋果，可以從現在就平衡蒐集兩種顏色，標註時使用 `red_apple` 與 `green_apple` 兩類。兩類的張數、距離、背景與光線分布要盡量接近，避免模型把背景或亮度誤認為顏色。

## ROS 2 bag 什麼時候使用

bag 會保存 ROS 訊息，之後即使沒有接相機，也能重播當時的彩色、深度、相機參數或 TF，適合：

- 相機或深度座標偶發錯誤，需要保留完整現場資料除錯。
- 想在不同 detector 版本上重播完全相同的場景。
- 需要日後重新抽取不同頻率或格式的影像。

只錄彩色影像的例子：

```bash
ros2 bag record /camera/camera/color/image_raw /camera/camera/color/camera_info
```

需要深度與 TF 除錯時才加入相關 topics。原始彩色與深度資料量很大，bag 不會取代 YOLO 標註，也不是這套 UI 自動拍照功能的必要條件。

## 檢查相機與手動執行蒐集器

UI 找不到相機時先檢查：

```bash
ros2 topic hz /camera/camera/color/image_raw
```

一般不需要手動執行蒐集器；若要在沒有 UI 的情況下測試，可以使用：

```bash
python3 /workspace/general_object_pick/dataset_capture.py \
  --output /workspace/datasets/printed_apple/raw/manual_test \
  --interval 0.5 \
  --max-images 20 \
  --scene-label red_apple
```

## 紅綠蘋果標註規則

目前模型使用兩個類別：

```text
0 = red_apple
1 = green_apple
```

每顆蘋果各自畫一個緊貼外形的矩形框，並包含果梗。混合照片中的紅、綠蘋果要分別標註；看得清楚但部分遮擋時仍要標註，嚴重模糊、幾乎完全遮住或無法確定時則排除，不要把它當成空背景。

`negative` 圖片保留空的 `.txt`，代表已確認畫面沒有目標。`review_exclude` 不進入標註與訓練流程。

## 自動預標註

在容器終端機執行：

```bash
export ROS_DOMAIN_ID=42
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
python3 /workspace/general_object_pick/auto_label_dataset.py
```

程式讀取既有的 `/workspace/models/yolo11s.pt`，不會下載另一個大型模型。輸出位置：

```text
/workspace/datasets/printed_apple/labels/
/workspace/datasets/printed_apple/auto_label_report.csv
```

預設不會覆蓋既有標註。人工審查開始後不要使用 `--overwrite`；預標註只是起點，仍需補上漏框、刪除錯框並確認紅綠類別。

## 人工審查標註

啟動審查介面：

```bash
python3 /workspace/general_object_pick/dataset_annotator.py
```

操作方式：

- 在空白處按住滑鼠左鍵拖曳：新增框。
- 框可能與既有框重疊時：按住 `Shift` 再拖曳。
- 點框內：選取框。
- `Delete Selected Box` 或 `Delete`：刪除錯誤框。
- 框的位置不準：先刪除，再重新畫框。
- `R`：紅蘋果；`G`：綠蘋果。
- `Ctrl+S`：儲存，但不會標記為審查完成。
- `Space` 或 `Mark Reviewed + Next`：儲存、標記完成並前往下一張。
- `Next (not reviewed)` 或右方向鍵：只前往下一張，不算人工審查完成。
- `Unreviewed only`：只顯示尚未人工確認的照片。

每張照片都要確認：

1. 每顆可辨識蘋果都有框。
2. 沒有把手、紙張、衛生紙包、控制器或背景框成蘋果。
3. 紅色是 class `0`，綠色是 class `1`。
4. 框包含完整可見蘋果與果梗，且不要包含太多背景。
5. 真正的負樣本保持零個框，再按 `Mark Reviewed + Next`。

審查進度保存在：

```text
/workspace/datasets/printed_apple/annotation_review.json
```

關閉程式時會保存目前修改過的標註；但只有 `Space` 或 `Mark Reviewed + Next` 會將照片記錄為已審查。全部照片審查完成後才能建立訓練集。

## 建立 train、validation 與 test

```bash
python3 /workspace/general_object_pick/build_training_dataset.py
```

程式會檢查每張照片都已審查、每個 YOLO 框格式正確，再將相鄰影格分組並切成約 `70% / 20% / 10%`。輸出位置：

```text
/workspace/datasets/printed_apple/yolo_dataset/
```

`training_exclude.txt` 記錄不應用於訓練的模糊或嚴重遮擋照片，避免空標籤讓模型誤學成「蘋果是背景」。切分結果與來源記錄在 `manifest.csv`。

## 微調 YOLO11s

目前使用 RTX GPU 及容器內既有的 Ultralytics，無須重建 Docker。訓練範例：

```bash
yolo detect train \
  model=/workspace/models/yolo11s.pt \
  data=/workspace/datasets/printed_apple/yolo_dataset/data.yaml \
  epochs=100 patience=20 imgsz=640 batch=16 device=0 workers=4 \
  project=/workspace/models/training_runs name=printed_apple_yolo11s \
  seed=42 deterministic=True \
  hsv_h=0.005 hsv_s=0.25 hsv_v=0.20 \
  degrees=8 translate=0.10 scale=0.30 fliplr=0.5 flipud=0.0 \
  mosaic=0.5 close_mosaic=10 plots=True
```

訓練完成後不能只看 mAP。至少要另外檢查：

- test 集的紅、綠蘋果 precision 與 recall。
- 所有空場景在實際使用門檻下是否產生假框。
- 單顆紅色、單顆綠色、紅綠混合及手部遮擋畫面。

這次訓練中，Ultralytics 依 mAP 選出的 `best.pt` 會在空場景產生誤判，因此實際採用第 24 輪的 `last.pt`。以 `0.55` 信心門檻測試時，test precision 為 `1.00`、recall 約 `0.974`，34 張空場景沒有假框。確認權重後複製到 UI 會掃描的模型目錄：

```bash
cp /workspace/models/training_runs/printed_apple_yolo11s/weights/last.pt \
  /workspace/models/printed_apple_yolo11s.pt
```

重新啟動 UI 後選擇 `printed_apple_yolo11s.pt`，Target 維持 `apple`。辨識節點會合併 `red_apple` 與 `green_apple` 供既有夾取流程使用，畫面上的框仍顯示實際顏色類別。
