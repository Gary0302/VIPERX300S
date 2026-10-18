# 辨識夾取任務

這個資料夾集中相機啟動、蘋果辨識、抓放、手眼校正及輔助程式。實際操作請從 [蘋果辨識與自動抓放操作指南](APPLE_PICK_GUIDE.md) 開始；需要重新校正時請看 [校正指南](CALIBRATION_GUIDE.md)。

| 檔案 | 用途 |
| --- | --- |
| `start_camera.sh` | 啟動 ROS 的 RealSense 相機節點與對齊深度影像 |
| `yolo_detector6.py` | YOLO 辨識與熟／青蘋果分類，發布目標 TF |
| `pick_and_place3.py` | 依序抓取熟蘋果並放置 |
| `hand_eye.launch.py` | 目前這套硬體的手眼校正 TF |
| `hand_eye_calibration.py`、`tags.yaml` | 重新校正使用 |
| `record_arm_pose.py` | 記錄手臂關節位置 |
| `paw.py` | 單獨測試夾爪 |
| `APPLE_PICK_GUIDE.md` | 從硬體確認到自動抓放的完整執行順序 |
| `CALIBRATION_GUIDE.md` | 手臂與手眼校正及常見問題排除 |

模型權重放在上一層的 `models/`。預設使用 `models/yolo11s.pt`。換模型時，於**容器內**啟動辨識程式並指定：

```bash
YOLO_MODEL=models/new_apple.pt APPLE_CLASS_ID=0 python3 /workspace/apple_pick/yolo_detector6.py
```

相對路徑從專案根目錄 `/workspace` 計算。`APPLE_CLASS_ID` 預設是 47；換成自訂模型時要核對其蘋果類別 ID。程式的熟／青蘋果分類、深度計算及 TF 名稱仍依現有邏輯運作，不能只換成完全不同任務的模型權重。
