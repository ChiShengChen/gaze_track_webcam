# Gaze Heatmap 使用指南

## 三種模式

| 模式 | 模型 | 精度 | 特點 |
|------|------|------|------|
| **L2CS-Net** ⭐ | ResNet50 深度學習 | ~3.9° | **推薦**，分離角度預測，更穩健 |
| **ETH-XGaze** | ResNet18 深度學習 | ~4.5° | 需要 `gaze_eth` 環境 |
| **MediaPipe** | Iris Tracking | ~5-10° | 輕量級，默認 |

---

## 方式一：L2CS-Net 模型（推薦，最佳精度）

### 安裝 L2CS

L2CS-Net 需要從 GitHub 安裝：

```bash
# 在現有環境中安裝
conda activate gaze_eth  # 或你的環境
pip install git+https://github.com/Ahmednull/L2CS-Net.git

# 或創建新環境
conda create -n gaze_l2cs python=3.10 -y
conda activate gaze_l2cs
pip install -r requirements.txt
pip install git+https://github.com/Ahmednull/L2CS-Net.git
```

### 下載模型權重（可選）

L2CS 會自動下載權重，或手動下載：

```bash
# 創建模型目錄
mkdir -p ~/.l2cs/models

# 下載權重（從 GitHub releases）
# https://github.com/Ahmednull/L2CS-Net/releases
# 將 L2CSNet_gaze360.pkl 放到 ~/.l2cs/models/
```

### 使用

```bash
cd /Users/michael/Desktop/webcam_voice_label/gaze_heatmap

# 確保 config.yaml 中設置 model: "l2cs"
# 1. 校準
python main.py calibrate --output l2cs_calibration.yaml

# 2. 即時展示
python main.py demo --calibration l2cs_calibration.yaml

# 3. 錄製
python main.py record --calibration l2cs_calibration.yaml --duration 60

# 4. 評估精度
python main.py evaluate --calibration l2cs_calibration.yaml --num-points 20
```

---

## 方式一：ETH-XGaze 模型（推薦）

使用專用啟動腳本：

```bash
cd /Users/michael/Desktop/webcam_voice_label/gaze_heatmap

# 1. 校準
./run_eth.sh calibrate my_calibration.yaml

# 2. 即時展示
./run_eth.sh demo my_calibration.yaml

# 3. 錄製 60 秒
./run_eth.sh record my_calibration.yaml 60

# 4. 評估精度
./run_eth.sh evaluate my_calibration.yaml 16
```

---

## 方式二：MediaPipe 模式

使用原有 conda 環境：

```bash
cd /Users/michael/Desktop/webcam_voice_label/gaze_heatmap

# 激活環境
conda activate pytorch_12

# 1. 校準
python main.py calibrate --output my_calibration.yaml --validate

# 2. 即時展示
python main.py demo --calibration my_calibration.yaml

# 3. 錄製
python main.py record --calibration my_calibration.yaml --duration 60

# 4. 評估
python main.py evaluate --calibration my_calibration.yaml --num-points 16

# 5. 標註
python main.py label --session session_001
```

---

## 環境說明

| 環境 | Python | 用途 |
|------|--------|------|
| `gaze_eth` | 3.10 | ETH-XGaze 深度學習模型 |
| `pytorch_12` | - | MediaPipe fallback |

---

## 快速開始

```bash
# 最快方式（ETH-XGaze）
./run_eth.sh calibrate
./run_eth.sh demo
```


cd /Users/michael/Desktop/webcam_voice_label/gaze_heatmap

# 1. 校準（使用 ETH-XGaze）
./run_eth.sh calibrate

# 2. 即時展示
./run_eth.sh demo

# 激活環境
conda activate gaze_eth

# 進入目錄
cd /Users/michael/Desktop/webcam_voice_label/gaze_heatmap

# 設置環境變數
export KMP_DUPLICATE_LIB_OK=TRUE

# 運行
python main.py calibrate --output eth_calibration.yaml
python main.py demo --calibration eth_calibration.yaml

python main.py demo --calibration test_calibration.yaml