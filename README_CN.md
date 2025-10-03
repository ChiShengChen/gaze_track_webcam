# 眼球追蹤系統 (Eye Tracking System)

一個基於Python的即時眼球追蹤系統，使用webcam進行視線追蹤、語音校正和即時熱區圖顯示。專為macOS設計，支援中英文語音指令。

## 功能特色

- 🎯 **即時視線追蹤**: 使用Mediapipe進行高精度臉部特徵提取
- 🎤 **語音校正**: 支援"here"/"這裡"語音指令進行校正和記錄
- 🔥 **即時熱區圖**: 即時顯示注視點熱區圖，支援平滑和衰減效果
- 📊 **數據記錄**: 自動記錄注視點數據到CSV檔案
- 🖥️ **多螢幕支援**: 自動偵測主螢幕解析度
- 🎨 **視覺化除錯**: 可選的攝影機除錯視窗
- 🚀 **高精度模式**: 多幀平均、多項式回歸、邊界加權提升準確度
- 🔧 **攝影機鏡像**: 可選水平鏡像修正左右追蹤問題
- 📈 **準確度評估**: 內建5x5測試網格進行定量準確度評估

## 系統需求

- macOS 10.15+
- Python 3.8+
- 內建或外接webcam
- 麥克風
- 至少4GB RAM

## 安裝步驟

### 1. 建立虛擬環境

```bash
# 建立虛擬環境
python3 -m venv venv

# 啟動虛擬環境
source venv/bin/activate
```

### 2. 安裝依賴

```bash
# 安裝Python套件
pip install -r requirements.txt

# 如果遇到音訊問題，安裝portaudio
brew install portaudio
pip install --force-reinstall sounddevice
```

### 3. 下載Vosk語音模型

下載並解壓縮Vosk語音模型到專案目錄：

**英文模型**:
```bash
# 下載英文小模型
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

**中文模型**:
```bash
# 下載中文小模型
wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
unzip vosk-model-small-cn-0.22.zip
```

## 使用方法

### 快速啟動（推薦）

```bash
# 最簡單的啟動方式，使用預設設定
./quick_start.sh
```

### 高精度模式（推薦最佳準確度）

```bash
# 高精度版本，包含多幀平均和多項式回歸
./run_high_precision.sh
```

### 基本使用

```bash
# 啟動基本版本（會詢問是否顯示除錯視窗）
./run_basic.sh

# 啟動進階版本（全螢幕疊圖）
./run_overlay.sh

# 手動啟動（使用預設英文模型）
python gaze_tracker.py

# 顯示攝影機除錯視窗
python gaze_tracker.py --show-cam-debug

# 高精度模式配合攝影機鏡像
python gaze_tracker.py --rows 4 --cols 4 --cam-mirror
```

### 進階參數

```bash
# 自定義校正點數量
python gaze_tracker.py --rows 4 --cols 4

# 使用不同攝影機
python gaze_tracker.py --camera 1

# 使用中文模型（如果已下載）
python gaze_tracker.py --vosk-model ./vosk-model-small-cn-0.22
```

### 參數說明

- `--vosk-model`: Vosk語音模型路徑，預設為 `./vosk-model-small-en-us-0.15`
- `--camera`: 攝影機索引，預設為0
- `--rows`: 校正點行數，預設為3
- `--cols`: 校正點列數，預設為3
- `--show-cam-debug`: 顯示攝影機除錯視窗
- `--cam-mirror`: 水平鏡像攝影機（修正左右追蹤問題）

## 使用流程

### 1. 校正階段

程式啟動後會進入全螢幕校正模式：

1. 螢幕會顯示綠色圓點
2. **盯著圓點**並說出"here"或"這裡"
3. 系統會記錄你的臉部特徵對應到螢幕座標
4. 重複此過程直到所有校正點完成

**校正建議**:
- 保持正常坐姿，距離螢幕50-70cm
- 確保光線充足且均勻
- 避免戴反光眼鏡
- 校正過程中保持頭部相對穩定
- **高精度模式**: 使用4x4或5x5校正網格提升邊角準確度
- **多幀平均**: 系統會自動收集每個校正點0.4秒的樣本

### 2. 推論階段

校正完成後進入即時追蹤模式：

- 即時顯示熱區圖視窗
- 可選顯示攝影機除錯視窗
- 說"here"或"這裡"可記錄當前注視點到CSV檔案
- 按'q'鍵退出程式

## 輸出檔案

### gaze_points.csv

記錄所有語音觸發的注視點數據：

```csv
timestamp,x,y
1640995200.123,960,540
1640995205.456,1200,300
```

## 準確度與限制

### 預期準確度

- **高精度模式**: 誤差約1-2cm（0.5-1度視角）
- **理想條件**: 誤差約1-3cm（1-2度視角）
- **一般條件**: 誤差約3-5cm
- **困難條件**: 誤差可能超過5cm

### 準確度改善（v2.0）

系統現在包含多項準確度提升功能：

- **多幀平均**: 每個校正點收集0.4秒樣本，降低雜訊
- **多項式回歸**: 使用2次多項式特徵，改善非線性映射
- **邊界加權**: 對邊緣校正點給予更高權重，提升角落準確度
- **攝影機鏡像**: 可選水平鏡像修正左右追蹤問題

### 影響因素

- **光線條件**: 背光、陰影會降低準確度
- **頭部姿態**: 大幅擺頭會影響追蹤
- **眼鏡**: 反光鏡片可能干擾
- **距離**: 過近或過遠都會影響準確度
- **攝影機位置**: 偏離中心的攝影機位置會影響邊緣準確度

### 改善建議

1. **使用高精度模式**: 執行 `./run_high_precision.sh` 獲得最佳準確度
2. **增加校正點**: 使用4x4或5x5網格提高準確度
3. **啟用攝影機鏡像**: 如果追蹤左右顛倒
4. **定期重校正**: 每30分鐘重新校正一次
5. **優化環境**: 確保光線均勻，避免背光
6. **保持姿勢**: 盡量保持穩定的觀看姿勢
7. **評估準確度**: 使用 `python evaluate_accuracy.py` 測試追蹤精度

## 故障排除

### 常見問題

**1. 攝影機無法開啟**
```bash
# 檢查攝影機權限
# 系統偏好設定 > 安全性與隱私 > 相機
```

**2. 麥克風無法使用**
```bash
# 檢查麥克風權限
# 系統偏好設定 > 安全性與隱私 > 麥克風
```

**3. 音訊錯誤**
```bash
# 重新安裝音訊驅動
brew install portaudio
pip install --force-reinstall sounddevice
```

**4. 語音辨識不準確**
- 確保環境安靜
- 說話清晰，音量適中
- 嘗試重新下載語音模型

**5. 追蹤不準確**
- 使用高精度模式重新校正
- 檢查光線條件
- 調整坐姿和距離
- 增加校正點數量
- 如果左右追蹤顛倒，嘗試攝影機鏡像
- 使用準確度評估工具測試精度

### 效能優化

**降低CPU使用率**:
```bash
# 降低攝影機解析度
python gaze_tracker.py --vosk-model ./model --camera 0
```

**提高準確度**:
```bash
# 使用高精度模式（推薦）
./run_high_precision.sh

# 或手動增加校正點配合多項式回歸
python gaze_tracker.py --rows 4 --cols 4 --cam-mirror

# 評估準確度
python evaluate_accuracy.py
```

## 進階功能

### 全螢幕疊圖模式

如需將熱區圖疊加在所有視窗上方，請參考 `gaze_overlay.py` 進階版本。

### 自定義熱區圖

可以修改 `Heatmap` 類別來自定義：
- 顏色映射
- 衰減速度
- 模糊程度
- 網格密度

### 數據分析

使用生成的CSV檔案進行注視點分析：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 讀取數據
df = pd.read_csv('gaze_points.csv')

# 繪製注視點分佈
plt.scatter(df['x'], df['y'], alpha=0.6)
plt.title('注視點分佈')
plt.xlabel('X座標')
plt.ylabel('Y座標')
plt.show()
```

## 技術架構

### 核心技術

- **Mediapipe**: 臉部特徵提取和虹膜偵測
- **Vosk**: 離線語音辨識
- **OpenCV**: 影像處理和顯示
- **Scikit-learn**: 回歸模型訓練
- **NumPy**: 數值計算

### 演算法流程

1. **特徵提取**: 提取左右眼虹膜相對於眼眶的歸一化座標
2. **校正映射**: 建立臉部特徵到螢幕座標的映射關係
3. **即時推論**: 使用訓練好的模型預測注視點
4. **平滑處理**: 使用指數移動平均減少抖動
5. **熱區圖生成**: 累積注視點並生成視覺化熱區圖

## 授權

本專案採用MIT授權條款。

## 貢獻

歡迎提交Issue和Pull Request來改善這個專案。

## 聯絡方式

如有問題或建議，請透過GitHub Issues聯繫。
