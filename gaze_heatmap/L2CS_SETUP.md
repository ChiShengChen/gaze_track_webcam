# L2CS-Net 設置指南

## 為什麼使用 L2CS-Net？

L2CS-Net 相比 ETH-XGaze 的優勢：
- ✅ **更好的準確度**: MPIIGaze 上 3.92° vs ETH-XGaze 的 4.5°
- ✅ **分離角度預測**: 分別預測 pitch 和 yaw，對 in-the-wild 場景更穩健
- ✅ **更簡單的集成**: 直接輸出角度，無需複雜的座標轉換

## 安裝步驟

### 1. 安裝 L2CS 包

L2CS-Net 需要從 GitHub 安裝，不能直接從 PyPI 安裝：

```bash
# 在 gaze_eth 環境中
conda activate gaze_eth
pip install git+https://github.com/Ahmednull/L2CS-Net.git

# 或在新環境中
conda create -n gaze_l2cs python=3.10 -y
conda activate gaze_l2cs
pip install -r requirements.txt
pip install git+https://github.com/Ahmednull/L2CS-Net.git
```

### 2. 下載模型權重（必需）

L2CS 需要預訓練權重文件。請手動下載：

#### 方法一：使用下載腳本（推薦）

```bash
cd gaze_heatmap
python download_l2cs_weights.py
```

腳本會嘗試自動下載，如果失敗會提供手動下載指引。

#### 方法二：手動下載

1. **訪問 GitHub Releases**：
   ```
   https://github.com/Ahmednull/L2CS-Net/releases
   ```

2. **下載權重文件**（選擇一個）：
   - `L2CSNet_gaze360.pkl` - 推薦，用於 Gaze360 數據集
   - `L2CSNet_MPIIGaze.pkl` - 用於 MPIIGaze 數據集

3. **放到正確位置**（按優先級）：
   
   **推薦位置**（項目目錄）：
   ```bash
   cd gaze_heatmap
   mkdir -p l2cs_model
   # 將下載的文件移動或複製到：
   mv ~/Downloads/L2CSNet_gaze360.pkl l2cs_model/
   ```
   
   **或標準位置**（用戶目錄）：
   ```bash
   mkdir -p ~/.l2cs/models
   # 將下載的文件移動或複製到：
   mv ~/Downloads/L2CSNet_gaze360.pkl ~/.l2cs/models/
   ```
   
   系統會按以下順序搜索權重文件：
   1. `gaze_heatmap/l2cs_model/L2CSNet_gaze360.pkl` （推薦）
   2. `~/.l2cs/models/L2CSNet_gaze360.pkl`
   3. `~/.l2cs/L2CSNet_gaze360.pkl`
   4. `gaze_heatmap/models/L2CSNet_gaze360.pkl`

#### 方法三：使用 wget/curl（如果知道直接鏈接）

```bash
mkdir -p ~/.l2cs/models
# 從 GitHub releases 頁面獲取直接下載鏈接後執行：
wget <direct_download_url> -O ~/.l2cs/models/L2CSNet_gaze360.pkl
# 或
curl -L <direct_download_url> -o ~/.l2cs/models/L2CSNet_gaze360.pkl
```

**注意**：
- 如果沒有權重文件，L2CS 將無法初始化
- 系統會**自動回退到 ETH-XGaze**（如果可用）
- 如果 ETH-XGaze 也不可用，會回退到 MediaPipe
- 當前系統已配置為優先使用 ETH-XGaze，所以即使沒有 L2CS 權重也能正常工作

### 3. 配置

編輯 `config.yaml`：

```yaml
gaze:
  model: "l2cs"  # 改為 l2cs
  device: "cpu"   # 或 "cuda" 如果有 GPU
```

## 使用

```bash
# 1. 校準
python main.py calibrate --output l2cs_calibration.yaml

# 2. 運行 demo
python main.py demo --calibration l2cs_calibration.yaml

# 3. 評估
python main.py evaluate --calibration l2cs_calibration.yaml --num-points 20
```

## 故障排除

### 問題：找不到 l2cs 模塊

```bash
pip install l2cs
```

### 問題：找不到模型權重

L2CS 會自動下載，或手動下載：
- 訪問：https://github.com/Ahmednull/L2CS-Net
- 下載預訓練權重
- 放到 `~/.l2cs/models/` 或項目目錄

### 問題：CUDA 錯誤（如果有 GPU）

```bash
# 確保安裝了正確的 PyTorch CUDA 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 性能比較

| 模型 | MPIIGaze 誤差 | 特點 |
|------|---------------|------|
| L2CS-Net | **3.92°** | 分離角度預測，推薦 |
| ETH-XGaze | 4.5° | 大規模訓練數據 |
| MediaPipe | ~5-10° | 輕量級，無需訓練 |

## 參考資料

- GitHub: https://github.com/Ahmednull/L2CS-Net
- Paper: "Looking at the Right Location: L2CS-Net for Gaze Estimation"

