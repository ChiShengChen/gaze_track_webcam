# GPU 加速使用指南

## Apple Silicon (M1/M2/M3/M4) GPU 支持

### 當前狀態

✅ **PyTorch MPS 支持**：已確認可用
- PyTorch 2.9.1 支持 MPS (Metal Performance Shaders)
- 可以使用 Apple Silicon GPU 加速

✅ **ETH-XGaze 支持 MPS**：
- ETH-XGaze 完全支持 MPS (Apple Silicon GPU)
- 可以獲得 GPU 加速，提升性能

⚠️ **L2CS 限制**：
- L2CS 使用的 RetinaFace 人臉檢測器**不支持 MPS**
- 當設置 `device: "mps"` 時，L2CS 會自動回退到 CPU
- ETH-XGaze 不受此限制，可以使用 MPS

### 設備選項

在 `config.yaml` 中設置：

```yaml
gaze:
  device: "cpu"    # 推薦：穩定可靠
  # device: "mps"  # Apple Silicon GPU（會自動回退到 CPU）
  # device: "cuda" # NVIDIA GPU（MacBook 不支持）
```

### 性能說明

**CPU 模式（推薦）**：
- ✅ 穩定可靠
- ✅ 所有功能正常
- ✅ 性能足夠（30 FPS）
- ✅ 無兼容性問題

**MPS 模式（實驗性）**：
- ⚠️ L2CS 會自動回退到 CPU（RetinaFace 限制）
- ⚠️ ETH-XGaze 可能支持 MPS（需要測試）
- ℹ️ 對於視線追蹤，CPU 性能通常足夠

### 檢查 GPU 支持

```bash
conda activate gaze_eth
python << 'EOF'
import torch
print(f"MPS 可用: {torch.backends.mps.is_available()}")
print(f"MPS 已構建: {torch.backends.mps.is_built()}")
EOF
```

### 建議

1. **使用 CPU 模式**（推薦）
   - 最穩定可靠
   - 性能足夠
   - 無兼容性問題

2. **如果需要 GPU 加速**
   - 可以嘗試 `device: "mps"`
   - 系統會自動處理兼容性問題
   - 如果失敗會自動回退到 CPU

### 性能對比

| 模型 | CPU | MPS (GPU) | 備註 |
|------|-----|-----------|------|
| **ETH-XGaze** | ✅ 支持 | ✅ **支持** | **推薦使用 MPS 獲得 GPU 加速** |
| **L2CS-Net** | ✅ 支持 | ⚠️ 回退到 CPU | RetinaFace 不支持 MPS |
| **MediaPipe** | ✅ 支持 | ✅ 支持 | 輕量級，無 GPU 需求 |

### 推薦配置

**使用 ETH-XGaze + MPS（推薦，GPU 加速）**：
```yaml
gaze:
  model: "eth-xgaze"
  device: "mps"  # 使用 Apple Silicon GPU
```

**使用 L2CS（最佳準確度，CPU）**：
```yaml
gaze:
  model: "l2cs"
  device: "cpu"  # L2CS 不支持 MPS，使用 CPU
```

### 總結

- ✅ **MacBook (M4) 可以使用 GPU 加速 ETH-XGaze**
- ✅ **ETH-XGaze + MPS**：推薦配置，獲得 GPU 加速
- ⚠️ **L2CS + MPS**：會自動回退到 CPU（RetinaFace 限制）
- ✅ **系統會自動處理**：選擇最適合的設備
- ✅ **所有功能正常**：無論使用哪個設備，功能都完全可用

