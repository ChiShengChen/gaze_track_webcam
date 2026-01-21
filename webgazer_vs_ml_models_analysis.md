# WebGazer.js vs L2CS-Net / ETH-XGaze：為何簡單方法勝過 SOTA 深度學習？

> 深度分析為何瀏覽器端的 WebGazer.js 視線追蹤效果遠比 Python SOTA ML 模型（L2CS-Net、ETH-XGaze）更好

---

## 核心發現：反直覺的結果

理論上，L2CS-Net 和 ETH-XGaze 是 SOTA (State-of-the-Art) 深度學習模型，應該比 WebGazer.js 的傳統回歸方法更準確。但實際使用中 WebGazer.js 表現更好，原因在於**系統設計哲學的差異**，而非算法優劣。

---

## 重要澄清：WebGazer.js 的真實架構

### 常見誤解
很多人以為 WebGazer.js 是「完全從頭訓練」的簡單方法。**這是錯誤的**。

### 實際架構：混合系統

```
┌─────────────────────────────────────────────────────────────┐
│                    預訓練組件 (固定)                          │
│  MediaPipe FaceMesh (TensorFlow.js) - 468 個面部關鍵點        │
│          ↓                                                   │
│     眼部區域裁剪 (基於 landmark 索引)                         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  特徵提取 (固定流程)                          │
│  眼部圖像 → 縮放至 10×6 → 灰度化 → 直方圖均衡化               │
│          → 120 維特徵向量 (左眼 60 + 右眼 60)                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│               校準時訓練 (Ridge Regression)                   │
│  120 維眼部像素特徵 → 螢幕座標 (x, y)                         │
│  - 用戶點擊收集訓練數據 (50 個窗口)                           │
│  - 滑鼠移動軌跡 (10 個窗口，1 秒內)                           │
│  - 每 500ms 重新訓練回歸係數                                  │
└─────────────────────────────────────────────────────────────┘
```

### WebGazer.js 源碼證據

**1. 使用 MediaPipe FaceMesh 預訓練模型**
```javascript
// facemesh.mjs
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';

const TFFaceMesh = function() {
  this.model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
  this.detector = null;
};
```

**2. 眼部特徵提取（120 維像素特徵）**
```javascript
// util.mjs
util.getEyeFeats = function(eyes) {
  let process = (eye) => {
    let resized = this.resizeEye(eye, 10, 6);  // 縮放至 10x6
    let gray = this.grayscale(resized.data, resized.width, resized.height);
    let hist = [];
    this.equalizeHistogram(gray, 5, hist);  // 直方圖均衡化
    return hist;  // 60 像素/眼
  };
  return [].concat(process(eyes.left), process(eyes.right));  // 共 120 維
}
```

**3. Ridge Regression 校準訓練**
```javascript
// ridgeReg.mjs
reg.RidgeReg.prototype.predict = function(eyesObj) {
  // 合併點擊和滑鼠軌跡數據
  var eyeFeatures = this.eyeFeaturesClicks.data.concat(trailFeat);
  var screenXArray = this.screenXClicksArray.data.concat(trailX);
  
  // 訓練 Ridge Regression
  var coefficientsX = util_regression.ridge(screenXArray, eyeFeatures, this.ridgeParameter);
  var coefficientsY = util_regression.ridge(screenYArray, eyeFeatures, this.ridgeParameter);
  
  // 預測：係數 × 特徵
  var predictedX = 0;
  for(var i=0; i< eyeFeats.length; i++){
    predictedX += eyeFeats[i] * coefficientsX[i];
  }
}
```

**4. 持續再訓練機制**
```javascript
// ridgeWorker.mjs
setInterval(retrain, trainInterval); // 每 500ms 重新訓練
```

### 組件對比：兩者都用預訓練模型！

| 組件 | WebGazer.js | L2CS-Net / ETH-XGaze |
|------|-------------|---------------------|
| **面部偵測** | MediaPipe FaceMesh (預訓練) | RetinaFace / MediaPipe (預訓練) |
| **眼部特徵** | 120 維原始像素 (無 CNN) | ResNet50 深度特徵 (預訓練) |
| **校準訓練的是什麼** | Ridge Regression 係數 | Polynomial Regression 係數 |
| **校準輸入** | 原始眼部像素特徵 | CNN 預測的 3D 角度 |

---

## 第一部分：WebGazer.js vs L2CS-Net 對比

### 算法對比

| 維度 | WebGazer.js (webapp) | L2CS-Net (Python) |
|------|---------------------|-------------------|
| **核心算法** | Ridge Regression (嶺回歸) | ResNet50 + 雙分支 CNN |
| **特徵提取** | 眼部區域原始像素 | 深度學習特徵 |
| **輸出** | 直接預測螢幕座標 (x, y) | 3D 視線角度 (pitch, yaw) |
| **校準** | 用戶點擊訓練回歸模型 | 多項式回歸映射到螢幕 |
| **平滑** | 內建簡單濾波 | Kalman / One Euro Filter |

### 關鍵差異分析

#### 1. 端到端 vs 兩階段映射（最重要）

**WebGazer.js 的優勢：直接映射**
```
眼部像素 → [Ridge Regression] → 螢幕座標 (x, y)
          校準時學習完整映射
```

**L2CS-Net 的劣勢：兩階段映射**
```
眼部像素 → [ResNet50 CNN] → 3D 視線角度 (pitch, yaw)
                                    ↓
          → [Polynomial Regression] → 螢幕座標 (x, y)
                    需要額外校準
```

**問題所在**：
- L2CS 訓練目標是 **3D 視線方向**（在 Gaze360 數據集上），而非螢幕座標
- 從 3D 角度到 2D 螢幕座標需要**二次映射**，引入額外誤差
- 校準系統（`calibration.py`）用多項式回歸擬合，但這個映射本身就是近似的

#### 2. 校準精度與用戶適應性

**WebGazer.js**
```typescript
// main.ts - 校準過程
calibrationDots.forEach(dot => {
    dot.onclick = () => {
        calibrationClicks++;
        // 每次點擊都直接訓練回歸模型
        // 模型學習的是：眼睛外觀 → 這個螢幕位置
    }
});
```
- 校準時**直接學習用戶特定的映射**
- 包含了所有個人因素：眼睛形狀、相機位置、螢幕距離

**L2CS-Net**
```python
# calibration.py - 校準過程
def _fit_models(self):
    # 只學習 gaze_angle → screen_position 的映射
    feature = [gaze_pitch, gaze_yaw, head_pitch, head_yaw]
    # 不直接學習眼睛外觀到螢幕的映射
```
- 校準只發生在**第二階段**
- CNN 模型是固定的，不針對用戶調整
- 累積誤差：CNN 預測誤差 + 多項式映射誤差

#### 3. 誤差累積問題

| 誤差來源 | WebGazer.js | L2CS-Net |
|---------|-------------|----------|
| 臉部偵測 | ~2-5px | ~2-5px |
| 眼部定位 | ~2-3px | ~2-3px |
| 視線估計 | → 直接到螢幕 | ~2-5° 角度誤差 |
| 角度→螢幕映射 | N/A | ~50-100px |
| **總誤差** | ~50-100px | ~100-200px |

#### 4. L2CS-Net 的設計目標不匹配

L2CS-Net 設計目標：
- 在 **Gaze360 數據集**上達到最低角度誤差
- 預測**絕對 3D 視線方向**
- 適用於：注意力分析、駕駛監控、VR/AR

但螢幕凝視追蹤需要：
- **相對於螢幕**的精確位置
- 用戶**個人化校準**
- 對小範圍移動的高敏感度

#### 5. 實時性能差異

**WebGazer.js**
- JavaScript 在瀏覽器中運行
- 利用 WebGL/GPU 加速
- 輕量級回歸模型，推理快速

**L2CS-Net**
```python
# gaze_estimator.py
self.estimator = L2CSPipeline(
    weights=weights_path,
    arch='ResNet50',  # 重量級模型
    device=torch_device,
    include_detector=True,  # RetinaFace 面部偵測
)
```
- ResNet50 + RetinaFace 組合**計算量大**
- 即使用 GPU，處理延遲也更高
- 延遲導致追蹤滯後，體感差

### 系統原理對比圖

#### WebGazer.js 架構
```
┌──────────────────────────────────────────────────┐
│                  用戶校準階段                      │
│  [點擊校準點] → 訓練 Ridge Regression 模型         │
│      (眼部像素特徵 → 螢幕座標直接映射)             │
└──────────────────────────────────────────────────┘
                      ↓ 使用階段
┌──────────────────────────────────────────────────┐
│  Webcam → 面部追蹤 → 眼部裁剪 → Ridge Reg → (x,y) │
│                                    ↑              │
│                          個人化訓練的模型          │
└──────────────────────────────────────────────────┘
```

#### L2CS-Net 架構
```
┌──────────────────────────────────────────────────┐
│              預訓練模型（固定不變）                 │
│  Gaze360 數據集 → 訓練 ResNet50 → 預測 3D 視線     │
└──────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────┐
│  Webcam → RetinaFace → 面部裁剪 → ResNet50        │
│                                      ↓            │
│                              (pitch, yaw) 角度    │
└──────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────┐
│              用戶校準階段（第二次映射）             │
│  [點擊校準點] → 訓練 Polynomial Regression        │
│      (gaze_angle, head_pose → 螢幕座標)          │
└──────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────┐
│  最終輸出：poly_reg(ResNet50(eye_image)) → (x,y)  │
│                 ↑                                │
│          兩階段映射，誤差累積                      │
└──────────────────────────────────────────────────┘
```

---

## 第二部分：ETH-XGaze 的相同問題

### ETH-XGaze 的使用方式

從 `gaze_estimator.py` 看 ETH-XGaze 的實現：

```python
# gaze_estimator.py lines 221-264
def _init_ptgaze(self):
    """Initialize ptgaze estimator."""
    # 使用 ptgaze 庫加載 ETH-XGaze 模型
    config_path = package_root / 'data' / 'configs' / 'eth-xgaze.yaml'
    self.estimator = PTGazeEstimator(config)
    
def _estimate_ptgaze(self, frame):
    # 輸出是 3D gaze_vector，不是螢幕座標
    gaze_vector = face.gaze_vector
    gaze_pitch, gaze_yaw = face.normalized_gaze_angles
```

**關鍵問題**：ETH-XGaze 輸出的是 **normalized 3D gaze vector**，不是螢幕座標。

### 訓練目標對比

| 模型 | 訓練目標 | 訓練數據 |
|------|---------|---------|
| **WebGazer.js** | 螢幕座標 (x, y) | 用戶點擊校準點時的眼部圖像 |
| **ETH-XGaze** | 3D 視線方向角度 | 80 人在不同頭部姿態下看 LED 陣列 |
| **L2CS-Net** | 3D 視線方向角度 | Gaze360 室內外環境 |

#### ETH-XGaze 數據集特點
- 用 **LED 陣列**作為注視目標（不是螢幕）
- 訓練目標是**歸一化的 3D 視線向量**
- 設計用於**跨人泛化**，而非個人精準

### 兩階段映射的致命缺陷

```
ETH-XGaze 管線：
┌─────────────────────────────────────────────────────────┐
│ 眼部圖像 → CNN → 3D gaze vector (pitch, yaw)            │
│                        ↓                                │
│              screen_mapper.py / calibration.py          │
│                        ↓                                │
│              螢幕座標 (x, y)                            │
└─────────────────────────────────────────────────────────┘

WebGazer.js 管線：
┌─────────────────────────────────────────────────────────┐
│ 眼部圖像 → Ridge Regression → 螢幕座標 (x, y)           │
│              (校準時直接學習)                            │
└─────────────────────────────────────────────────────────┘
```

### 誤差分析

**ETH-XGaze 報告精度**：~4-5° 角度誤差（在測試集上）

但這個角度誤差轉換到螢幕上是多少？

```python
# screen_mapper.py lines 48-83
def gaze_angles_to_screen(self, pitch, yaw, ...):
    # viewing_distance_px ≈ 60cm * 37.8 px/cm ≈ 2268 px
    dx = self.config.viewing_distance_px * np.tan(combined_yaw)
    dy = self.config.viewing_distance_px * np.tan(combined_pitch)
```

計算：
- 5° 角度誤差 → `tan(5°) × 2268px ≈ 198px`
- 這是**在校準之前**的理論誤差
- 校準後還要加上**多項式回歸的擬合誤差**

### 為什麼「更好的模型」反而更差？

#### 1. 泛化 vs 個人化的權衡

| | ETH-XGaze | WebGazer.js |
|--|-----------|-------------|
| 設計目標 | 跨人泛化（不需校準就能用） | 個人化（需要校準） |
| 實際效果 | 泛化能力好但精度有限 | 校準後精度極高 |

ETH-XGaze 犧牲了個人精度換取泛化能力，但在**固定用戶+固定設備**的場景下，這個權衡是錯的。

#### 2. 信息丟失

ETH-XGaze 的訓練過程：
```
原始眼部圖像 (豐富信息) 
    → 壓縮成 3D 角度 (2 個數字)
        → 再映射回螢幕座標
```

WebGazer.js：
```
原始眼部圖像 (豐富信息)
    → 直接映射到螢幕座標
        (保留了所有與螢幕位置相關的信息)
```

#### 3. 校準數據的利用效率

**WebGazer.js**：校準數據直接訓練回歸模型
- 每個校準點 = 一個訓練樣本
- 模型直接學習：這個眼睛外觀 → 這個螢幕位置

**ETH-XGaze**：校準數據只訓練二次映射
- CNN 是固定的，不會從校準中學習
- 只有多項式回歸從校準中學習
- 但多項式回歸的輸入是**已經有誤差的角度**

### 從代碼看證據

`calibration.py` 中的驗證函數：
```python
def validate_calibration(self, ...):
    # 在驗證點上測量誤差
    error = np.sqrt((mean_pred[0] - tx) ** 2 + (mean_pred[1] - ty) ** 2)
```

即使校準後，ETH-XGaze 管線的誤差仍然來自：
1. CNN 預測角度的誤差（~5°）
2. 多項式回歸無法完美擬合非線性映射
3. 頭部補償不完整（只用了 30%：`head_yaw * 0.3`）

---

## 第三部分：改進建議

### 如何改進 L2CS-Net / ETH-XGaze 系統

1. **端到端微調**
   - 凍結 ResNet50 前幾層，在最後增加螢幕座標預測頭
   - 用戶校準數據直接微調模型

2. **更好的校準方法**
   - 增加校準點數量（9 點 → 25 點）
   - 使用神經網絡替代多項式回歸
   - 考慮 Gaussian Process Regression

3. **降低延遲**
   - 使用 MobileNet 替代 ResNet50
   - 量化模型到 FP16/INT8
   - 使用 ONNX Runtime 加速

4. **考慮換用 WebGazer.js 架構**
   - 如果目標就是螢幕凝視追蹤，WebGazer 的設計哲學更適合
   - 或使用 GazeCapture 等專為螢幕設計的數據集重新訓練

5. **混合方法**
   - 用 L2CS/ETH-XGaze 做粗略估計
   - 用 Ridge Regression 做個人化微調

---

## 總結

### 問題與答案

| 問題 | 答案 |
|------|------|
| WebGazer 完全從頭訓練？ | **否**，面部偵測用 MediaPipe FaceMesh 預訓練模型 |
| 為何 WebGazer 更好？ | **校準直接作用於原始像素→螢幕的映射** |
| L2CS/ETH-XGaze 為何表現差？ | **校準只能修正「角度→螢幕」，無法修正 CNN 的角度預測誤差** |
| SOTA = 最好嗎？ | **不是。任務匹配比算法先進更重要** |

### 核心洞見（修正版）

> **WebGazer**：預訓練模型只負責「找到眼睛在哪」，校準學習「眼睛像素→螢幕座標」
> 
> **L2CS/ETH-XGaze**：預訓練模型負責「預測 3D 視線角度」，校準只學習「角度→螢幕座標」

前者讓校準有機會學習**完整的映射**，後者的校準只能做**誤差修正**，但 CNN 的系統性誤差難以用簡單多項式修正。

### 真正的差異不在「是否預訓練」，而在於：

#### 1. 映射的直接性

**WebGazer.js**：
```
眼部像素 (120 維) → [Ridge Regression] → 螢幕座標 (x, y)
                     ↑
              校準時直接學習這個映射
```

**L2CS-Net**：
```
眼部像素 → [預訓練 ResNet50] → 3D 角度 (pitch, yaw)
                                      ↓
                              [Polynomial Regression] → 螢幕座標
                                      ↑
                              校準只學習這一段
```

#### 2. 特徵的「純度」

| | WebGazer.js | L2CS-Net |
|--|-------------|----------|
| 輸入到校準模型 | 原始眼部像素 (與螢幕位置直接相關) | 預測的 3D 角度 (已有誤差) |
| 校準學習的難度 | 簡單線性關係 | 需要修正 CNN 的預測誤差 |

**關鍵洞見**：
- WebGazer 的 Ridge Regression 輸入是**原始信號**
- L2CS 的 Polynomial Regression 輸入是**已經被 CNN 處理過且有誤差的信號**

#### 3. 即時再訓練

WebGazer 會**持續收集數據並重新訓練**（每 500ms），包括：
- 用戶點擊事件 (50 個窗口)
- 滑鼠移動軌跡 (10 個窗口，1 秒內)

L2CS/ETH-XGaze 的校準是**一次性**的，完成後模型就固定了。

### 一句話總結

> ETH-XGaze / L2CS-Net 學的是「眼睛看向空間哪個方向」，WebGazer 學的是「眼睛看螢幕哪個位置」。對於螢幕追蹤，後者是**直接解決問題**，前者是**繞了一圈**。

---

## 相關文件參考

- `gaze_heatmap/core/gaze_estimator.py` - ML 模型接口
- `gaze_heatmap/core/calibration.py` - 校準系統實現
- `gaze_heatmap/core/screen_mapper.py` - 螢幕映射邏輯
- `gaze_heatmap/core/smoother.py` - 平滑濾波器

---

## 附錄：WebGazer.js 技術細節

### Ridge Regression 數學公式

```
係數 = (X^T X + kI)^-1 * X^T y

其中：
- X: 眼部像素特徵矩陣 (n_samples × 120)
- y: 螢幕座標 (X 或 Y)
- k: 正則化參數 (10^-5)
- I: 單位矩陣
```

### 關鍵參數

| 參數 | 值 | 說明 |
|------|-----|------|
| Ridge 正則化 (k) | 10^-5 | 防止過擬合 |
| 點擊數據窗口 | 50 | 最近 50 次點擊 |
| 軌跡數據窗口 | 10 | 最近 1 秒內的滑鼠移動 |
| 再訓練間隔 | 500ms | 持續適應用戶 |
| 眼部圖像大小 | 10×6 | 每眼 60 像素 |
| Kalman 測量噪聲 | 47 px | 平滑預測 |

### 眼部 Landmark 索引 (MediaPipe FaceMesh)

```javascript
const EYE_INDICES = {
  leftEyeUpper0: [466, 388, 387, 386, 385, 384, 398],
  leftEyeLower0: [263, 249, 390, 373, 374, 380, 381, 382, 362],
  rightEyeUpper0: [246, 161, 160, 159, 158, 157, 173],
  rightEyeLower0: [33, 7, 163, 144, 145, 153, 154, 155, 133],
};
```

---

*分析日期：2026-01-21*
*更新：加入 WebGazer.js 真實架構澄清*
