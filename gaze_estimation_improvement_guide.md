# Webcam Gaze Estimation 改進指南

ETH-XGaze 模型在 benchmark 上表現優異（~3.5°），但實際 webcam 部署時常遭遇 **domain gap** 問題——模型在實驗室條件下訓練，部署到不同光線、相機、使用者時性能下降 40-60%。以下是系統性的改進策略。

---

## 🎯 快速診斷：你的問題在哪裡？

| 症狀 | 可能原因 | 優先解決方案 |
|------|----------|--------------|
| 所有使用者誤差都很大 | Domain gap（光線/相機） | Section 1, 2 |
| 特定使用者誤差大 | 個人差異（眼睛形狀） | Section 3 |
| 頭部轉動時誤差增大 | Head pose 補償不足 | Section 4 |
| 校準後仍不準 | 校準點數不夠/方法錯誤 | Section 5 |
| 輸出抖動嚴重 | Smoothing 不足 | Section 6 |

---

## 1. 嘗試其他 SoTA 模型

ETH-XGaze baseline 不一定是最佳選擇。以下模型可能在你的場景表現更好：

### 1.1 L2CS-Net（推薦首選）

**優勢**：分離 pitch/yaw 預測，對 in-the-wild 場景更穩健

```bash
pip install l2cs
```

```python
from l2cs import Pipeline
import torch

gaze_pipeline = Pipeline(
    weights='models/L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

results = gaze_pipeline.step(frame)
pitch, yaw = results.pitch, results.yaw  # 直接輸出角度
```

**Benchmark**：
- MPIIGaze: 3.92°
- Gaze360: 10.41°

**GitHub**: https://github.com/Ahmednull/L2CS-Net

### 1.2 GazeTR（Transformer-based）

**優勢**：Hybrid CNN-Transformer，捕捉長距離依賴，對極端頭部姿態更穩健

```python
# 下載預訓練模型：ETH-XGaze 上訓練
# https://drive.google.com/drive/folders/1Hx8Ux8KDfJ2RnJGO0Q...

from model import Model
GazeTR = Model()
img = {'face': face_tensor}  # (B, 3, 224, 224)
gaze = GazeTR(img)  # (B, 2) pitch, yaw
```

**Benchmark**：
- MPIIGaze: ~4.0°
- 對極端頭部姿態表現優於 CNN-only 方法

**GitHub**: https://github.com/yihuacheng/GazeTR

### 1.3 模型比較表

| 模型 | MPIIGaze | Gaze360 | 優勢 | 劣勢 |
|------|----------|---------|------|------|
| ETH-XGaze baseline | 4.5° | - | 大規模訓練 | Domain gap 嚴重 |
| L2CS-Net | 3.92° | 10.41° | 分離角度預測 | 需要 Gaze360 權重 |
| GazeTR | ~4.0° | - | 極端姿態好 | 計算量較大 |
| RT-GENE | 4.3° | - | 自然環境 | 需要 RGB-D |

---

## 2. Domain Adaptation（關鍵！）

這是解決 webcam 實際部署問題的核心。

### 2.1 PnP-GA+（Plug-and-Play，推薦）

**原理**：利用模型變體的多樣性，無需 target domain 標註資料

```python
# PnP-GA+ 核心思想：組合多個模型變體的預測
# 1. Color space variants (RGB, YUV, HSV)
# 2. Data augmentation variants
# 3. Model structure variants

class PnPGAPlus:
    def __init__(self, base_models):
        self.models = base_models  # 多個變體模型
        self.attention = IntraGroupAttention()
    
    def predict(self, x):
        preds = [m(x) for m in self.models]
        weights = self.attention(preds)
        return weighted_average(preds, weights)
```

**Paper**: "PnP-GA+: Plug-and-Play Domain Adaptation for Gaze Estimation" (TPAMI 2024)

### 2.2 簡易 Domain Adaptation：Test-Time Adaptation

```python
import torch.nn as nn

class TestTimeAdaptation:
    """
    無需標註資料的 self-training 方法
    """
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr
        )
    
    def adapt(self, unlabeled_frames, num_iterations=100):
        """
        使用 entropy minimization 進行 self-training
        """
        self.model.train()
        
        for _ in range(num_iterations):
            for frame in unlabeled_frames:
                # Forward pass
                gaze_pred = self.model(frame)
                
                # Entropy minimization loss
                # 鼓勵模型產生更確定的預測
                entropy = -torch.sum(
                    F.softmax(gaze_pred, dim=-1) * 
                    F.log_softmax(gaze_pred, dim=-1)
                )
                
                # Update
                self.optimizer.zero_grad()
                entropy.backward()
                self.optimizer.step()
```

### 2.3 Feature Normalization

```python
def normalize_features_for_domain(features, target_mean, target_std):
    """
    簡單但有效的 feature normalization
    將你的 domain 的 feature 分布對齊到訓練資料的分布
    """
    # 計算當前 batch 的統計量
    current_mean = features.mean(dim=0, keepdim=True)
    current_std = features.std(dim=0, keepdim=True)
    
    # Normalize then denormalize to target distribution
    normalized = (features - current_mean) / (current_std + 1e-6)
    adapted = normalized * target_std + target_mean
    
    return adapted
```

---

## 3. 個人化校準（最有效的改進方式）

### 3.1 FAZE: Few-Shot Adaptive Gaze Estimation

**核心思想**：Meta-learning (MAML) 讓模型能用 3-9 個校準點快速適應新使用者

```python
class FAZEPersonalization:
    """
    基於 MAML 的 few-shot 個人化
    """
    def __init__(self, base_model, inner_lr=0.01, outer_lr=0.001):
        self.model = base_model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
    
    def personalize(self, calibration_samples, num_steps=5):
        """
        calibration_samples: [(image, gaze_label), ...] 
        只需要 3-9 個樣本
        """
        # Clone model for personalization
        personal_model = copy.deepcopy(self.model)
        
        for step in range(num_steps):
            for img, label in calibration_samples:
                pred = personal_model(img)
                loss = F.mse_loss(pred, label)
                
                # Inner loop gradient update
                grads = torch.autograd.grad(
                    loss, personal_model.parameters()
                )
                
                for param, grad in zip(personal_model.parameters(), grads):
                    param.data -= self.inner_lr * grad
        
        return personal_model
```

**效果**：
- 3 個校準點：誤差降低 ~19%
- 9 個校準點：誤差降低 ~30%

**Paper**: "Few-Shot Adaptive Gaze Estimation" (ICCV 2019, NVIDIA)

### 3.2 改進的校準程序

```python
class ImprovedCalibration:
    def __init__(self, num_points=13):
        self.num_points = num_points
        # 13-point pattern: 9 grid + 4 intermediate
        self.calibration_points = self._generate_points()
    
    def _generate_points(self):
        """
        擴展到 13 點：3x3 網格 + 4 個中間點
        """
        points = []
        margin = 0.1
        
        # 3x3 grid
        for i in range(3):
            for j in range(3):
                x = margin + (1 - 2*margin) * j / 2
                y = margin + (1 - 2*margin) * i / 2
                points.append((x, y))
        
        # 4 intermediate points (提高精度)
        points.extend([
            (0.25, 0.25), (0.75, 0.25),
            (0.25, 0.75), (0.75, 0.75)
        ])
        
        return points
    
    def fit_personalized_mapping(self, 
                                  gaze_vectors, 
                                  screen_coords,
                                  degree=3):
        """
        使用 3 階多項式 + Ridge Regression
        """
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        
        # 構建特徵：[gaze_pitch, gaze_yaw, head_pitch, head_yaw]
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(gaze_vectors)
        
        # Ridge regression 防止 overfitting
        self.model_x = Ridge(alpha=1.0).fit(X_poly, screen_coords[:, 0])
        self.model_y = Ridge(alpha=1.0).fit(X_poly, screen_coords[:, 1])
        
        self.poly = poly
        
        return self
    
    def predict(self, gaze_vector):
        X_poly = self.poly.transform(gaze_vector.reshape(1, -1))
        x = self.model_x.predict(X_poly)[0]
        y = self.model_y.predict(X_poly)[0]
        return x, y
```

### 3.3 SVR-based Calibration（更穩健）

```python
from sklearn.svm import SVR

class SVRCalibration:
    """
    SVR 對 outliers 更穩健
    """
    def __init__(self, kernel='rbf', C=10, epsilon=0.1):
        self.svr_x = SVR(kernel=kernel, C=C, epsilon=epsilon)
        self.svr_y = SVR(kernel=kernel, C=C, epsilon=epsilon)
    
    def fit(self, features, screen_coords):
        """
        features: (N, 4) - [gaze_pitch, gaze_yaw, head_pitch, head_yaw]
        screen_coords: (N, 2) - [screen_x, screen_y]
        """
        self.svr_x.fit(features, screen_coords[:, 0])
        self.svr_y.fit(features, screen_coords[:, 1])
        return self
    
    def predict(self, features):
        x = self.svr_x.predict(features)
        y = self.svr_y.predict(features)
        return np.column_stack([x, y])
```

---

## 4. Head Pose Compensation

頭部姿態變化是誤差的主要來源之一。

### 4.1 Gaze Decomposition Method

```python
class GazeDecomposition:
    """
    將 gaze 分解為：
    1. Subject-independent gaze（從圖像估計）
    2. Subject-dependent bias（從校準學習）
    3. Head-pose compensation
    
    Paper: "Offset Calibration for Appearance-Based Gaze Estimation"
    """
    def __init__(self, base_model):
        self.base_model = base_model
        self.subject_bias = None
        self.head_pose_compensation = None
    
    def calibrate(self, calibration_data):
        """
        calibration_data: [(image, head_pose, true_gaze), ...]
        """
        biases = []
        
        for img, head_pose, true_gaze in calibration_data:
            pred_gaze = self.base_model(img)
            
            # 計算 bias = true - predicted
            bias = true_gaze - pred_gaze
            biases.append(bias)
        
        # Subject-dependent bias
        self.subject_bias = np.mean(biases, axis=0)
        
        # 學習 head pose -> bias 的映射
        head_poses = np.array([d[1] for d in calibration_data])
        biases = np.array(biases)
        
        from sklearn.linear_model import LinearRegression
        self.head_pose_compensation = LinearRegression()
        self.head_pose_compensation.fit(head_poses, biases)
    
    def predict(self, image, head_pose):
        # Base prediction
        gaze = self.base_model(image)
        
        # Add subject bias
        gaze += self.subject_bias
        
        # Head pose compensation
        compensation = self.head_pose_compensation.predict(
            head_pose.reshape(1, -1)
        )[0]
        gaze += compensation
        
        return gaze
```

### 4.2 Head Pose Normalization

```python
def normalize_head_pose(image, landmarks, target_pose=(0, 0, 0)):
    """
    將圖像 warp 到標準頭部姿態
    減少 head pose variation 對 gaze 估計的影響
    """
    import cv2
    
    # 估計 3D head pose
    # 使用 solvePnP
    model_points = get_3d_face_model()
    camera_matrix = get_camera_matrix(image.shape)
    
    _, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, landmarks, camera_matrix, None
    )
    
    # 計算 rotation matrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    
    # Warp image to frontal view
    # ... (詳細實現見 data normalization 文獻)
    
    return normalized_image, inverse_transform
```

---

## 5. 資料增強與預處理

### 5.1 Gaze-Aware Data Augmentation

```python
class GazeAwareAugmentation:
    """
    專為 gaze estimation 設計的增強策略
    關鍵：增強時要同步調整 gaze label
    """
    def __init__(self):
        self.transforms = []
    
    def horizontal_flip(self, image, gaze):
        """
        水平翻轉：yaw 要取反
        """
        flipped = cv2.flip(image, 1)
        gaze_flipped = gaze.copy()
        gaze_flipped[1] = -gaze_flipped[1]  # yaw
        return flipped, gaze_flipped
    
    def brightness_adjustment(self, image, gaze, factor_range=(0.7, 1.3)):
        """
        亮度調整：gaze 不變
        """
        factor = np.random.uniform(*factor_range)
        adjusted = np.clip(image * factor, 0, 255).astype(np.uint8)
        return adjusted, gaze
    
    def gaussian_noise(self, image, gaze, std=10):
        """
        加噪：gaze 不變
        """
        noise = np.random.normal(0, std, image.shape).astype(np.uint8)
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy, gaze
    
    def color_jitter(self, image, gaze, 
                     hue_shift=10, sat_scale=0.2, val_scale=0.2):
        """
        色彩抖動：模擬不同光線條件
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        hsv[:,:,0] += np.random.uniform(-hue_shift, hue_shift)
        hsv[:,:,1] *= np.random.uniform(1-sat_scale, 1+sat_scale)
        hsv[:,:,2] *= np.random.uniform(1-val_scale, 1+val_scale)
        
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return augmented, gaze
```

### 5.2 Eye Region Enhancement

```python
def enhance_eye_region(image, landmarks, clahe_clip=2.0):
    """
    增強眼睛區域的對比度
    CLAHE 對低光線環境特別有效
    """
    # 提取眼睛 ROI
    left_eye_indices = [36, 37, 38, 39, 40, 41]
    right_eye_indices = [42, 43, 44, 45, 46, 47]
    
    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    
    # 轉換到 LAB 色彩空間
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced
```

---

## 6. Temporal Smoothing 改進

### 6.1 1€ Filter（推薦）

```python
import math

class OneEuroFilter:
    """
    1€ Filter: 自適應低通濾波器
    - 靜止時高度平滑
    - 快速移動時保持響應
    """
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
    
    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x
        
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        
        # Derivative
        dx = (x - self.x_prev) / dt
        
        # Smoothed derivative
        alpha_d = self._smoothing_factor(dt, self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        
        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Smoothed value
        alpha = self._smoothing_factor(dt, cutoff)
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        
        # Store
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat
    
    def _smoothing_factor(self, dt, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
```

### 6.2 Fixation-Aware Smoothing

```python
class FixationAwareSmoother:
    """
    根據是否在 fixation 期間調整 smoothing 強度
    """
    def __init__(self, 
                 fixation_velocity_threshold=100,  # px/s
                 fixation_duration_threshold=100): # ms
        self.velocity_threshold = fixation_velocity_threshold
        self.duration_threshold = fixation_duration_threshold
        
        self.filter_fixation = OneEuroFilter(min_cutoff=0.5, beta=0.001)
        self.filter_saccade = OneEuroFilter(min_cutoff=2.0, beta=0.05)
        
        self.positions = []
        self.timestamps = []
    
    def update(self, x, y, timestamp):
        self.positions.append((x, y))
        self.timestamps.append(timestamp)
        
        # 計算速度
        if len(self.positions) >= 2:
            dt = self.timestamps[-1] - self.timestamps[-2]
            if dt > 0:
                dx = self.positions[-1][0] - self.positions[-2][0]
                dy = self.positions[-1][1] - self.positions[-2][1]
                velocity = math.sqrt(dx**2 + dy**2) / dt * 1000  # px/s
            else:
                velocity = 0
        else:
            velocity = 0
        
        # 選擇濾波器
        if velocity < self.velocity_threshold:
            # Fixation: 更強的平滑
            x_smooth = self.filter_fixation(x, timestamp)
            y_smooth = self.filter_fixation(y, timestamp)
            is_fixation = True
        else:
            # Saccade: 較弱的平滑，保持響應
            x_smooth = self.filter_saccade(x, timestamp)
            y_smooth = self.filter_saccade(y, timestamp)
            is_fixation = False
        
        return x_smooth, y_smooth, is_fixation
```

---

## 7. 環境優化 Checklist

### 7.1 硬體設置

```yaml
camera:
  resolution: 1280x720  # 至少 720p
  fps: 30              # 穩定的 frame rate
  position: below_screen_center  # 或螢幕上方正中央
  angle: minimize_vertical_offset  # 減少俯視/仰視角度

lighting:
  type: diffuse        # 避免直射光
  position: front      # 正面打光
  intensity: uniform   # 均勻分布
  avoid:
    - side_lighting    # 會造成虹膜反射
    - backlighting     # 會造成瞳孔難以偵測
    - flickering       # 會造成不穩定

user_position:
  distance_cm: 50-70   # 最佳距離
  head_pose: frontal   # 盡量正對螢幕
  glasses: position_camera_lower  # 戴眼鏡者將相機放低
```

### 7.2 軟體配置

```python
CONFIG = {
    # 預處理
    'face_detection': 'mediapipe',  # 比 dlib 快 3x
    'eye_enhancement': True,
    'clahe_clip_limit': 2.0,
    
    # 模型
    'gaze_model': 'L2CS-Net',  # 或 GazeTR
    'model_weights': 'gaze360',  # gaze360 通常比 mpiigaze 更穩健
    
    # 校準
    'calibration_points': 13,  # 9 + 4 intermediate
    'calibration_samples_per_point': 30,  # 每點收集 30 frame
    'calibration_regression': 'svr',  # SVR 比多項式穩健
    
    # 濾波
    'smoother': 'one_euro',
    'min_cutoff': 1.0,
    'beta': 0.007,
    
    # 閾值
    'confidence_threshold': 0.7,
    'face_not_detected_timeout_ms': 500,
}
```

---

## 8. 進階：Fine-tuning on Your Data

如果你有標註資料，可以在你的 domain 上 fine-tune。

### 8.1 收集自己的資料

```python
class GazeDataCollector:
    """
    收集自己環境下的校準資料用於 fine-tuning
    """
    def __init__(self, save_dir='./my_gaze_data'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def collect_session(self, num_targets=50, duration_per_target=2.0):
        """
        顯示 50 個隨機目標點，每個注視 2 秒
        """
        import pyautogui
        
        screen_w, screen_h = pyautogui.size()
        data = []
        
        cap = cv2.VideoCapture(0)
        
        for i in range(num_targets):
            # 隨機目標位置
            target_x = random.randint(100, screen_w - 100)
            target_y = random.randint(100, screen_h - 100)
            
            # 顯示目標
            show_target(target_x, target_y)
            
            # 收集 frames
            start_time = time.time()
            frames = []
            
            while time.time() - start_time < duration_per_target:
                ret, frame = cap.read()
                if ret:
                    frames.append({
                        'image': frame,
                        'timestamp': time.time(),
                        'target': (target_x, target_y)
                    })
            
            # 只保留後半段（穩定注視期間）
            stable_frames = frames[len(frames)//2:]
            data.extend(stable_frames)
        
        # 保存
        self.save_data(data)
        return data
```

### 8.2 Fine-tuning Script

```python
def finetune_gaze_model(
    base_model,
    train_data,
    val_data,
    epochs=20,
    lr=1e-5,
    freeze_backbone=True
):
    """
    在自己的資料上 fine-tune
    """
    # 凍結 backbone，只訓練最後幾層
    if freeze_backbone:
        for name, param in base_model.named_parameters():
            if 'fc' not in name and 'head' not in name:
                param.requires_grad = False
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, base_model.parameters()),
        lr=lr,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    best_error = float('inf')
    
    for epoch in range(epochs):
        # Training
        base_model.train()
        train_loss = 0
        
        for batch in train_data:
            images, labels = batch
            
            preds = base_model(images)
            loss = F.mse_loss(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        base_model.eval()
        val_errors = []
        
        with torch.no_grad():
            for batch in val_data:
                images, labels = batch
                preds = base_model(images)
                
                # Angular error
                error = angular_error(preds, labels)
                val_errors.append(error)
        
        mean_error = np.mean(val_errors)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, "
              f"Val Angular Error = {mean_error:.2f}°")
        
        if mean_error < best_error:
            best_error = mean_error
            torch.save(base_model.state_dict(), 'best_finetuned.pth')
        
        scheduler.step()
    
    return base_model
```

---

## 9. 建議的改進優先順序

### Quick Wins（立即可做）

1. **換模型**：ETH-XGaze → L2CS-Net（Gaze360 權重）
2. **增加校準點**：9 → 13 點
3. **使用 SVR 而非多項式回歸**
4. **加入 1€ Filter**

### Medium Effort（一週內）

5. **實作 Head Pose Compensation**
6. **收集自己環境的校準資料**
7. **Fine-tune 最後幾層**

### High Effort（需要更多資源）

8. **實作 FAZE few-shot adaptation**
9. **Domain adaptation training**
10. **收集大規模自己環境的資料重新訓練**

---

## 10. 預期改進幅度

| 改進措施 | 預期誤差降低 | 實作難度 |
|----------|--------------|----------|
| 換 L2CS-Net | 10-20% | ⭐ |
| 增加校準點 | 5-10% | ⭐ |
| SVR 校準 | 5-15% | ⭐⭐ |
| 1€ Filter | 減少抖動 50%+ | ⭐ |
| Head Pose Compensation | 15-25% | ⭐⭐⭐ |
| FAZE few-shot | 20-30% | ⭐⭐⭐⭐ |
| Domain fine-tuning | 30-50% | ⭐⭐⭐⭐ |

**綜合以上改進**，從原始 5-6° 誤差可望降至 **2-3°**，接近可用於實際 heatmap 應用的水平。

---

## 參考文獻

1. L2CS-Net (ICFSP 2023): https://github.com/Ahmednull/L2CS-Net
2. GazeTR (ICPR 2022): https://github.com/yihuacheng/GazeTR
3. FAZE (ICCV 2019): Few-Shot Adaptive Gaze Estimation
4. PnP-GA+ (TPAMI 2024): Plug-and-Play Domain Adaptation
5. Gaze-BAR (2024): Branch-out Auxiliary Regularization
6. 1€ Filter: https://cristal.univ-lille.fr/~casiez/1euro/
