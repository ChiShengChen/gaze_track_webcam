# Cursor Project Prompt: Webcam Gaze Heatmap Application

## Project Overview

Build a **real-time webcam-based gaze estimation system** that generates screen attention heatmaps. The application captures where users look on the screen using a standard webcam, performs calibration, and produces visual heatmaps for UX analysis. Include an evaluation pipeline with angular error metrics for ground-truth comparison.

---

## Tech Stack

```
Python 3.10+
├── ptgaze              # Gaze estimation (ETH-XGaze model)
├── mediapipe           # Face Mesh (468 landmarks, iris tracking)
├── opencv-python       # Video capture & visualization
├── numpy               # Numerical operations
├── scipy               # Polynomial regression, Gaussian filters
├── filterpy            # Kalman filter implementation
├── matplotlib          # Heatmap visualization
├── screeninfo          # Multi-monitor support
└── pyyaml              # Configuration management
```

---

## Core Architecture

```
gaze_heatmap/
├── main.py                    # Entry point with CLI
├── config.yaml                # Configuration file
├── core/
│   ├── __init__.py
│   ├── gaze_estimator.py      # ptgaze wrapper
│   ├── face_detector.py       # MediaPipe Face Mesh
│   ├── calibration.py         # 9-point calibration + polynomial regression
│   ├── smoother.py            # Kalman & 1€ filter implementations
│   └── screen_mapper.py       # Gaze vector → screen coordinate projection
├── heatmap/
│   ├── __init__.py
│   ├── accumulator.py         # Gaussian-weighted fixation aggregation
│   ├── renderer.py            # Heatmap overlay rendering
│   └── exporter.py            # Save heatmaps (PNG, NPY, JSON metadata)
├── evaluation/
│   ├── __init__.py
│   ├── error_metrics.py       # Angular error calculation
│   ├── labeling_tool.py       # Manual ground-truth annotation UI
│   └── benchmark.py           # Evaluation pipeline
├── utils/
│   ├── __init__.py
│   ├── geometry.py            # 3D→2D projection, angle calculations
│   └── logger.py              # Structured logging
└── data/
    ├── calibrations/          # Saved calibration profiles
    ├── sessions/              # Raw gaze data per session
    └── heatmaps/              # Exported heatmaps for labeling
```

---

## Module Specifications

### 1. Gaze Estimator (`core/gaze_estimator.py`)

```python
"""
Wrapper for ptgaze library with ETH-XGaze model.

Requirements:
- Initialize ptgaze with mode='eth-xgaze' for best accuracy
- Accept BGR frame input from OpenCV
- Return: 
  - gaze_vector: np.ndarray (3,) - normalized 3D gaze direction
  - head_pose: np.ndarray (3,) - roll, pitch, yaw in radians
  - landmarks: np.ndarray (468, 2) - MediaPipe face landmarks
  - confidence: float - detection confidence score
- Handle face not detected gracefully (return None)
- Support both CPU and CUDA backends
"""

class GazeEstimator:
    def __init__(self, device: str = 'cpu', model: str = 'eth-xgaze'):
        pass
    
    def estimate(self, frame: np.ndarray) -> Optional[GazeResult]:
        """Process single frame, return gaze data or None if no face."""
        pass
```

### 2. Calibration System (`core/calibration.py`)

```python
"""
9-point calibration with polynomial regression for gaze-to-screen mapping.

Calibration Flow:
1. Display calibration points sequentially (3x3 grid + center)
2. For each point:
   - Show target dot with shrinking animation (1.5s)
   - Collect gaze vectors during final 0.5s (stable fixation)
   - Store: (screen_x, screen_y, mean_gaze_vector, head_pose)
3. Fit 2nd-degree polynomial regression:
   - screen_x = f(gaze_pitch, gaze_yaw, head_pitch, head_yaw)
   - screen_y = g(gaze_pitch, gaze_yaw, head_pitch, head_yaw)
4. Save calibration profile with timestamp

Calibration Point Layout (normalized 0-1):
    (0.1, 0.1)  (0.5, 0.1)  (0.9, 0.1)
    (0.1, 0.5)  (0.5, 0.5)  (0.9, 0.5)
    (0.1, 0.9)  (0.5, 0.9)  (0.9, 0.9)

Requirements:
- Support re-calibration of individual points
- Calculate and display calibration quality (mean error on validation points)
- Save/load calibration profiles (YAML format)
- Polynomial features: [1, p, y, p², y², p*y, hp, hy, hp², hy²]
  where p=gaze_pitch, y=gaze_yaw, hp=head_pitch, hy=head_yaw
"""

class CalibrationSystem:
    def __init__(self, screen_width: int, screen_height: int):
        pass
    
    def run_calibration(self, gaze_estimator: GazeEstimator) -> CalibrationProfile:
        """Interactive calibration procedure."""
        pass
    
    def map_gaze_to_screen(self, gaze_vector: np.ndarray, head_pose: np.ndarray) -> Tuple[int, int]:
        """Apply calibration to get screen coordinates."""
        pass
    
    def save_profile(self, path: str):
        pass
    
    def load_profile(self, path: str):
        pass
```

### 3. Temporal Smoothing (`core/smoother.py`)

```python
"""
Implement both Kalman filter and 1€ filter for gaze smoothing.

Kalman Filter Configuration:
- State: [x, y, vx, vy] (position + velocity)
- Measurement: [x, y] (screen coordinates)
- Process noise: Q = diag([1, 1, 10, 10]) (tunable)
- Measurement noise: R = diag([50, 50]) (tunable based on calibration error)

1€ Filter Configuration:
- min_cutoff: 1.0 Hz (smoothness vs responsiveness tradeoff)
- beta: 0.007 (speed coefficient)
- d_cutoff: 1.0 Hz (derivative cutoff)

Requirements:
- Abstract base class for filter interface
- Both filters should handle:
  - Initialization on first sample
  - Reset on face lost (gap > 500ms)
  - Return smoothed (x, y) coordinates
"""

class GazeSmoother(ABC):
    @abstractmethod
    def update(self, x: float, y: float, timestamp: float) -> Tuple[float, float]:
        pass
    
    @abstractmethod
    def reset(self):
        pass

class KalmanGazeSmoother(GazeSmoother):
    pass

class OneEuroGazeSmoother(GazeSmoother):
    pass
```

### 4. Heatmap Accumulator (`heatmap/accumulator.py`)

```python
"""
Gaussian-weighted fixation aggregation for heatmap generation.

Algorithm:
1. Maintain 2D accumulator array matching screen resolution (or downsampled)
2. For each gaze point (x, y):
   - Add Gaussian blob centered at (x, y)
   - Gaussian sigma based on estimated uncertainty (~50-100 pixels)
   - Weight by fixation duration (points during saccades get lower weight)
3. Fixation detection:
   - Velocity-based: if speed < 100 px/s for > 100ms → fixation
   - Only accumulate during fixations

Heatmap Resolution Strategy:
- Internal: screen_width/4 × screen_height/4 (performance)
- Export: full resolution with interpolation

Requirements:
- Real-time update capability (< 5ms per frame)
- Configurable Gaussian sigma and fixation thresholds
- Temporal decay option (recent fixations weighted more)
- Export raw accumulator as .npy for evaluation
"""

class HeatmapAccumulator:
    def __init__(self, width: int, height: int, 
                 sigma: float = 50.0, 
                 downsample: int = 4,
                 decay_rate: float = 0.0):
        pass
    
    def add_gaze_point(self, x: float, y: float, 
                       timestamp: float, 
                       is_fixation: bool = True):
        """Add single gaze point with Gaussian weighting."""
        pass
    
    def get_heatmap(self, normalize: bool = True) -> np.ndarray:
        """Return current heatmap (HxW float array, 0-1 if normalized)."""
        pass
    
    def reset(self):
        """Clear accumulator for new session."""
        pass
    
    def save(self, path: str, include_metadata: bool = True):
        """Save heatmap as .npy with JSON metadata sidecar."""
        pass
```

### 5. Angular Error Metrics (`evaluation/error_metrics.py`)

```python
"""
Calculate angular error between predicted gaze and ground truth.

Error Metrics:
1. Angular Error (degrees):
   θ = arccos(g_pred · g_true / (|g_pred| × |g_true|))
   
2. Screen-Space Error (pixels/cm):
   d = sqrt((x_pred - x_true)² + (y_pred - y_true)²)
   
3. Precision (within-subject std of errors)

4. Accuracy (mean error across subjects)

Ground Truth Collection:
- User clicks on displayed target points
- Store: timestamp, target_screen_pos, predicted_gaze_pos, raw_gaze_vector
- Minimum 20 points across screen regions for reliable metrics

Requirements:
- Support both online (during session) and offline (post-hoc) evaluation
- Calculate per-region error (9 screen regions matching calibration grid)
- Generate error report with visualizations
"""

@dataclass
class EvaluationResult:
    mean_angular_error: float      # degrees
    std_angular_error: float
    mean_screen_error_px: float    # pixels
    mean_screen_error_cm: float    # requires monitor DPI
    precision: float               # within-subject std
    per_region_errors: Dict[str, float]  # 'top-left', 'center', etc.
    raw_data: pd.DataFrame         # all individual measurements

class ErrorMetricsCalculator:
    def __init__(self, screen_width: int, screen_height: int, 
                 monitor_dpi: float = 96.0,
                 viewing_distance_cm: float = 60.0):
        pass
    
    def add_measurement(self, 
                        target_screen: Tuple[int, int],
                        predicted_screen: Tuple[int, int],
                        gaze_vector: np.ndarray,
                        timestamp: float):
        """Record single ground truth measurement."""
        pass
    
    def calculate_angular_error(self, 
                                 target_screen: Tuple[int, int],
                                 predicted_screen: Tuple[int, int]) -> float:
        """
        Convert screen positions to angles and compute error.
        
        Angle calculation:
        - Assume eye at (screen_center_x, screen_center_y, -viewing_distance)
        - Target angle: atan2(target - screen_center, viewing_distance)
        - Error = angle between target and predicted directions
        """
        pass
    
    def compute_results(self) -> EvaluationResult:
        """Calculate all metrics from collected measurements."""
        pass
    
    def export_report(self, path: str):
        """Generate PDF/HTML report with error visualizations."""
        pass
```

### 6. Labeling Tool (`evaluation/labeling_tool.py`)

```python
"""
Manual ground-truth annotation interface for heatmap evaluation.

Features:
1. Load saved heatmap + session metadata
2. Display heatmap overlay on screenshot/stimulus
3. User annotates:
   - Expected attention regions (draw polygons/rectangles)
   - Attention ranking (which area should have most attention)
   - Binary labels: correct/incorrect for each AOI (Area of Interest)
4. Save annotations in standardized format

UI Components:
- Side-by-side: original stimulus | recorded heatmap
- Drawing tools: rectangle, polygon, point
- AOI list with attention scores
- Navigation: prev/next session

Export Format (JSON):
{
    "session_id": "...",
    "stimulus": "screenshot.png",
    "heatmap_path": "session_001_heatmap.npy",
    "annotations": [
        {
            "aoi_id": "button_cta",
            "type": "rectangle",
            "coords": [x1, y1, x2, y2],
            "expected_attention": 0.8,  # 0-1 scale
            "actual_attention": 0.65,   # computed from heatmap
            "correct": true
        }
    ],
    "overall_score": 0.78,
    "annotator": "user_id",
    "timestamp": "..."
}
"""

class LabelingTool:
    def __init__(self, sessions_dir: str, output_dir: str):
        pass
    
    def load_session(self, session_id: str):
        """Load heatmap and metadata for annotation."""
        pass
    
    def run_ui(self):
        """Launch annotation interface (tkinter or web-based)."""
        pass
    
    def compute_aoi_attention(self, heatmap: np.ndarray, 
                               aoi_coords: List[Tuple]) -> float:
        """Calculate attention score within AOI from heatmap."""
        pass
    
    def export_annotations(self, path: str):
        pass
```

---

## Configuration (`config.yaml`)

```yaml
# Gaze Estimation
gaze:
  model: "eth-xgaze"  # eth-xgaze | mpiigaze | mpiifacegaze
  device: "cpu"       # cpu | cuda
  camera_id: 0
  frame_width: 1280
  frame_height: 720
  fps: 30

# Calibration
calibration:
  num_points: 9
  point_duration_ms: 1500
  collection_window_ms: 500
  polynomial_degree: 2
  margin: 0.1  # screen edge margin (0-0.5)

# Smoothing
smoothing:
  method: "kalman"  # kalman | one_euro | none
  kalman:
    process_noise: [1, 1, 10, 10]
    measurement_noise: [50, 50]
  one_euro:
    min_cutoff: 1.0
    beta: 0.007
    d_cutoff: 1.0

# Heatmap
heatmap:
  sigma: 50.0           # Gaussian sigma in pixels
  downsample: 4         # internal resolution divisor
  decay_rate: 0.0       # 0 = no decay, 0.001 = slow decay
  fixation_velocity_threshold: 100  # px/s
  fixation_duration_threshold: 100  # ms

# Evaluation
evaluation:
  viewing_distance_cm: 60.0
  monitor_dpi: 96.0
  num_test_points: 20

# Output
output:
  sessions_dir: "./data/sessions"
  heatmaps_dir: "./data/heatmaps"
  calibrations_dir: "./data/calibrations"
  save_raw_gaze: true
  save_video: false
```

---

## Main Application Flow (`main.py`)

```python
"""
CLI Entry Points:

1. Calibration Mode:
   python main.py calibrate --output calibration_001.yaml
   
2. Recording Mode:
   python main.py record --calibration calibration_001.yaml \
                         --duration 60 \
                         --output session_001

3. Evaluation Mode:
   python main.py evaluate --calibration calibration_001.yaml \
                           --num-points 20

4. Labeling Mode:
   python main.py label --session session_001

5. Live Demo Mode:
   python main.py demo --calibration calibration_001.yaml

Application States:
- CALIBRATING: Running calibration procedure
- RECORDING: Capturing gaze + building heatmap
- EVALUATING: Collecting ground truth measurements
- IDLE: Paused/waiting

Real-time Display:
- Camera feed with face mesh overlay
- Current gaze point (crosshair)
- Heatmap overlay (semi-transparent)
- FPS and status indicators
"""

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    
    # Add subcommands...
    
    args = parser.parse_args()
    config = load_config('config.yaml')
    
    if args.command == 'calibrate':
        run_calibration(config, args)
    elif args.command == 'record':
        run_recording(config, args)
    elif args.command == 'evaluate':
        run_evaluation(config, args)
    elif args.command == 'label':
        run_labeling(config, args)
    elif args.command == 'demo':
        run_demo(config, args)
```

---

## Session Data Format

每個 recording session 儲存以下資料：

```
data/sessions/session_001/
├── metadata.json          # Session info, calibration used, config
├── gaze_data.csv          # Raw gaze timeseries
├── heatmap.npy            # Final accumulated heatmap (numpy array)
├── heatmap.png            # Rendered heatmap visualization
├── heatmap_overlay.png    # Heatmap overlaid on screenshot
├── screenshot.png         # Screen capture at session start
└── video.mp4              # (optional) screen recording
```

**gaze_data.csv columns:**
```
timestamp_ms, gaze_x, gaze_y, gaze_pitch, gaze_yaw, head_pitch, head_yaw, head_roll, confidence, is_fixation, smoothed_x, smoothed_y
```

**metadata.json:**
```json
{
    "session_id": "session_001",
    "start_time": "2024-01-15T14:30:00Z",
    "duration_sec": 60.0,
    "calibration_file": "calibration_001.yaml",
    "screen_resolution": [1920, 1080],
    "config": { ... },
    "statistics": {
        "total_frames": 1800,
        "valid_frames": 1650,
        "fixation_count": 45,
        "mean_fixation_duration_ms": 250
    }
}
```

---

## Evaluation Pipeline

```python
"""
Complete evaluation workflow:

1. Calibration Quality Check:
   - After calibration, measure error on 5 held-out points
   - Report mean angular error
   - Recommend re-calibration if error > 5°

2. Ground Truth Collection:
   - Display 20 random target points across screen
   - User fixates on each for 1.5s
   - Record predicted vs actual positions
   - Calculate angular and screen-space errors

3. Heatmap Accuracy Assessment:
   - Record session with known AOIs (Areas of Interest)
   - Compare heatmap peaks with expected attention regions
   - Calculate: IoU, correlation, KL divergence

4. Cross-Session Analysis:
   - Compare calibration drift over time
   - Measure consistency across multiple sessions

Output Metrics:
- Angular error: mean, std, 95th percentile
- Screen error: pixels, cm, visual degrees
- Per-region breakdown (9 regions)
- Calibration quality score (0-100)
"""
```

---

## Key Implementation Notes

### Gaze Vector to Screen Coordinate Conversion

```python
def gaze_to_screen_angle(gaze_vector: np.ndarray, 
                          screen_center: Tuple[int, int],
                          viewing_distance_px: float) -> Tuple[float, float]:
    """
    Convert 3D gaze vector to screen-space angles.
    
    Coordinate system:
    - X: right (positive)
    - Y: down (positive) 
    - Z: into screen (positive)
    
    Returns pitch (vertical) and yaw (horizontal) in radians.
    """
    # Normalize gaze vector
    gaze_norm = gaze_vector / np.linalg.norm(gaze_vector)
    
    # Extract angles
    yaw = np.arctan2(gaze_norm[0], gaze_norm[2])    # horizontal
    pitch = np.arctan2(gaze_norm[1], gaze_norm[2])  # vertical
    
    return pitch, yaw
```

### Angular Error Calculation

```python
def calculate_angular_error(target_pos: Tuple[int, int],
                             predicted_pos: Tuple[int, int],
                             screen_center: Tuple[int, int],
                             viewing_distance_cm: float,
                             pixels_per_cm: float) -> float:
    """
    Calculate angular error between target and predicted gaze points.
    
    Returns error in degrees.
    """
    # Convert positions to cm from screen center
    target_cm = np.array([
        (target_pos[0] - screen_center[0]) / pixels_per_cm,
        (target_pos[1] - screen_center[1]) / pixels_per_cm
    ])
    
    predicted_cm = np.array([
        (predicted_pos[0] - screen_center[0]) / pixels_per_cm,
        (predicted_pos[1] - screen_center[1]) / pixels_per_cm
    ])
    
    # Create 3D vectors (assuming eye at center, viewing_distance back)
    target_3d = np.array([target_cm[0], target_cm[1], viewing_distance_cm])
    predicted_3d = np.array([predicted_cm[0], predicted_cm[1], viewing_distance_cm])
    
    # Normalize
    target_3d = target_3d / np.linalg.norm(target_3d)
    predicted_3d = predicted_3d / np.linalg.norm(predicted_3d)
    
    # Angular error
    cos_angle = np.clip(np.dot(target_3d, predicted_3d), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    
    return np.degrees(angle_rad)
```

---

## Testing Requirements

```python
# Unit tests for each module
tests/
├── test_gaze_estimator.py    # Mock frame input, verify output format
├── test_calibration.py       # Test polynomial fitting, save/load
├── test_smoother.py          # Verify filter behavior, edge cases
├── test_heatmap.py           # Gaussian accumulation correctness
├── test_error_metrics.py     # Known input → expected error values
└── test_integration.py       # End-to-end pipeline test

# Test data
tests/fixtures/
├── sample_frame.jpg          # Test face image
├── calibration_profile.yaml  # Known good calibration
└── expected_heatmap.npy      # Reference heatmap for comparison
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Frame processing | < 33ms | 30 FPS real-time |
| Calibration time | < 30s | 9 points × 1.5s + overhead |
| Heatmap update | < 5ms | Per gaze point |
| Angular error | < 4° | After calibration, median |
| Screen error | < 100px | At 60cm viewing distance |

---

## Usage Examples

```bash
# Initial setup
pip install ptgaze mediapipe opencv-python numpy scipy filterpy matplotlib screeninfo pyyaml

# Run calibration (required first time)
python main.py calibrate --output my_calibration.yaml

# Record 60-second session
python main.py record --calibration my_calibration.yaml --duration 60 --output session_001

# Evaluate accuracy with 20 test points
python main.py evaluate --calibration my_calibration.yaml --num-points 20

# Annotate recorded heatmap
python main.py label --session session_001

# Live demo with heatmap overlay
python main.py demo --calibration my_calibration.yaml
```

---

## Extension Points

1. **Multi-monitor support**: Detect active monitor, adjust coordinates
2. **Stimulus presentation**: Integrate with PsychoPy for controlled experiments  
3. **Remote eye tracking**: Stream gaze data over network
4. **Attention metrics**: Dwell time, scan path analysis, AOI transitions
5. **ML-based calibration**: Learn user-specific mapping without explicit calibration
6. **Blink detection**: Filter out blink periods from heatmap
7. **Pupil dilation**: Track cognitive load alongside attention
