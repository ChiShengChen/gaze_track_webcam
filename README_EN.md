# Eye Tracking System

A real-time eye tracking system based on Python, using webcam for gaze tracking, voice calibration, and real-time heatmap display. Designed for macOS with support for Chinese and English voice commands.

## Features

- 🎯 **Real-time Gaze Tracking**: High-precision facial feature extraction using Mediapipe
- 🎤 **Voice Calibration**: Support for "here"/"這裡" voice commands for calibration and recording
- 🔥 **Real-time Heatmap**: Real-time gaze point heatmap display with smooth and decay effects
- 📊 **Data Recording**: Automatic recording of gaze point data to CSV files
- 🖥️ **Multi-screen Support**: Automatic detection of primary screen resolution
- 🎨 **Visual Debugging**: Optional camera debug window
- 🚀 **High Precision Mode**: Multi-frame averaging, polynomial regression, edge weighting for improved accuracy
- 🔧 **Camera Mirroring**: Optional horizontal mirroring to fix left-right tracking issues
- 📈 **Accuracy Evaluation**: Built-in 5x5 test grid for quantitative accuracy assessment

## System Requirements

- macOS 10.15+
- Python 3.8+
- Built-in or external webcam
- Microphone
- At least 4GB RAM

## Installation Steps

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# If encountering audio issues, install portaudio
brew install portaudio
pip install --force-reinstall sounddevice
```

### 3. Download Vosk Speech Models

Download and extract Vosk speech models to project directory:

**English Model**:
```bash
# Download English small model
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

**Chinese Model**:
```bash
# Download Chinese small model
wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
unzip vosk-model-small-cn-0.22.zip
```

## Usage

### Quick Start (Recommended)

```bash
# Simplest startup method using default settings
./quick_start.sh
```

### High Precision Mode (Recommended for Best Accuracy)

```bash
# High precision version with multi-frame averaging and polynomial regression
./run_high_precision.sh
```

### Basic Usage

```bash
# Start basic version (will ask whether to show debug window)
./run_basic.sh

# Start advanced version (full-screen overlay)
./run_overlay.sh

# Manual startup (using default English model)
python gaze_tracker.py

# Show camera debug window
python gaze_tracker.py --show-cam-debug

# High precision with camera mirroring
python gaze_tracker.py --rows 4 --cols 4 --cam-mirror
```

### Advanced Parameters

```bash
# Custom calibration point count
python gaze_tracker.py --rows 4 --cols 4

# Use different camera
python gaze_tracker.py --camera 1

# Use Chinese model (if downloaded)
python gaze_tracker.py --vosk-model ./vosk-model-small-cn-0.22
```

### Parameter Description

- `--vosk-model`: Vosk speech model folder path, default is `./vosk-model-small-en-us-0.15`
- `--camera`: Camera index, default is 0
- `--rows`: Calibration points rows, default is 3
- `--cols`: Calibration points columns, default is 3
- `--show-cam-debug`: Show camera debug window
- `--cam-mirror`: Mirror camera horizontally (fix left-right tracking issues)

## Usage Flow

### 1. Calibration Phase

After program startup, it enters full-screen calibration mode:

1. Screen displays green dots
2. **Look at the dot** and say "here" or "這裡"
3. System records your facial features corresponding to screen coordinates
4. Repeat this process until all calibration points are completed

**Calibration Tips**:
- Maintain normal sitting posture, 50-70cm from screen
- Ensure adequate and uniform lighting
- Avoid wearing reflective glasses
- Keep head relatively stable during calibration
- **High Precision Mode**: Use 4x4 or 5x5 calibration grid for better edge accuracy
- **Multi-frame Averaging**: System automatically collects 0.4 seconds of samples per calibration point

### 2. Inference Phase

After calibration completion, enters real-time tracking mode:

- Real-time heatmap window display
- Optional camera debug window display
- Say "here" or "這裡" to record current gaze point to CSV file
- Press 'q' key to exit program

## Output Files

### gaze_points.csv

Records all voice-triggered gaze point data:

```csv
timestamp,x,y
1640995200.123,960,540
1640995205.456,1200,300
```

## Accuracy and Limitations

### Expected Accuracy

- **High Precision Mode**: Error about 1-2cm (0.5-1 degree viewing angle)
- **Ideal conditions**: Error about 1-3cm (1-2 degrees viewing angle)
- **General conditions**: Error about 3-5cm
- **Difficult conditions**: Error may exceed 5cm

### Accuracy Improvements (v2.0)

The system now includes several accuracy enhancements:

- **Multi-frame Averaging**: Collects 0.4 seconds of samples per calibration point, reducing noise
- **Polynomial Regression**: Uses 2nd-degree polynomial features for better non-linear mapping
- **Edge Weighting**: Gives higher weight to edge calibration points for better corner accuracy
- **Camera Mirroring**: Optional horizontal mirroring to fix left-right tracking issues

### Influencing Factors

- **Lighting conditions**: Backlighting and shadows reduce accuracy
- **Head posture**: Large head movements affect tracking
- **Glasses**: Reflective lenses may interfere
- **Distance**: Too close or too far affects accuracy
- **Camera position**: Off-center camera placement affects edge accuracy

### Improvement Suggestions

1. **Use High Precision Mode**: Run `./run_high_precision.sh` for best accuracy
2. **Increase calibration points**: Use 4x4 or 5x5 grid for higher accuracy
3. **Enable camera mirroring**: If tracking seems reversed left-right
4. **Regular recalibration**: Recalibrate every 30 minutes
5. **Optimize environment**: Ensure uniform lighting, avoid backlighting
6. **Maintain posture**: Try to maintain stable viewing posture
7. **Evaluate accuracy**: Use `python evaluate_accuracy.py` to test tracking precision

## Troubleshooting

### Common Issues

**1. Camera cannot open**
```bash
# Check camera permissions
# System Preferences > Security & Privacy > Camera
```

**2. Microphone cannot be used**
```bash
# Check microphone permissions
# System Preferences > Security & Privacy > Microphone
```

**3. Audio errors**
```bash
# Reinstall audio drivers
brew install portaudio
pip install --force-reinstall sounddevice
```

**4. Inaccurate voice recognition**
- Ensure quiet environment
- Speak clearly with moderate volume
- Try redownloading speech models

**5. Inaccurate tracking**
- Recalibrate using high precision mode
- Check lighting conditions
- Adjust sitting posture and distance
- Increase calibration point count
- Try camera mirroring if left-right tracking is reversed
- Use accuracy evaluation tool to test precision

### Performance Optimization

**Reduce CPU usage**:
```bash
# Lower camera resolution
python gaze_tracker.py --vosk-model ./model --camera 0
```

**Improve accuracy**:
```bash
# Use high precision mode (recommended)
./run_high_precision.sh

# Or manually increase calibration points with polynomial regression
python gaze_tracker.py --rows 4 --cols 4 --cam-mirror

# Evaluate accuracy
python evaluate_accuracy.py
```

## Advanced Features

### Full-screen Overlay Mode

To overlay heatmap on all windows, refer to the `gaze_overlay.py` advanced version.

### Custom Heatmap

You can modify the `Heatmap` class to customize:
- Color mapping
- Decay speed
- Blur level
- Grid density

### Data Analysis

Use generated CSV files for gaze point analysis:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv('gaze_points.csv')

# Plot gaze point distribution
plt.scatter(df['x'], df['y'], alpha=0.6)
plt.title('Gaze Point Distribution')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
```

## Technical Architecture

### Core Technologies

- **Mediapipe**: Facial feature extraction and iris detection
- **Vosk**: Offline speech recognition
- **OpenCV**: Image processing and display
- **Scikit-learn**: Regression model training
- **NumPy**: Numerical computation

### Algorithm Flow

1. **Feature Extraction**: Extract normalized coordinates of left and right eye irises relative to eye sockets
2. **Calibration Mapping**: Establish mapping relationship from facial features to screen coordinates
3. **Real-time Inference**: Use trained model to predict gaze points
4. **Smoothing Processing**: Use exponential moving average to reduce jitter
5. **Heatmap Generation**: Accumulate gaze points and generate visual heatmap

### Technical Principles

#### 1. **Feature Extraction (MediaPipe)**
```python
# Extract eye landmarks from 468 facial keypoints
left_eye_landmarks = landmarks[33:42]    # Left eye contour
right_eye_landmarks = landmarks[468:477]  # Right eye contour
iris_landmarks = landmarks[468:478]       # Iris center points
```

#### 2. **Machine Learning Regression (Not Simple Geometry)**
The system uses **polynomial regression** rather than simple geometric calculations:

```python
# Multi-frame averaging for noise reduction
samples = []
for 0.4 seconds:
    samples.append(extract_features(frame))
mean_features = np.mean(samples, axis=0)

# Polynomial regression pipeline
pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),  # 2nd-degree polynomial features
    ("reg", Ridge(alpha=1.0))               # L2 regularized regression
])

# Edge weighting for better corner accuracy
edge_weight = 1.0 + 1.5 * max(0.0, edge_distance)
```

#### 3. **Why Machine Learning vs. Geometry?**

| Method | Accuracy | Limitations |
|--------|----------|-------------|
| **Simple Geometry** | 5-10cm error | Only works in ideal conditions |
| **ML Regression** | 1-3cm error | Handles real-world variations |

**Key advantages of ML approach:**
- **Non-linear mapping**: Eye rotation vs. screen coordinates is non-linear
- **Personal adaptation**: Each person has different facial geometry
- **Robust to variations**: Handles different lighting, head poses, camera positions
- **Edge accuracy**: Polynomial features + edge weighting improve corner precision

#### 4. **High Precision Improvements (v2.0)**

**Multi-frame Averaging:**
```python
# Collect 0.4 seconds of samples per calibration point
samples = []
t_end = time.time() + 0.4
while time.time() < t_end:
    feat = extract_features(frame)
    if feat is not None:
        samples.append(feat)
mean_feat = np.mean(samples, axis=0)  # Noise reduction
```

**Edge Weighting:**
```python
# Higher weight for edge calibration points
edge = 1.0 - 2.0 * min(nx, 1-nx, ny, 1-ny)  # 0=center, 1=edge
weight = 1.0 + 1.5 * max(0.0, edge)         # γ=1.5 adjustable
```

**Polynomial Features:**
```python
# 2nd-degree polynomial features for non-linear mapping
PolynomialFeatures(degree=2, include_bias=True)
# Transforms: [x, y] → [1, x, y, x², xy, y²]
```

#### 5. **Mathematical Model**

The regression model learns the mapping:
```
f: R^n → R²
f(facial_features) = (screen_x, screen_y)
```

Where:
- **Input**: Normalized eye/iris coordinates (n-dimensional feature vector)
- **Output**: Screen pixel coordinates (x, y)
- **Model**: Polynomial regression with edge weighting
- **Training**: Ridge regression with L2 regularization

#### 6. **Accuracy Factors**

**Why edge accuracy is challenging:**
- **Non-linear relationship**: Eye rotation vs. screen position is quadratic
- **Camera parallax**: Off-center camera creates systematic bias
- **Head pose variations**: Different viewing angles affect mapping
- **Limited training data**: Fewer edge calibration points

**How our improvements address this:**
- **Polynomial features**: Capture non-linear relationships
- **Edge weighting**: Give more importance to edge calibration points
- **Multi-frame averaging**: Reduce noise in training data
- **Higher calibration density**: 4x4 or 5x5 grids for better coverage

## License

This project is licensed under the MIT License.

## Contributing

Welcome to submit Issues and Pull Requests to improve this project.

## Contact

For questions or suggestions, please contact through GitHub Issues.
