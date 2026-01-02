# Eye Tracking System

A real-time eye tracking system based on Python, using webcam for gaze tracking, voice calibration, and real-time heatmap display. Designed for macOS with support for Chinese and English voice commands.
<img width="2048" height="1234" alt="image" src="https://github.com/user-attachments/assets/4a3bbb7c-ac1e-4c9d-b9ea-ae4b0405d8f1" />

## 📖 Documentation

- **English**: [README_EN.md](README_EN.md) - Complete English documentation
- **中文**: [README_CN.md](README_CN.md) - 完整中文說明文件

## 🆕 Two Systems Available

This repository contains **two eye tracking systems**:

### 1. **Gaze Heatmap System** (New, Recommended) ⭐
- **Location**: `gaze_heatmap/`
- **Features**: ETH-XGaze model (default, best accuracy), L2CS-Net support, comprehensive CLI, advanced calibration
- **Best for**: Research, accuracy evaluation, detailed analysis
- **Quick Start**: See [Gaze Heatmap Quick Start](#-gaze-heatmap-system-quick-start)

### 2. **Voice-Controlled System** (Original)
- **Location**: Root directory (`gaze_tracker.py`)
- **Features**: Voice commands, simple interface
- **Best for**: Quick demos, simple use cases
- **Quick Start**: See [Original System Quick Start](#-original-system-quick-start)

---

## 🚀 Gaze Heatmap System Quick Start

The new **Gaze Heatmap System** provides advanced gaze tracking with **ETH-XGaze as the default model** (best accuracy, supports MPS GPU acceleration):

### ⚠️ Important Usage Notes

**Before using the system, please note:**

- 📏 **Viewing Distance**: Maintain approximately **60cm** distance from the screen for optimal accuracy
- 🎯 **Head Position**: Keep your **head relatively still** during calibration and tracking
- 💡 **Lighting**: Ensure adequate and uniform lighting; avoid backlighting or shadows
- 👓 **Glasses**: Avoid reflective glasses that may interfere with tracking
- 🪑 **Posture**: Maintain a stable sitting posture throughout the session
- 🔄 **Recalibration**: Recalibrate if you change position or lighting conditions significantly

### Setup and Usage

```bash
cd gaze_heatmap

# Setup conda environment (one-time)
conda create -n gaze_eth python=3.10 -y
conda activate gaze_eth
pip install -r requirements.txt

# 1. Calibrate (required first)
python main.py calibrate --output my_calibration.yaml

# 2. Run live demo
python main.py demo --calibration my_calibration.yaml

# 3. Record session
python main.py record --calibration my_calibration.yaml --duration 60

# 4. Evaluate accuracy
python main.py evaluate --calibration my_calibration.yaml --num-points 20
```

### Available Commands

| Command | Description |
|---------|-------------|
| `calibrate` | Run 9/16-point calibration procedure |
| `demo` | Live demo with real-time heatmap |
| `record` | Record gaze session with heatmap |
| `evaluate` | Evaluate tracking accuracy |
| `label` | Annotate recorded heatmaps |

### Key Features

- 🎯 **ETH-XGaze Model**: State-of-the-art gaze estimation (default, supports MPS GPU acceleration)
- 📊 **Comprehensive CLI**: Full command-line interface for all operations
- 🔬 **Accuracy Evaluation**: Built-in metrics (angular error, screen error, precision)
- 📈 **Advanced Calibration**: Polynomial regression with edge weighting
- 🎨 **Real-time Heatmap**: Live visualization with smoothing and fixation detection
- 💾 **Data Export**: Save sessions, heatmaps, and evaluation reports

### Model Accuracy Comparison

Based on testing results, the supported models rank as follows:

| Model | Accuracy | Performance | Notes |
|-------|----------|-------------|-------|
| **ETH-XGaze** | ⭐⭐⭐⭐⭐ Best | Fast (MPS GPU support) | Default model, recommended for best accuracy |
| **L2CS-Net** | ⭐⭐⭐⭐ Very Good | Moderate | Good alternative, requires manual weight download |
| **MediaPipe** | ⭐⭐⭐ Good | Fastest | Fallback option, no additional setup required |

**Test Results**: ETH-XGaze > L2CS > MediaPipe

ETH-XGaze provides the best accuracy and supports Apple Silicon GPU acceleration via MPS, making it the recommended default choice.

For detailed documentation, see [`gaze_heatmap/how_to_run.md`](gaze_heatmap/how_to_run.md)

---

## 🚀 Original System Quick Start

```bash
# 1. Setup (one-time)
./setup.sh

# 2. Quick start
./quick_start.sh

# 3. High precision mode (best accuracy)
./run_high_precision.sh
```

## 🎯 Features

- 🎯 **Real-time Gaze Tracking**: High-precision facial feature extraction using Mediapipe
- 🎤 **Voice Calibration**: Support for "here"/"這裡" voice commands for calibration and recording
- 🔥 **Real-time Heatmap**: Real-time gaze point heatmap display with smooth and decay effects
- 📊 **Data Recording**: Automatic recording of gaze point data to CSV files
- 🖥️ **Multi-screen Support**: Automatic detection of primary screen resolution
- 🎨 **Visual Debugging**: Optional camera debug window
- 🚀 **High Precision Mode**: Multi-frame averaging, polynomial regression, edge weighting for improved accuracy
- 🔧 **Camera Mirroring**: Optional horizontal mirroring to fix left-right tracking issues
- 📈 **Accuracy Evaluation**: Built-in 5x5 test grid for quantitative accuracy assessment

## 🛠️ System Requirements

- macOS 10.15+
- Python 3.8+
- Built-in or external webcam
- Microphone
- At least 4GB RAM

## 📋 Available Scripts

| Script | Description | Language |
|--------|-------------|----------|
| `./quick_start.sh` | Quick start with default settings | English |
| `./run_high_precision.sh` | High precision mode (best accuracy) | English |
| `./run_basic.sh` | Basic version with options | English |
| `./run_overlay.sh` | Advanced full-screen overlay version | English |
| `./run_high_sensitivity.sh` | High sensitivity version | English |

## 🔧 Installation

```bash
# Clone the repository
git clone <repository-url>
cd webcam_voice_label

# Run setup script
./setup.sh
```

## 📚 Usage Examples

### Basic Usage
```bash
# Quick start (recommended)
./quick_start.sh

# High precision mode (best accuracy)
./run_high_precision.sh

# Basic version with debug window
./run_basic.sh

# Advanced full-screen overlay
./run_overlay.sh
```

### Manual Usage
```bash
# Basic tracking
python gaze_tracker.py

# With debug window
python gaze_tracker.py --show-cam-debug

# High precision with camera mirroring
python gaze_tracker.py --rows 4 --cols 4 --cam-mirror

# Use Chinese model
python gaze_tracker.py --vosk-model ./vosk-model-small-cn-0.22

# Evaluate accuracy
python evaluate_accuracy.py
```

## 📊 Output

The system generates a `gaze_points.csv` file with timestamped gaze coordinates:

```csv
timestamp,x,y
1640995200.123,960,540
1640995205.456,1200,300
```

## 🎯 Accuracy

- **High Precision Mode**: ~1-2cm error (0.5-1° viewing angle)
- **Ideal conditions**: ~1-3cm error (1-2° viewing angle)
- **General conditions**: ~3-5cm error
- **Difficult conditions**: >5cm error

### Accuracy Improvements (v2.0)

- **Multi-frame Averaging**: Collects 0.4 seconds of samples per calibration point
- **Polynomial Regression**: Uses 2nd-degree polynomial features for better non-linear mapping
- **Edge Weighting**: Gives higher weight to edge calibration points for better corner accuracy
- **Camera Mirroring**: Optional horizontal mirroring to fix left-right tracking issues

## 🔧 Troubleshooting

### Common Issues

1. **Camera not working**: Check System Preferences > Security & Privacy > Camera
2. **Microphone not working**: Check System Preferences > Security & Privacy > Microphone
3. **Audio errors**: Run `brew install portaudio && pip install --force-reinstall sounddevice`
4. **Poor tracking**: Use high precision mode, recalibrate, check lighting, adjust posture, try camera mirroring

## 🏗️ Technical Stack

- **Mediapipe**: Facial feature extraction and iris detection
- **Vosk**: Offline speech recognition
- **OpenCV**: Image processing and display
- **Scikit-learn**: Regression model training
- **PyQt6**: Advanced overlay interface

## 🔬 Technical Principles

### **Machine Learning vs. Simple Geometry**

This system uses **polynomial regression** rather than simple geometric calculations:

| Method | Accuracy | Why Better |
|--------|----------|------------|
| **Simple Geometry** | 5-10cm error | Only works in ideal conditions |
| **ML Regression** | 1-3cm error | Handles real-world variations |

### **Key Technical Features**

- **Polynomial Regression**: 2nd-degree features for non-linear mapping
- **Multi-frame Averaging**: 0.4s samples per calibration point for noise reduction
- **Edge Weighting**: Higher weight for corner calibration points
- **Personal Adaptation**: Learns individual facial geometry

### **Mathematical Model**
```
f: R^n → R²
f(facial_features) = (screen_x, screen_y)
```

**Input**: Normalized eye/iris coordinates  
**Output**: Screen pixel coordinates  
**Model**: Polynomial regression with edge weighting

For detailed technical documentation, see:
- [English Technical Details](README_EN.md#technical-principles)
- [中文技術詳情](README_CN.md#技術原理)

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve this project.

## 📞 Contact

For questions or suggestions, please contact through GitHub Issues.
