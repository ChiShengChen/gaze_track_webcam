# Eye Tracking System

A real-time eye tracking system based on Python, using webcam for gaze tracking, voice calibration, and real-time heatmap display. Designed for macOS with support for Chinese and English voice commands.

## 📖 Documentation

- **English**: [README_EN.md](README_EN.md) - Complete English documentation
- **中文**: [README_CN.md](README_CN.md) - 完整中文說明文件

## 🚀 Quick Start

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

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve this project.

## 📞 Contact

For questions or suggestions, please contact through GitHub Issues.