#!/bin/bash

# High Precision Eye Tracking System Startup Script
# Uses 4x4 calibration grid, multi-frame averaging, polynomial regression

echo "=== Eye Tracking System (High Precision Version) ==="
echo ""

# Check for conda environment or virtual environment
if command -v conda &> /dev/null && conda info --envs | grep -q "pytorch_12"; then
    echo "Using conda environment: pytorch_12"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate pytorch_12
elif [ -d "venv" ]; then
    echo "Using virtual environment: venv"
    source venv/bin/activate
else
    echo "Error: No suitable environment found"
    echo "Please either:"
    echo "1. Create conda environment: conda create -n pytorch_12 python=3.11"
    echo "2. Or run ./setup.sh to create venv"
    exit 1
fi

# Check Vosk model
if [ ! -d "vosk-model-small-en-us-0.15" ]; then
    echo "Error: English Vosk model not found, please run ./setup.sh first"
    exit 1
fi

# Choose calibration point count
echo "Choose calibration point count (more points = higher precision):"
echo "1) 4x4 (16 points) - High precision calibration"
echo "2) 5x5 (25 points) - Highest precision calibration"
read -p "Please enter your choice (1-2): " calib_choice

case $calib_choice in
    1)
        ROWS=4
        COLS=4
        ;;
    2)
        ROWS=5
        COLS=5
        ;;
    *)
        echo "Invalid choice, using default 4x4"
        ROWS=4
        COLS=4
        ;;
esac

# Choose whether to show debug window
echo ""
echo "Show camera debug window? (y/n)"
read -p "Please enter your choice: " debug_choice

if [ "$debug_choice" = "y" ] || [ "$debug_choice" = "Y" ]; then
    DEBUG_FLAG="--show-cam-debug"
else
    DEBUG_FLAG=""
fi

# Choose camera mirroring
echo ""
echo "Mirror camera horizontally? (y/n)"
echo "Try this if tracking seems reversed left/right"
read -p "Please enter your choice: " mirror_choice

if [ "$mirror_choice" = "y" ] || [ "$mirror_choice" = "Y" ]; then
    MIRROR_FLAG="--cam-mirror"
else
    MIRROR_FLAG=""
fi

# Start program
echo ""
echo "Starting high precision eye tracking system..."
echo "Using English model: ./vosk-model-small-en-us-0.15"
echo "Calibration points: ${ROWS}x${COLS}"
echo "Features: Multi-frame averaging, polynomial regression, edge weighting"
echo ""

python gaze_tracker.py --rows $ROWS --cols $COLS $DEBUG_FLAG $MIRROR_FLAG
