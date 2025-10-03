#!/bin/bash

# Advanced Eye Tracking System Startup Script (Full-screen overlay version)

echo "=== Eye Tracking System (Advanced Version - Full-screen Overlay) ==="
echo ""

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found, please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check Vosk model (default English model)
if [ ! -d "vosk-model-small-en-us-0.15" ]; then
    echo "Error: English Vosk model not found, please run ./setup.sh first"
    exit 1
fi

# Choose calibration point count
echo "Choose calibration point count:"
echo "1) 3x3 (9 points) - Quick calibration"
echo "2) 4x4 (16 points) - High precision calibration"
echo "3) 5x5 (25 points) - Highest precision calibration"
read -p "Please enter your choice (1-3): " calib_choice

case $calib_choice in
    1)
        ROWS=3
        COLS=3
        ;;
    2)
        ROWS=4
        COLS=4
        ;;
    3)
        ROWS=5
        COLS=5
        ;;
    *)
        echo "Invalid choice, using default 3x3"
        ROWS=3
        COLS=3
        ;;
esac

# Start program
echo ""
echo "Starting advanced eye tracking system..."
echo "Using English model: ./vosk-model-small-en-us-0.15"
echo "Calibration points: ${ROWS}x${COLS}"
echo ""
echo "Note: This version will display semi-transparent heatmap overlay on full screen"
echo "First run may require screen recording permission authorization"
echo ""

python gaze_overlay.py --rows $ROWS --cols $COLS
