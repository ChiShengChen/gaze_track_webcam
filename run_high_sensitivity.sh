#!/bin/bash

# High Sensitivity Eye Tracking System Startup Script

echo "=== Eye Tracking System (High Sensitivity Version) ==="
echo ""

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found, please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check Vosk model
if [ ! -d "vosk-model-small-en-us-0.15" ]; then
    echo "Error: English Vosk model not found, please run ./setup.sh first"
    exit 1
fi

# Choose calibration point count
echo "Choose calibration point count (more points = higher precision):"
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

# Start program
echo ""
echo "Starting high sensitivity eye tracking system..."
echo "Using English model: ./vosk-model-small-en-us-0.15"
echo "Calibration points: ${ROWS}x${COLS}"
echo "Optimized settings: High sensitivity, fast response, high resolution heatmap"
echo ""

python gaze_tracker.py --rows $ROWS --cols $COLS $DEBUG_FLAG
