#!/bin/bash

# Basic Eye Tracking System Startup Script

echo "=== Eye Tracking System (Basic Version) ==="
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

# Check Vosk model (default English model)
if [ ! -d "vosk-model-small-en-us-0.15" ]; then
    echo "Error: English Vosk model not found, please run ./setup.sh first"
    exit 1
fi

# Choose whether to show debug window
echo "Show camera debug window? (y/n)"
read -p "Please enter your choice: " debug_choice

if [ "$debug_choice" = "y" ] || [ "$debug_choice" = "Y" ]; then
    DEBUG_FLAG="--show-cam-debug"
else
    DEBUG_FLAG=""
fi

# Start program
echo ""
echo "Starting eye tracking system..."
echo "Using English model: ./vosk-model-small-en-us-0.15"
echo ""

python gaze_tracker.py $DEBUG_FLAG
