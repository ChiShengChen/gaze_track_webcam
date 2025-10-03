#!/bin/bash

# Quick Start Script - No selection needed, uses default settings directly

echo "=== Eye Tracking System (Quick Start) ==="
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

# Start basic version directly
echo "Starting eye tracking system (basic version)..."
echo "Using default settings: English model, 3x3 calibration points, no debug window"
echo ""

python gaze_tracker.py
