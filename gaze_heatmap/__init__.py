"""
Gaze Heatmap Application

Real-time webcam-based gaze estimation system that generates 
screen attention heatmaps for UX analysis.

Features:
- Gaze estimation using ptgaze (ETH-XGaze) or MediaPipe fallback
- 9-point polynomial calibration with edge weighting
- Kalman and 1€ filter smoothing options
- Real-time Gaussian-weighted heatmap generation
- Angular error evaluation metrics
- AOI annotation and labeling tools

Usage:
    # Calibration
    python -m gaze_heatmap calibrate --output my_calibration.yaml
    
    # Recording session
    python -m gaze_heatmap record --calibration my_calibration.yaml --duration 60
    
    # Evaluation
    python -m gaze_heatmap evaluate --calibration my_calibration.yaml
    
    # Live demo
    python -m gaze_heatmap demo --calibration my_calibration.yaml
"""

__version__ = "1.0.0"
__author__ = "Gaze Heatmap Team"

from .core import (
    GazeEstimator,
    GazeResult,
    FaceDetector,
    CalibrationSystem,
    CalibrationProfile,
    GazeSmoother,
    KalmanGazeSmoother,
    OneEuroGazeSmoother,
    ScreenMapper,
)

from .heatmap import (
    HeatmapAccumulator,
    HeatmapRenderer,
    HeatmapExporter,
)

from .evaluation import (
    ErrorMetricsCalculator,
    EvaluationResult,
    LabelingTool,
    Benchmark,
)

__all__ = [
    # Core
    'GazeEstimator',
    'GazeResult',
    'FaceDetector',
    'CalibrationSystem',
    'CalibrationProfile',
    'GazeSmoother',
    'KalmanGazeSmoother',
    'OneEuroGazeSmoother',
    'ScreenMapper',
    # Heatmap
    'HeatmapAccumulator',
    'HeatmapRenderer',
    'HeatmapExporter',
    # Evaluation
    'ErrorMetricsCalculator',
    'EvaluationResult',
    'LabelingTool',
    'Benchmark',
]

