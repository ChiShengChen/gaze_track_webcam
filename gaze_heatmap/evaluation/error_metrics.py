"""
Error Metrics - Angular and screen-space error calculation.

Provides comprehensive accuracy evaluation for gaze estimation systems.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import json


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    mean_angular_error: float      # degrees
    std_angular_error: float
    median_angular_error: float
    percentile_95_angular: float
    mean_screen_error_px: float    # pixels
    std_screen_error_px: float
    mean_screen_error_cm: float    # centimeters
    precision: float               # within-subject std
    per_region_errors: Dict[str, float]  # 'top-left', 'center', etc.
    raw_data: pd.DataFrame         # all individual measurements
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'mean_angular_error': self.mean_angular_error,
            'std_angular_error': self.std_angular_error,
            'median_angular_error': self.median_angular_error,
            'percentile_95_angular': self.percentile_95_angular,
            'mean_screen_error_px': self.mean_screen_error_px,
            'std_screen_error_px': self.std_screen_error_px,
            'mean_screen_error_cm': self.mean_screen_error_cm,
            'precision': self.precision,
            'per_region_errors': self.per_region_errors,
        }
        
    def save(self, path: str):
        """Save results to JSON."""
        data = self.to_dict()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


@dataclass
class Measurement:
    """Single ground truth measurement."""
    timestamp: float
    target_screen: Tuple[int, int]
    predicted_screen: Tuple[int, int]
    gaze_vector: Optional[np.ndarray] = None
    head_pose: Optional[np.ndarray] = None
    region: str = ""


class ErrorMetricsCalculator:
    """
    Calculate angular and screen-space errors between predicted and ground truth gaze.
    
    Supports both online (during session) and offline (post-hoc) evaluation.
    """
    
    # Screen regions for per-region analysis
    REGIONS = {
        'top-left': (0.0, 0.33, 0.0, 0.33),
        'top-center': (0.33, 0.67, 0.0, 0.33),
        'top-right': (0.67, 1.0, 0.0, 0.33),
        'middle-left': (0.0, 0.33, 0.33, 0.67),
        'center': (0.33, 0.67, 0.33, 0.67),
        'middle-right': (0.67, 1.0, 0.33, 0.67),
        'bottom-left': (0.0, 0.33, 0.67, 1.0),
        'bottom-center': (0.33, 0.67, 0.67, 1.0),
        'bottom-right': (0.67, 1.0, 0.67, 1.0),
    }
    
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        monitor_dpi: float = 96.0,
        viewing_distance_cm: float = 60.0
    ):
        """
        Initialize error calculator.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            monitor_dpi: Monitor DPI for cm conversion
            viewing_distance_cm: Viewing distance in centimeters
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.monitor_dpi = monitor_dpi
        self.viewing_distance_cm = viewing_distance_cm
        
        self.pixels_per_cm = monitor_dpi / 2.54
        self.screen_center = (screen_width // 2, screen_height // 2)
        
        self.measurements: List[Measurement] = []
        
    def add_measurement(
        self,
        target_screen: Tuple[int, int],
        predicted_screen: Tuple[int, int],
        gaze_vector: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        head_pose: Optional[np.ndarray] = None
    ):
        """
        Record single ground truth measurement.
        
        Args:
            target_screen: True screen position
            predicted_screen: Predicted gaze position
            gaze_vector: Raw gaze vector (optional)
            timestamp: Measurement timestamp
            head_pose: Head pose angles (optional)
        """
        import time
        
        if timestamp is None:
            timestamp = time.time()
            
        # Determine region
        region = self._get_region(target_screen)
        
        measurement = Measurement(
            timestamp=timestamp,
            target_screen=target_screen,
            predicted_screen=predicted_screen,
            gaze_vector=gaze_vector,
            head_pose=head_pose,
            region=region
        )
        
        self.measurements.append(measurement)
        
    def _get_region(self, pos: Tuple[int, int]) -> str:
        """Get screen region for position."""
        nx = pos[0] / self.screen_width
        ny = pos[1] / self.screen_height
        
        for name, (x_min, x_max, y_min, y_max) in self.REGIONS.items():
            if x_min <= nx < x_max and y_min <= ny < y_max:
                return name
                
        return 'unknown'
        
    def calculate_angular_error(
        self,
        target_screen: Tuple[int, int],
        predicted_screen: Tuple[int, int]
    ) -> float:
        """
        Calculate angular error between target and predicted positions.
        
        Uses viewing geometry to convert screen positions to visual angles.
        
        Args:
            target_screen: Target screen position
            predicted_screen: Predicted screen position
            
        Returns:
            Angular error in degrees
        """
        # Convert positions to cm from screen center
        target_cm = np.array([
            (target_screen[0] - self.screen_center[0]) / self.pixels_per_cm,
            (target_screen[1] - self.screen_center[1]) / self.pixels_per_cm
        ])
        
        predicted_cm = np.array([
            (predicted_screen[0] - self.screen_center[0]) / self.pixels_per_cm,
            (predicted_screen[1] - self.screen_center[1]) / self.pixels_per_cm
        ])
        
        # Create 3D vectors (eye at origin, looking at screen)
        target_3d = np.array([
            target_cm[0],
            target_cm[1],
            self.viewing_distance_cm
        ])
        
        predicted_3d = np.array([
            predicted_cm[0],
            predicted_cm[1],
            self.viewing_distance_cm
        ])
        
        # Normalize
        target_3d = target_3d / np.linalg.norm(target_3d)
        predicted_3d = predicted_3d / np.linalg.norm(predicted_3d)
        
        # Angular error
        cos_angle = np.clip(np.dot(target_3d, predicted_3d), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        
        return np.degrees(angle_rad)
        
    def calculate_screen_error(
        self,
        target_screen: Tuple[int, int],
        predicted_screen: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Calculate screen-space error.
        
        Returns:
            Tuple of (error_pixels, error_cm)
        """
        error_px = np.sqrt(
            (target_screen[0] - predicted_screen[0]) ** 2 +
            (target_screen[1] - predicted_screen[1]) ** 2
        )
        
        error_cm = error_px / self.pixels_per_cm
        
        return error_px, error_cm
        
    def compute_results(self) -> EvaluationResult:
        """
        Calculate all metrics from collected measurements.
        
        Returns:
            EvaluationResult with comprehensive metrics
        """
        if not self.measurements:
            raise ValueError("No measurements collected")
            
        # Calculate errors for each measurement
        angular_errors = []
        screen_errors_px = []
        screen_errors_cm = []
        
        for m in self.measurements:
            ang_err = self.calculate_angular_error(m.target_screen, m.predicted_screen)
            px_err, cm_err = self.calculate_screen_error(m.target_screen, m.predicted_screen)
            
            angular_errors.append(ang_err)
            screen_errors_px.append(px_err)
            screen_errors_cm.append(cm_err)
            
        angular_errors = np.array(angular_errors)
        screen_errors_px = np.array(screen_errors_px)
        screen_errors_cm = np.array(screen_errors_cm)
        
        # Per-region analysis
        per_region_errors = {}
        for region in self.REGIONS.keys():
            region_measurements = [m for m in self.measurements if m.region == region]
            if region_measurements:
                region_errors = [
                    self.calculate_angular_error(m.target_screen, m.predicted_screen)
                    for m in region_measurements
                ]
                per_region_errors[region] = float(np.mean(region_errors))
            else:
                per_region_errors[region] = float('nan')
                
        # Build raw data DataFrame
        raw_data = pd.DataFrame([
            {
                'timestamp': m.timestamp,
                'target_x': m.target_screen[0],
                'target_y': m.target_screen[1],
                'predicted_x': m.predicted_screen[0],
                'predicted_y': m.predicted_screen[1],
                'angular_error': angular_errors[i],
                'screen_error_px': screen_errors_px[i],
                'screen_error_cm': screen_errors_cm[i],
                'region': m.region,
            }
            for i, m in enumerate(self.measurements)
        ])
        
        return EvaluationResult(
            mean_angular_error=float(np.mean(angular_errors)),
            std_angular_error=float(np.std(angular_errors)),
            median_angular_error=float(np.median(angular_errors)),
            percentile_95_angular=float(np.percentile(angular_errors, 95)),
            mean_screen_error_px=float(np.mean(screen_errors_px)),
            std_screen_error_px=float(np.std(screen_errors_px)),
            mean_screen_error_cm=float(np.mean(screen_errors_cm)),
            precision=float(np.std(angular_errors)),  # Within-subject variability
            per_region_errors=per_region_errors,
            raw_data=raw_data
        )
        
    def reset(self):
        """Clear all measurements."""
        self.measurements = []
        
    def export_report(self, path: str, include_plots: bool = True):
        """
        Generate evaluation report.
        
        Args:
            path: Output path for report
            include_plots: Include visualization plots
        """
        results = self.compute_results()
        path = Path(path)
        
        # Save JSON summary
        results.save(str(path.with_suffix('.json')))
        
        # Save raw data CSV
        results.raw_data.to_csv(str(path.with_suffix('.csv')), index=False)
        
        if include_plots:
            self._generate_plots(results, path)
            
    def _generate_plots(self, results: EvaluationResult, path: Path):
        """Generate visualization plots."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Error distribution histogram
        ax = axes[0, 0]
        ax.hist(results.raw_data['angular_error'], bins=20, edgecolor='black')
        ax.axvline(results.mean_angular_error, color='r', linestyle='--', 
                   label=f'Mean: {results.mean_angular_error:.2f}°')
        ax.set_xlabel('Angular Error (degrees)')
        ax.set_ylabel('Count')
        ax.set_title('Angular Error Distribution')
        ax.legend()
        
        # 2. Per-region heatmap
        ax = axes[0, 1]
        region_grid = np.zeros((3, 3))
        region_map = {
            'top-left': (0, 0), 'top-center': (0, 1), 'top-right': (0, 2),
            'middle-left': (1, 0), 'center': (1, 1), 'middle-right': (1, 2),
            'bottom-left': (2, 0), 'bottom-center': (2, 1), 'bottom-right': (2, 2),
        }
        for region, error in results.per_region_errors.items():
            if region in region_map and not np.isnan(error):
                r, c = region_map[region]
                region_grid[r, c] = error
                
        im = ax.imshow(region_grid, cmap='RdYlGn_r', vmin=0)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['Left', 'Center', 'Right'])
        ax.set_yticklabels(['Top', 'Middle', 'Bottom'])
        ax.set_title('Per-Region Angular Error (°)')
        plt.colorbar(im, ax=ax)
        
        # Annotate values
        for region, error in results.per_region_errors.items():
            if region in region_map and not np.isnan(error):
                r, c = region_map[region]
                ax.text(c, r, f'{error:.1f}°', ha='center', va='center', fontsize=12)
        
        # 3. Scatter plot of predicted vs target
        ax = axes[1, 0]
        ax.scatter(
            results.raw_data['target_x'],
            results.raw_data['target_y'],
            c='green', label='Target', alpha=0.6, s=50
        )
        ax.scatter(
            results.raw_data['predicted_x'],
            results.raw_data['predicted_y'],
            c='red', label='Predicted', alpha=0.6, s=30
        )
        # Draw lines connecting pairs
        for _, row in results.raw_data.iterrows():
            ax.plot(
                [row['target_x'], row['predicted_x']],
                [row['target_y'], row['predicted_y']],
                'b-', alpha=0.3
            )
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title('Target vs Predicted Positions')
        ax.legend()
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        stats_text = f"""
        Evaluation Summary
        ─────────────────────────────
        
        Angular Error:
          Mean:     {results.mean_angular_error:.2f}°
          Std:      {results.std_angular_error:.2f}°
          Median:   {results.median_angular_error:.2f}°
          95th %:   {results.percentile_95_angular:.2f}°
        
        Screen Error:
          Mean:     {results.mean_screen_error_px:.1f} px
          Mean:     {results.mean_screen_error_cm:.2f} cm
        
        Precision:  {results.precision:.2f}°
        
        Measurements: {len(results.raw_data)}
        """
        ax.text(0.1, 0.5, stats_text, fontsize=11, fontfamily='monospace',
                verticalalignment='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(str(path.with_suffix('.png')), dpi=150)
        plt.close()


class OnlineEvaluator:
    """
    Online evaluation during active gaze tracking.
    
    Displays periodic test points and collects measurements.
    """
    
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        num_test_points: int = 20,
        point_duration_ms: int = 1500,
        collection_window_ms: int = 500,
        **kwargs
    ):
        """
        Initialize online evaluator.
        
        Args:
            screen_width: Screen width
            screen_height: Screen height
            num_test_points: Number of test points to display
            point_duration_ms: Display duration per point
            collection_window_ms: Sample collection window
            **kwargs: Additional arguments for ErrorMetricsCalculator
        """
        self.calculator = ErrorMetricsCalculator(
            screen_width, screen_height, **kwargs
        )
        self.num_test_points = num_test_points
        self.point_duration_ms = point_duration_ms
        self.collection_window_ms = collection_window_ms
        
        # Generate test points
        self.test_points = self._generate_test_points(screen_width, screen_height)
        
    def _generate_test_points(
        self,
        width: int,
        height: int
    ) -> List[Tuple[int, int]]:
        """Generate random test points across screen."""
        margin = 0.1
        points = []
        
        for _ in range(self.num_test_points):
            x = int(np.random.uniform(margin, 1 - margin) * width)
            y = int(np.random.uniform(margin, 1 - margin) * height)
            points.append((x, y))
            
        return points
        
    def run_evaluation(
        self,
        gaze_estimator,
        calibration_system,
        cap,
        window_name: str = "evaluation"
    ) -> EvaluationResult:
        """
        Run interactive evaluation procedure.
        
        Args:
            gaze_estimator: GazeEstimator instance
            calibration_system: CalibrationSystem with fitted model
            cap: OpenCV VideoCapture
            window_name: Window name
            
        Returns:
            EvaluationResult
        """
        import cv2
        import time
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        screen_h = self.calculator.screen_height
        screen_w = self.calculator.screen_width
        
        for idx, (tx, ty) in enumerate(self.test_points):
            start_time = time.time()
            collection_start = start_time + (self.point_duration_ms - self.collection_window_ms) / 1000.0
            
            predictions = []
            
            while True:
                current_time = time.time()
                elapsed = (current_time - start_time) * 1000
                
                if elapsed >= self.point_duration_ms:
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                gaze_result = gaze_estimator.estimate(frame)
                
                # Display test point
                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                cv2.circle(canvas, (tx, ty), 15, (0, 255, 0), -1)
                cv2.putText(
                    canvas,
                    f"Look at the dot ({idx + 1}/{len(self.test_points)})",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (200, 200, 200),
                    2
                )
                cv2.imshow(window_name, canvas)
                
                # Collect predictions in collection window
                if current_time >= collection_start and gaze_result is not None:
                    pred_x, pred_y = calibration_system.map_gaze_to_screen(
                        gaze_result.gaze_pitch,
                        gaze_result.gaze_yaw,
                        gaze_result.head_pose[1],
                        gaze_result.head_pose[2]
                    )
                    predictions.append((pred_x, pred_y))
                    
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    cv2.destroyWindow(window_name)
                    return self.calculator.compute_results()
                    
            # Record mean prediction for this point
            if len(predictions) > 0:
                mean_pred = np.mean(predictions, axis=0)
                self.calculator.add_measurement(
                    target_screen=(tx, ty),
                    predicted_screen=(int(mean_pred[0]), int(mean_pred[1]))
                )
                
        cv2.destroyWindow(window_name)
        
        return self.calculator.compute_results()

