"""
Benchmark - Complete evaluation pipeline.

Provides comprehensive testing and comparison of gaze tracking performance.
"""

import numpy as np
import cv2
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime

from .error_metrics import ErrorMetricsCalculator, EvaluationResult


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    num_calibration_points: int = 9
    num_test_points: int = 20
    point_duration_ms: int = 1500
    collection_window_ms: int = 500
    repetitions: int = 1
    screen_regions: bool = True
    save_raw_data: bool = True
    

@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    config: BenchmarkConfig
    calibration_time: float
    evaluation_result: EvaluationResult
    timestamp: str
    
    def to_dict(self) -> dict:
        return {
            'calibration_time': self.calibration_time,
            'evaluation': self.evaluation_result.to_dict(),
            'timestamp': self.timestamp,
        }


class Benchmark:
    """
    Comprehensive benchmark for gaze tracking systems.
    
    Runs calibration and evaluation multiple times to assess:
    - Accuracy (mean error)
    - Precision (error variability)
    - Calibration stability
    - Per-region performance
    """
    
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        output_dir: str = "./data/benchmarks",
        **calculator_kwargs
    ):
        """
        Initialize benchmark.
        
        Args:
            screen_width: Screen width
            screen_height: Screen height
            output_dir: Directory for benchmark results
            **calculator_kwargs: Arguments for ErrorMetricsCalculator
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.calculator_kwargs = calculator_kwargs
        self.results: List[BenchmarkResult] = []
        
    def run(
        self,
        gaze_estimator,
        calibration_system,
        cap: cv2.VideoCapture,
        config: Optional[BenchmarkConfig] = None
    ) -> List[BenchmarkResult]:
        """
        Run complete benchmark.
        
        Args:
            gaze_estimator: GazeEstimator instance
            calibration_system: CalibrationSystem instance
            cap: OpenCV VideoCapture
            config: Benchmark configuration
            
        Returns:
            List of BenchmarkResult for each repetition
        """
        if config is None:
            config = BenchmarkConfig()
            
        self.results = []
        
        for rep in range(config.repetitions):
            print(f"\n=== Benchmark Run {rep + 1}/{config.repetitions} ===\n")
            
            # Run calibration
            print("Running calibration...")
            cal_start = time.time()
            
            try:
                calibration_system.run_calibration(gaze_estimator, cap)
            except KeyboardInterrupt:
                print("Calibration cancelled")
                continue
                
            cal_time = time.time() - cal_start
            print(f"Calibration completed in {cal_time:.1f}s")
            
            # Run evaluation
            print("\nRunning evaluation...")
            calculator = ErrorMetricsCalculator(
                self.screen_width,
                self.screen_height,
                **self.calculator_kwargs
            )
            
            test_points = self._generate_test_points(config.num_test_points)
            
            cv2.namedWindow("benchmark", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("benchmark", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            for idx, (tx, ty) in enumerate(test_points):
                predictions = self._collect_predictions(
                    gaze_estimator,
                    calibration_system,
                    cap,
                    (tx, ty),
                    config,
                    idx,
                    len(test_points)
                )
                
                if predictions:
                    mean_pred = np.mean(predictions, axis=0)
                    calculator.add_measurement(
                        target_screen=(tx, ty),
                        predicted_screen=(int(mean_pred[0]), int(mean_pred[1]))
                    )
                    
            cv2.destroyWindow("benchmark")
            
            # Compute results
            eval_result = calculator.compute_results()
            
            result = BenchmarkResult(
                config=config,
                calibration_time=cal_time,
                evaluation_result=eval_result,
                timestamp=datetime.now().isoformat()
            )
            
            self.results.append(result)
            
            # Print summary
            print(f"\n--- Run {rep + 1} Summary ---")
            print(f"Angular Error: {eval_result.mean_angular_error:.2f}° ± {eval_result.std_angular_error:.2f}°")
            print(f"Screen Error: {eval_result.mean_screen_error_px:.1f} px ({eval_result.mean_screen_error_cm:.2f} cm)")
            
        return self.results
        
    def _generate_test_points(self, num_points: int) -> List[tuple]:
        """Generate test points across screen."""
        margin = 0.1
        points = []
        
        # Ensure coverage of all regions
        if num_points >= 9:
            # Add region centers
            for ny in [0.2, 0.5, 0.8]:
                for nx in [0.2, 0.5, 0.8]:
                    points.append((
                        int(nx * self.screen_width),
                        int(ny * self.screen_height)
                    ))
                    
        # Add random points
        remaining = num_points - len(points)
        for _ in range(remaining):
            x = int(np.random.uniform(margin, 1 - margin) * self.screen_width)
            y = int(np.random.uniform(margin, 1 - margin) * self.screen_height)
            points.append((x, y))
            
        return points[:num_points]
        
    def _collect_predictions(
        self,
        gaze_estimator,
        calibration_system,
        cap: cv2.VideoCapture,
        target: tuple,
        config: BenchmarkConfig,
        idx: int,
        total: int
    ) -> List[tuple]:
        """Collect predictions for single test point."""
        tx, ty = target
        predictions = []
        
        start_time = time.time()
        collection_start = start_time + (config.point_duration_ms - config.collection_window_ms) / 1000.0
        
        while True:
            current_time = time.time()
            elapsed = (current_time - start_time) * 1000
            
            if elapsed >= config.point_duration_ms:
                break
                
            ret, frame = cap.read()
            if not ret:
                continue
                
            gaze_result = gaze_estimator.estimate(frame)
            
            # Display
            canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            cv2.circle(canvas, (tx, ty), 15, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                f"Test point {idx + 1}/{total}",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (200, 200, 200),
                2
            )
            cv2.imshow("benchmark", canvas)
            
            if current_time >= collection_start and gaze_result is not None:
                pred_x, pred_y = calibration_system.map_gaze_to_screen(
                    gaze_result.gaze_pitch,
                    gaze_result.gaze_yaw,
                    gaze_result.head_pose[1],
                    gaze_result.head_pose[2]
                )
                predictions.append((pred_x, pred_y))
                
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
        return predictions
        
    def save_results(self, name: Optional[str] = None):
        """Save benchmark results."""
        if not self.results:
            return
            
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        output_path = self.output_dir / f"benchmark_{name}.json"
        
        data = {
            'name': name,
            'screen_size': [self.screen_width, self.screen_height],
            'runs': [r.to_dict() for r in self.results],
            'summary': self._compute_summary(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        # Save detailed results if available
        for i, result in enumerate(self.results):
            csv_path = self.output_dir / f"benchmark_{name}_run{i+1}.csv"
            result.evaluation_result.raw_data.to_csv(csv_path, index=False)
            
        print(f"Results saved to {output_path}")
        
    def _compute_summary(self) -> dict:
        """Compute summary statistics across all runs."""
        if not self.results:
            return {}
            
        angular_errors = [r.evaluation_result.mean_angular_error for r in self.results]
        screen_errors = [r.evaluation_result.mean_screen_error_px for r in self.results]
        cal_times = [r.calibration_time for r in self.results]
        
        return {
            'num_runs': len(self.results),
            'angular_error': {
                'mean': float(np.mean(angular_errors)),
                'std': float(np.std(angular_errors)),
                'min': float(np.min(angular_errors)),
                'max': float(np.max(angular_errors)),
            },
            'screen_error_px': {
                'mean': float(np.mean(screen_errors)),
                'std': float(np.std(screen_errors)),
            },
            'calibration_time': {
                'mean': float(np.mean(cal_times)),
                'std': float(np.std(cal_times)),
            },
        }
        
    def compare_methods(
        self,
        methods: Dict[str, Any],
        gaze_estimator,
        cap: cv2.VideoCapture,
        config: Optional[BenchmarkConfig] = None
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Compare multiple calibration/smoothing methods.
        
        Args:
            methods: Dictionary of method_name -> calibration_system
            gaze_estimator: GazeEstimator instance
            cap: OpenCV VideoCapture
            config: Benchmark configuration
            
        Returns:
            Dictionary of method_name -> list of BenchmarkResult
        """
        all_results = {}
        
        for name, calibration_system in methods.items():
            print(f"\n{'='*50}")
            print(f"Testing method: {name}")
            print(f"{'='*50}")
            
            results = self.run(gaze_estimator, calibration_system, cap, config)
            all_results[name] = results
            
        # Print comparison
        self._print_comparison(all_results)
        
        return all_results
        
    def _print_comparison(self, all_results: Dict[str, List[BenchmarkResult]]):
        """Print comparison table."""
        print("\n" + "=" * 60)
        print("Method Comparison")
        print("=" * 60)
        print(f"{'Method':<20} {'Angular Error':<20} {'Screen Error':<15}")
        print("-" * 60)
        
        for name, results in all_results.items():
            if results:
                mean_angular = np.mean([r.evaluation_result.mean_angular_error for r in results])
                std_angular = np.std([r.evaluation_result.mean_angular_error for r in results])
                mean_screen = np.mean([r.evaluation_result.mean_screen_error_px for r in results])
                
                print(f"{name:<20} {mean_angular:.2f}° ± {std_angular:.2f}°    {mean_screen:.1f} px")
                
        print("=" * 60)

