"""
Calibration System - 9-point calibration with polynomial regression.

Implements personalized gaze-to-screen mapping using polynomial features
and ridge regression with edge weighting for improved corner accuracy.
"""

import numpy as np
import yaml
import cv2
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


@dataclass
class CalibrationPoint:
    """Single calibration point data."""
    screen_pos: Tuple[int, int]
    gaze_samples: List[np.ndarray] = field(default_factory=list)
    head_samples: List[np.ndarray] = field(default_factory=list)
    mean_gaze: Optional[np.ndarray] = None
    mean_head: Optional[np.ndarray] = None
    weight: float = 1.0


@dataclass
class CalibrationProfile:
    """Complete calibration profile."""
    screen_width: int
    screen_height: int
    points: List[CalibrationPoint]
    model_x: Optional[Pipeline] = None
    model_y: Optional[Pipeline] = None
    polynomial_degree: int = 2
    created_at: str = ""
    validation_error: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'polynomial_degree': self.polynomial_degree,
            'created_at': self.created_at,
            'validation_error': self.validation_error,
            'points': [
                {
                    'screen_pos': list(p.screen_pos),
                    'mean_gaze': p.mean_gaze.tolist() if p.mean_gaze is not None else None,
                    'mean_head': p.mean_head.tolist() if p.mean_head is not None else None,
                    'weight': p.weight
                }
                for p in self.points
            ]
        }


class CalibrationSystem:
    """
    9-point calibration system with polynomial regression.
    
    Calibration flow:
    1. Display calibration points sequentially (3x3 grid)
    2. Collect gaze vectors during stable fixation
    3. Fit polynomial regression for screen mapping
    4. Validate and save calibration profile
    """
    
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        num_points: int = 9,
        point_duration_ms: int = 1500,
        collection_window_ms: int = 500,
        polynomial_degree: int = 2,
        margin: float = 0.1,
        alpha: float = 1.0
    ):
        """
        Initialize calibration system.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            num_points: Number of calibration points (9 or 16)
            point_duration_ms: Display duration per point
            collection_window_ms: Sample collection window at end
            polynomial_degree: Polynomial feature degree
            margin: Screen edge margin (0-0.5)
            alpha: Ridge regression regularization
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.num_points = num_points
        self.point_duration_ms = point_duration_ms
        self.collection_window_ms = collection_window_ms
        self.polynomial_degree = polynomial_degree
        self.margin = margin
        self.alpha = alpha
        
        self.profile: Optional[CalibrationProfile] = None
        self.calibration_points = self._generate_calibration_grid()
        
    def _generate_calibration_grid(self) -> List[Tuple[int, int]]:
        """Generate calibration point grid positions."""
        if self.num_points == 9:
            rows, cols = 3, 3
        elif self.num_points == 16:
            rows, cols = 4, 4
        elif self.num_points == 5:
            # 5-point: corners + center
            return [
                (int(self.margin * self.screen_width), int(self.margin * self.screen_height)),
                (int((1 - self.margin) * self.screen_width), int(self.margin * self.screen_height)),
                (self.screen_width // 2, self.screen_height // 2),
                (int(self.margin * self.screen_width), int((1 - self.margin) * self.screen_height)),
                (int((1 - self.margin) * self.screen_width), int((1 - self.margin) * self.screen_height)),
            ]
        else:
            rows = int(np.sqrt(self.num_points))
            cols = self.num_points // rows
            
        points = []
        for row in range(rows):
            for col in range(cols):
                x = int(self.margin * self.screen_width + 
                       col * (1 - 2 * self.margin) * self.screen_width / (cols - 1))
                y = int(self.margin * self.screen_height + 
                       row * (1 - 2 * self.margin) * self.screen_height / (rows - 1))
                points.append((x, y))
                
        return points
        
    def _calculate_edge_weight(self, screen_pos: Tuple[int, int]) -> float:
        """
        Calculate weight for calibration point based on position.
        
        Edge and corner points get higher weights for better accuracy
        in peripheral regions.
        """
        x, y = screen_pos
        nx = x / self.screen_width
        ny = y / self.screen_height
        
        # Distance from edges (0 at center, 1 at edges)
        edge_dist = 1.0 - 2.0 * min(nx, 1 - nx, ny, 1 - ny)
        
        # Weight increases for edge points
        weight = 1.0 + 1.5 * max(0.0, edge_dist)
        
        return weight
        
    def _build_feature_vector(
        self,
        gaze_pitch: float,
        gaze_yaw: float,
        head_pitch: float = 0.0,
        head_yaw: float = 0.0
    ) -> np.ndarray:
        """
        Build feature vector for regression.
        
        Features: [gaze_pitch, gaze_yaw, head_pitch, head_yaw]
        Polynomial features are added by the pipeline.
        """
        return np.array([gaze_pitch, gaze_yaw, head_pitch, head_yaw])
        
    def run_calibration(
        self,
        gaze_estimator,
        cap: cv2.VideoCapture,
        window_name: str = "calibration"
    ) -> CalibrationProfile:
        """
        Run interactive calibration procedure.
        
        Args:
            gaze_estimator: GazeEstimator instance
            cap: OpenCV VideoCapture
            window_name: Calibration window name
            
        Returns:
            Fitted CalibrationProfile
        """
        # Create window - use normal window (NOT fullscreen) to avoid macOS issues
        print(f"[DEBUG] Creating window: {window_name}, size: {self.screen_width}x{self.screen_height}")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Resize to screen size (but not fullscreen)
        cv2.resizeWindow(window_name, self.screen_width, self.screen_height)
        cv2.moveWindow(window_name, 0, 0)
        
        # DO NOT use fullscreen - it causes issues on macOS
        # Just use a large window that covers the screen
        print(f"[DEBUG] Using normal window (not fullscreen) to avoid exit issues")
        
        # Force window to front (macOS specific) - do this after fullscreen
        try:
            import platform
            if platform.system() == 'Darwin':  # macOS
                import subprocess
                import time
                time.sleep(0.2)  # Wait a bit
                # Use AppleScript to bring window to front
                script = '''
                tell application "System Events"
                    try
                        set frontmost of process "Python" to true
                        set frontmost of process "python" to true
                    end try
                end tell
                '''
                subprocess.run(['osascript', '-e', script], capture_output=True, timeout=2)
        except Exception as e:
            print(f"[DEBUG] Could not bring window to front: {e}")
        
        # Show preparation screen
        prep_canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        cv2.putText(
            prep_canvas,
            "Calibration Starting...",
            (self.screen_width // 2 - 300, self.screen_height // 2 - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 255, 255),
            3
        )
        cv2.putText(
            prep_canvas,
            f"Look at {len(self.calibration_points)} points on screen",
            (self.screen_width // 2 - 400, self.screen_height // 2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (200, 200, 200),
            2
        )
        cv2.putText(
            prep_canvas,
            "Press any key to start",
            (self.screen_width // 2 - 250, self.screen_height // 2 + 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3
        )
        cv2.putText(
            prep_canvas,
            "Press ESC or 'q' to cancel (any time)",
            (self.screen_width // 2 - 400, self.screen_height // 2 + 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (200, 200, 200),
            2
        )
        cv2.imshow(window_name, prep_canvas)
        cv2.waitKey(100)  # Give window time to appear
        
        # Wait for key with timeout to allow ESC check
        print("[DEBUG] Waiting for key press (Press ESC or 'q' to cancel)...")
        start_wait = time.time()
        key = -1
        while time.time() - start_wait < 60:  # 60 second timeout
            key = cv2.waitKey(100) & 0xFF
            if key != 255:  # Key pressed
                break
            # Also check if window still exists
            try:
                cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
            except:
                break
        
        if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or 'q'
            print("[DEBUG] Calibration cancelled by user")
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt("Calibration cancelled")
        
        # Wait a moment before starting and ensure window is ready
        print("[DEBUG] Waiting for window to be ready...")
        for _ in range(10):
            cv2.waitKey(100)
        time.sleep(0.5)
        
        calibration_data = []
        
        print(f"[DEBUG] Starting calibration with {len(self.calibration_points)} points")
        for idx, (tx, ty) in enumerate(self.calibration_points):
            print(f"[DEBUG] Point {idx + 1}/{len(self.calibration_points)} at ({tx}, {ty})")
            point_data = CalibrationPoint(
                screen_pos=(tx, ty),
                weight=self._calculate_edge_weight((tx, ty))
            )
            
            start_time = time.time()
            collection_start = start_time + (self.point_duration_ms - self.collection_window_ms) / 1000.0
            
            # Show first frame immediately before loop
            initial_canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            radius = 40
            cv2.circle(initial_canvas, (tx, ty), radius + 15, (50, 50, 50), 4)
            cv2.circle(initial_canvas, (tx, ty), radius, (0, 255, 0), -1)
            cv2.circle(initial_canvas, (tx, ty), radius - 5, (0, 200, 0), 2)
            cv2.circle(initial_canvas, (tx, ty), 8, (255, 255, 255), -1)
            text = f"Look at the GREEN dot ({idx + 1}/{len(self.calibration_points)})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)[0]
            text_x = (self.screen_width - text_size[0]) // 2
            cv2.putText(initial_canvas, text, (text_x, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
            cv2.imshow(window_name, initial_canvas)
            for _ in range(5):
                cv2.waitKey(10)  # Force display update
            
            while True:
                current_time = time.time()
                elapsed = (current_time - start_time) * 1000
                
                if elapsed >= self.point_duration_ms:
                    break
                
                # Display calibration point with animation FIRST (before camera read)
                # Create black canvas (BGR format) - ensure correct dimensions
                canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                
                # Debug: verify canvas size
                if idx == 0 and elapsed < 100:  # Only print once per point
                    print(f"[DEBUG] Canvas shape: {canvas.shape}, expected: ({self.screen_height}, {self.screen_width}, 3)")
                
                # Shrinking animation
                progress = elapsed / self.point_duration_ms
                radius = int(40 * (1 - 0.2 * progress))  # Larger initial radius, slower shrink
                radius = max(15, radius)  # Minimum radius
                
                # Draw large, bright green circle
                cv2.circle(canvas, (tx, ty), radius + 15, (50, 50, 50), 4)  # Dark gray outer ring
                cv2.circle(canvas, (tx, ty), radius, (0, 255, 0), -1)  # Bright green fill
                cv2.circle(canvas, (tx, ty), radius - 5, (0, 200, 0), 2)  # Darker green inner ring
                cv2.circle(canvas, (tx, ty), 8, (255, 255, 255), -1)  # White center dot
                
                # Progress indicator (yellow ring when collecting)
                if current_time >= collection_start:
                    cv2.circle(canvas, (tx, ty), radius + 20, (0, 255, 255), 4)
                    
                # Instructions - larger and more visible
                text = f"Look at the GREEN dot ({idx + 1}/{len(self.calibration_points)})"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)[0]
                text_x = (self.screen_width - text_size[0]) // 2
                cv2.putText(
                    canvas,
                    text,
                    (text_x, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.8,
                    (255, 255, 255),
                    4
                )
                
                # Progress bar at bottom
                bar_width = int(self.screen_width * 0.7)
                bar_height = 30
                bar_x = (self.screen_width - bar_width) // 2
                bar_y = self.screen_height - 80
                # Background bar
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
                # Progress bar
                progress_width = int(bar_width * progress)
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
                # Border
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
                
                # Verify canvas has content
                if canvas.max() == 0:
                    print(f"[WARNING] Canvas is all zeros at point {idx + 1}!")
                
                # Ensure window is on top and visible
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                cv2.imshow(window_name, canvas)
                
                # Check for exit key FIRST (before camera read)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or 'q'
                    print(f"[DEBUG] Calibration cancelled at point {idx + 1}")
                    cv2.destroyWindow(window_name)
                    raise KeyboardInterrupt("Calibration cancelled")
                
                # Force window update
                cv2.waitKey(1)
                
                # Read frame and estimate gaze AFTER displaying
                # This ensures the display is not blocked by camera I/O
                ret, frame = cap.read()
                gaze_result = None
                if ret:
                    try:
                        gaze_result = gaze_estimator.estimate(frame)
                    except:
                        pass  # Ignore estimation errors during calibration
                
                # Collect samples in collection window
                if current_time >= collection_start and gaze_result is not None:
                    point_data.gaze_samples.append(
                        np.array([gaze_result.gaze_pitch, gaze_result.gaze_yaw])
                    )
                    point_data.head_samples.append(
                        np.array([gaze_result.head_pose[1], gaze_result.head_pose[2]])  # pitch, yaw
                    )
                    
            # Calculate mean values
            if len(point_data.gaze_samples) > 0:
                point_data.mean_gaze = np.mean(point_data.gaze_samples, axis=0)
                point_data.mean_head = np.mean(point_data.head_samples, axis=0)
                calibration_data.append(point_data)
                
            # Visual feedback
            canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            cv2.circle(canvas, (tx, ty), 15, (0, 200, 255), -1)
            cv2.imshow(window_name, canvas)
            cv2.waitKey(200)
            
        cv2.destroyWindow(window_name)
        
        # Create and fit profile
        self.profile = CalibrationProfile(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            points=calibration_data,
            polynomial_degree=self.polynomial_degree,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        
        self._fit_models()
        
        return self.profile
        
    def _fit_models(self):
        """Fit polynomial regression models for x and y prediction."""
        if self.profile is None or len(self.profile.points) < 5:
            raise RuntimeError("Not enough calibration points")
            
        # Build training data
        X = []
        Y_x = []
        Y_y = []
        weights = []
        
        for point in self.profile.points:
            if point.mean_gaze is None:
                continue
                
            feature = self._build_feature_vector(
                point.mean_gaze[0],  # gaze pitch
                point.mean_gaze[1],  # gaze yaw
                point.mean_head[0] if point.mean_head is not None else 0,  # head pitch
                point.mean_head[1] if point.mean_head is not None else 0   # head yaw
            )
            X.append(feature)
            Y_x.append(point.screen_pos[0])
            Y_y.append(point.screen_pos[1])
            weights.append(point.weight)
            
        X = np.array(X)
        Y_x = np.array(Y_x)
        Y_y = np.array(Y_y)
        weights = np.array(weights)
        
        # Create polynomial regression pipelines
        self.profile.model_x = Pipeline([
            ("poly", PolynomialFeatures(degree=self.polynomial_degree, include_bias=True)),
            ("reg", Ridge(alpha=self.alpha))
        ])
        self.profile.model_y = Pipeline([
            ("poly", PolynomialFeatures(degree=self.polynomial_degree, include_bias=True)),
            ("reg", Ridge(alpha=self.alpha))
        ])
        
        # Fit with sample weights
        self.profile.model_x.fit(X, Y_x, reg__sample_weight=weights)
        self.profile.model_y.fit(X, Y_y, reg__sample_weight=weights)
        
    def map_gaze_to_screen(
        self,
        gaze_pitch: float,
        gaze_yaw: float,
        head_pitch: float = 0.0,
        head_yaw: float = 0.0
    ) -> Tuple[int, int]:
        """
        Apply calibration to get screen coordinates.
        
        Args:
            gaze_pitch: Vertical gaze angle
            gaze_yaw: Horizontal gaze angle
            head_pitch: Head pitch angle
            head_yaw: Head yaw angle
            
        Returns:
            Screen coordinates (x, y)
        """
        if self.profile is None or self.profile.model_x is None:
            return (self.screen_width // 2, self.screen_height // 2)
            
        feature = self._build_feature_vector(gaze_pitch, gaze_yaw, head_pitch, head_yaw)
        feature = feature.reshape(1, -1)
        
        x = float(self.profile.model_x.predict(feature)[0])
        y = float(self.profile.model_y.predict(feature)[0])
        
        # Clamp to screen bounds
        x = max(0, min(self.screen_width - 1, int(x)))
        y = max(0, min(self.screen_height - 1, int(y)))
        
        return x, y
        
    def save_profile(self, path: str):
        """Save calibration profile to YAML file."""
        if self.profile is None:
            raise RuntimeError("No calibration profile to save")
            
        with open(path, 'w') as f:
            yaml.dump(self.profile.to_dict(), f, default_flow_style=False)
            
    def load_profile(self, path: str):
        """Load calibration profile from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            
        points = []
        for p in data['points']:
            point = CalibrationPoint(
                screen_pos=tuple(p['screen_pos']),
                mean_gaze=np.array(p['mean_gaze']) if p['mean_gaze'] else None,
                mean_head=np.array(p['mean_head']) if p['mean_head'] else None,
                weight=p['weight']
            )
            points.append(point)
            
        self.profile = CalibrationProfile(
            screen_width=data['screen_width'],
            screen_height=data['screen_height'],
            points=points,
            polynomial_degree=data['polynomial_degree'],
            created_at=data['created_at'],
            validation_error=data.get('validation_error', 0.0)
        )
        
        # Refit models from loaded data
        self._fit_models()
        
    def validate_calibration(
        self,
        gaze_estimator,
        cap: cv2.VideoCapture,
        num_validation_points: int = 5
    ) -> float:
        """
        Validate calibration accuracy with held-out points.
        
        Returns mean pixel error on validation points.
        """
        # Generate random validation points
        validation_points = [
            (
                int(np.random.uniform(self.margin, 1 - self.margin) * self.screen_width),
                int(np.random.uniform(self.margin, 1 - self.margin) * self.screen_height)
            )
            for _ in range(num_validation_points)
        ]
        
        errors = []
        
        cv2.namedWindow("validation", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("validation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        for tx, ty in validation_points:
            # Display point for 1 second
            start_time = time.time()
            predictions = []
            
            while time.time() - start_time < 1.0:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                gaze_result = gaze_estimator.estimate(frame)
                
                canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                cv2.circle(canvas, (tx, ty), 10, (0, 255, 0), -1)
                cv2.imshow("validation", canvas)
                cv2.waitKey(1)
                
                if gaze_result is not None and time.time() - start_time > 0.5:
                    pred_x, pred_y = self.map_gaze_to_screen(
                        gaze_result.gaze_pitch,
                        gaze_result.gaze_yaw,
                        gaze_result.head_pose[1],
                        gaze_result.head_pose[2]
                    )
                    predictions.append((pred_x, pred_y))
                    
            if len(predictions) > 0:
                mean_pred = np.mean(predictions, axis=0)
                error = np.sqrt((mean_pred[0] - tx) ** 2 + (mean_pred[1] - ty) ** 2)
                errors.append(error)
                
        cv2.destroyWindow("validation")
        
        if len(errors) > 0:
            mean_error = np.mean(errors)
            if self.profile:
                self.profile.validation_error = mean_error
            return mean_error
        return float('inf')

