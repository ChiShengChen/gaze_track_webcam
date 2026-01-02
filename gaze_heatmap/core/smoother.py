"""
Gaze Smoothing - Kalman and 1€ filter implementations.

Provides temporal smoothing for gaze coordinates to reduce jitter
while maintaining responsiveness.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import time

try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False


class GazeSmoother(ABC):
    """Abstract base class for gaze smoothing filters."""
    
    @abstractmethod
    def update(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        """
        Update filter with new measurement.
        
        Args:
            x: X coordinate
            y: Y coordinate
            timestamp: Measurement timestamp (seconds)
            
        Returns:
            Smoothed (x, y) coordinates
        """
        pass
        
    @abstractmethod
    def reset(self):
        """Reset filter state."""
        pass
        
    @abstractmethod
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate (pixels/second)."""
        pass


class KalmanGazeSmoother(GazeSmoother):
    """
    Kalman filter for gaze smoothing.
    
    State: [x, y, vx, vy] (position + velocity)
    Measurement: [x, y] (screen coordinates)
    """
    
    def __init__(
        self,
        process_noise: Tuple[float, ...] = (1, 1, 10, 10),
        measurement_noise: Tuple[float, ...] = (50, 50),
        dt: float = 1/30.0,
        gap_threshold_ms: float = 500.0
    ):
        """
        Initialize Kalman filter.
        
        Args:
            process_noise: Q diagonal values [x, y, vx, vy]
            measurement_noise: R diagonal values [x, y]
            dt: Expected time step (seconds)
            gap_threshold_ms: Reset filter if gap exceeds this
        """
        self.process_noise = np.array(process_noise)
        self.measurement_noise = np.array(measurement_noise)
        self.dt = dt
        self.gap_threshold = gap_threshold_ms / 1000.0
        
        self.initialized = False
        self.last_timestamp = None
        
        if FILTERPY_AVAILABLE:
            self._init_filterpy()
        else:
            self._init_simple()
            
    def _init_filterpy(self):
        """Initialize using filterpy library."""
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise
        self.kf.Q = np.diag(self.process_noise)
        
        # Measurement noise
        self.kf.R = np.diag(self.measurement_noise)
        
        # Initial covariance
        self.kf.P *= 1000
        
    def _init_simple(self):
        """Initialize simple Kalman filter without filterpy."""
        self.x = np.zeros(4)  # State [x, y, vx, vy]
        self.P = np.eye(4) * 1000  # Covariance
        
        # State transition
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process and measurement noise
        self.Q = np.diag(self.process_noise)
        self.R = np.diag(self.measurement_noise)
        
    def update(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        """Update filter with new measurement."""
        if timestamp is None:
            timestamp = time.time()
            
        # Check for time gap - reset if too long
        if self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            if dt > self.gap_threshold:
                self.reset()
                
        self.last_timestamp = timestamp
        
        z = np.array([x, y])
        
        if not self.initialized:
            self._initialize_state(x, y)
            self.initialized = True
            return x, y
            
        if FILTERPY_AVAILABLE:
            return self._update_filterpy(z)
        else:
            return self._update_simple(z)
            
    def _initialize_state(self, x: float, y: float):
        """Initialize state with first measurement."""
        if FILTERPY_AVAILABLE:
            self.kf.x = np.array([x, y, 0, 0])
        else:
            self.x = np.array([x, y, 0, 0])
            
    def _update_filterpy(self, z: np.ndarray) -> Tuple[float, float]:
        """Update using filterpy."""
        self.kf.predict()
        self.kf.update(z)
        return float(self.kf.x[0]), float(self.kf.x[1])
        
    def _update_simple(self, z: np.ndarray) -> Tuple[float, float]:
        """Update using simple implementation."""
        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # Update
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        
        return float(self.x[0]), float(self.x[1])
        
    def reset(self):
        """Reset filter state."""
        self.initialized = False
        self.last_timestamp = None
        
        if FILTERPY_AVAILABLE:
            self.kf.x = np.zeros(4)
            self.kf.P = np.eye(4) * 1000
        else:
            self.x = np.zeros(4)
            self.P = np.eye(4) * 1000
            
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate."""
        if not self.initialized:
            return (0.0, 0.0)
            
        if FILTERPY_AVAILABLE:
            return (float(self.kf.x[2]), float(self.kf.x[3]))
        else:
            return (float(self.x[2]), float(self.x[3]))


class OneEuroGazeSmoother(GazeSmoother):
    """
    1€ (One Euro) filter for gaze smoothing.
    
    Adaptive low-pass filter that adjusts cutoff frequency based on speed.
    Good balance between smoothness and responsiveness.
    
    Reference: https://cristal.univ-lille.fr/~casiez/1euro/
    """
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        gap_threshold_ms: float = 500.0
    ):
        """
        Initialize 1€ filter.
        
        Args:
            min_cutoff: Minimum cutoff frequency (Hz) - higher = smoother
            beta: Speed coefficient - higher = more responsive to fast movements
            d_cutoff: Derivative cutoff frequency (Hz)
            gap_threshold_ms: Reset filter if gap exceeds this
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.gap_threshold = gap_threshold_ms / 1000.0
        
        # Filter state
        self.x_filter = LowPassFilter(self._alpha(min_cutoff))
        self.y_filter = LowPassFilter(self._alpha(min_cutoff))
        self.dx_filter = LowPassFilter(self._alpha(d_cutoff))
        self.dy_filter = LowPassFilter(self._alpha(d_cutoff))
        
        self.last_timestamp = None
        self.initialized = False
        
    def _alpha(self, cutoff: float, dt: float = 1/30.0) -> float:
        """Calculate smoothing factor from cutoff frequency."""
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
        
    def update(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        """Update filter with new measurement."""
        if timestamp is None:
            timestamp = time.time()
            
        # Check for time gap
        if self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            if dt > self.gap_threshold:
                self.reset()
        else:
            dt = 1/30.0
            
        self.last_timestamp = timestamp
        
        if not self.initialized:
            self.x_filter.reset(x)
            self.y_filter.reset(y)
            self.dx_filter.reset(0)
            self.dy_filter.reset(0)
            self.initialized = True
            self.prev_x = x
            self.prev_y = y
            return x, y
            
        # Estimate derivatives
        dx = (x - self.prev_x) / dt
        dy = (y - self.prev_y) / dt
        
        # Filter derivatives
        dx_hat = self.dx_filter.update(dx, self._alpha(self.d_cutoff, dt))
        dy_hat = self.dy_filter.update(dy, self._alpha(self.d_cutoff, dt))
        
        # Calculate adaptive cutoff
        speed = np.sqrt(dx_hat ** 2 + dy_hat ** 2)
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._alpha(cutoff, dt)
        
        # Filter positions
        x_hat = self.x_filter.update(x, alpha)
        y_hat = self.y_filter.update(y, alpha)
        
        self.prev_x = x
        self.prev_y = y
        
        return x_hat, y_hat
        
    def reset(self):
        """Reset filter state."""
        self.x_filter.reset()
        self.y_filter.reset()
        self.dx_filter.reset()
        self.dy_filter.reset()
        self.initialized = False
        self.last_timestamp = None
        
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate."""
        if not self.initialized:
            return (0.0, 0.0)
        return (self.dx_filter.value, self.dy_filter.value)


class LowPassFilter:
    """Simple exponential smoothing low-pass filter."""
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False
        
    def update(self, value: float, alpha: Optional[float] = None) -> float:
        """Update filter with new value."""
        if alpha is not None:
            self.alpha = alpha
            
        if not self.initialized:
            self.value = value
            self.initialized = True
        else:
            self.value = self.alpha * value + (1 - self.alpha) * self.value
            
        return self.value
        
    def reset(self, value: float = 0.0):
        """Reset filter state."""
        self.value = value
        self.initialized = value != 0.0


class ExponentialMovingAverage(GazeSmoother):
    """
    Simple exponential moving average smoother.
    
    Lighter weight alternative when Kalman/1€ are not needed.
    """
    
    def __init__(self, alpha: float = 0.3, gap_threshold_ms: float = 500.0):
        """
        Initialize EMA smoother.
        
        Args:
            alpha: Smoothing factor (0-1), higher = less smoothing
            gap_threshold_ms: Reset if gap exceeds this
        """
        self.alpha = alpha
        self.gap_threshold = gap_threshold_ms / 1000.0
        
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.last_timestamp = None
        self.initialized = False
        
    def update(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        """Update with new measurement."""
        if timestamp is None:
            timestamp = time.time()
            
        if self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            if dt > self.gap_threshold:
                self.reset()
        else:
            dt = 1/30.0
            
        self.last_timestamp = timestamp
        
        if not self.initialized:
            self.x = x
            self.y = y
            self.initialized = True
            return x, y
            
        # Update velocity estimate
        self.vx = (x - self.x) / dt
        self.vy = (y - self.y) / dt
        
        # EMA update
        self.x = self.alpha * x + (1 - self.alpha) * self.x
        self.y = self.alpha * y + (1 - self.alpha) * self.y
        
        return self.x, self.y
        
    def reset(self):
        """Reset state."""
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.initialized = False
        self.last_timestamp = None
        
    def get_velocity(self) -> Tuple[float, float]:
        """Get velocity estimate."""
        return (self.vx, self.vy)


def create_smoother(method: str = "kalman", **kwargs) -> GazeSmoother:
    """
    Factory function to create gaze smoother.
    
    Args:
        method: "kalman", "one_euro", "ema", or "none"
        **kwargs: Arguments passed to smoother constructor
        
    Returns:
        GazeSmoother instance
    """
    if method == "kalman":
        return KalmanGazeSmoother(**kwargs)
    elif method == "one_euro":
        return OneEuroGazeSmoother(**kwargs)
    elif method == "ema":
        return ExponentialMovingAverage(**kwargs)
    else:
        # No smoothing - passthrough
        return PassthroughSmoother()
        

class PassthroughSmoother(GazeSmoother):
    """No-op smoother that passes through values unchanged."""
    
    def update(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        return x, y
        
    def reset(self):
        pass
        
    def get_velocity(self) -> Tuple[float, float]:
        return (0.0, 0.0)

