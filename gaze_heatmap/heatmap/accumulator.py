"""
Heatmap Accumulator - Gaussian-weighted fixation aggregation.

Accumulates gaze points with Gaussian weighting and optional
temporal decay for heatmap generation.
"""

import numpy as np
import cv2
from typing import Optional, Tuple
import json
from pathlib import Path
from dataclasses import dataclass
import time


@dataclass
class FixationState:
    """State for fixation detection."""
    last_x: float = 0.0
    last_y: float = 0.0
    last_time: float = 0.0
    fixation_start: float = 0.0
    is_fixating: bool = False
    fixation_x: float = 0.0
    fixation_y: float = 0.0


class HeatmapAccumulator:
    """
    Gaussian-weighted fixation aggregation for heatmap generation.
    
    Features:
    - Real-time Gaussian blob accumulation
    - Velocity-based fixation detection
    - Temporal decay for recent-weighted heatmaps
    - Efficient downsampled internal representation
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        sigma: float = 50.0,
        downsample: int = 4,
        decay_rate: float = 0.0,
        fixation_velocity_threshold: float = 100.0,
        fixation_duration_threshold: float = 100.0
    ):
        """
        Initialize heatmap accumulator.
        
        Args:
            width: Screen width in pixels
            height: Screen height in pixels
            sigma: Gaussian sigma in pixels
            downsample: Internal resolution divisor
            decay_rate: Temporal decay rate (0 = no decay)
            fixation_velocity_threshold: Max velocity for fixation (px/s)
            fixation_duration_threshold: Min duration for fixation (ms)
        """
        self.width = width
        self.height = height
        self.sigma = sigma
        self.downsample = downsample
        self.decay_rate = decay_rate
        self.fixation_velocity_threshold = fixation_velocity_threshold
        self.fixation_duration_threshold = fixation_duration_threshold / 1000.0
        
        # Internal grid (downsampled)
        self.grid_width = max(8, width // downsample)
        self.grid_height = max(8, height // downsample)
        
        # Accumulator array
        self.accumulator = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        # Precompute Gaussian kernel
        self._precompute_gaussian_kernel()
        
        # Fixation detection state
        self.fixation_state = FixationState()
        
        # Statistics
        self.total_points = 0
        self.total_fixations = 0
        self.fixation_durations = []
        
    def _precompute_gaussian_kernel(self):
        """Precompute Gaussian kernel for efficient accumulation."""
        # Kernel size based on sigma (3 sigma covers 99.7%)
        kernel_size = int(self.sigma / self.downsample * 6) | 1  # Ensure odd
        kernel_size = max(3, min(kernel_size, min(self.grid_width, self.grid_height) // 2))
        
        # Create 2D Gaussian kernel
        x = np.arange(kernel_size) - kernel_size // 2
        y = np.arange(kernel_size) - kernel_size // 2
        xx, yy = np.meshgrid(x, y)
        
        sigma_grid = self.sigma / self.downsample
        self.gaussian_kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma_grid**2))
        self.gaussian_kernel /= self.gaussian_kernel.sum()  # Normalize
        
        self.kernel_size = kernel_size
        self.kernel_half = kernel_size // 2
        
    def add_gaze_point(
        self,
        x: float,
        y: float,
        timestamp: Optional[float] = None,
        is_fixation: Optional[bool] = None,
        weight: float = 1.0
    ):
        """
        Add single gaze point with Gaussian weighting.
        
        Args:
            x: Screen x coordinate
            y: Screen y coordinate
            timestamp: Point timestamp (for fixation detection)
            is_fixation: Override fixation detection
            weight: Point weight multiplier
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Detect fixation if not provided
        if is_fixation is None:
            is_fixation = self._detect_fixation(x, y, timestamp)
            
        # Only accumulate during fixations (or if fixation detection disabled)
        if not is_fixation and self.fixation_velocity_threshold > 0:
            return
            
        # Apply temporal decay before adding new point
        if self.decay_rate > 0:
            self.accumulator *= (1 - self.decay_rate)
            
        # Convert to grid coordinates
        gx = int(x / self.downsample)
        gy = int(y / self.downsample)
        
        # Clamp to valid range
        gx = max(0, min(gx, self.grid_width - 1))
        gy = max(0, min(gy, self.grid_height - 1))
        
        # Add Gaussian blob at location
        self._add_gaussian_blob(gx, gy, weight)
        
        self.total_points += 1
        
    def _detect_fixation(self, x: float, y: float, timestamp: float) -> bool:
        """
        Velocity-based fixation detection.
        
        A fixation is detected when gaze velocity stays below threshold
        for minimum duration.
        """
        state = self.fixation_state
        
        if state.last_time == 0:
            # First point
            state.last_x = x
            state.last_y = y
            state.last_time = timestamp
            state.fixation_start = timestamp
            state.fixation_x = x
            state.fixation_y = y
            return False
            
        # Calculate velocity
        dt = max(timestamp - state.last_time, 1e-6)
        dx = x - state.last_x
        dy = y - state.last_y
        velocity = np.sqrt(dx**2 + dy**2) / dt
        
        # Update state
        state.last_x = x
        state.last_y = y
        state.last_time = timestamp
        
        if velocity < self.fixation_velocity_threshold:
            # Low velocity - potential fixation
            if not state.is_fixating:
                # Start of new fixation
                state.fixation_start = timestamp
                state.fixation_x = x
                state.fixation_y = y
                
            duration = timestamp - state.fixation_start
            
            if duration >= self.fixation_duration_threshold:
                if not state.is_fixating:
                    self.total_fixations += 1
                state.is_fixating = True
                return True
        else:
            # High velocity - saccade
            if state.is_fixating:
                # End of fixation
                duration = timestamp - state.fixation_start
                self.fixation_durations.append(duration * 1000)  # Store in ms
            state.is_fixating = False
            
        return False
        
    def _add_gaussian_blob(self, gx: int, gy: int, weight: float = 1.0):
        """Add Gaussian blob at grid location."""
        # Calculate valid kernel region
        y_start = max(0, gy - self.kernel_half)
        y_end = min(self.grid_height, gy + self.kernel_half + 1)
        x_start = max(0, gx - self.kernel_half)
        x_end = min(self.grid_width, gx + self.kernel_half + 1)
        
        # Calculate kernel region
        ky_start = y_start - (gy - self.kernel_half)
        ky_end = ky_start + (y_end - y_start)
        kx_start = x_start - (gx - self.kernel_half)
        kx_end = kx_start + (x_end - x_start)
        
        # Add weighted kernel
        self.accumulator[y_start:y_end, x_start:x_end] += (
            self.gaussian_kernel[ky_start:ky_end, kx_start:kx_end] * weight
        )
        
    def get_heatmap(self, normalize: bool = True) -> np.ndarray:
        """
        Return current heatmap.
        
        Args:
            normalize: Normalize to 0-1 range
            
        Returns:
            2D float array (grid_height x grid_width)
        """
        heatmap = self.accumulator.copy()
        
        if normalize and heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            
        return heatmap
        
    def get_full_resolution_heatmap(self, normalize: bool = True) -> np.ndarray:
        """
        Return heatmap at full screen resolution.
        
        Args:
            normalize: Normalize to 0-1 range
            
        Returns:
            2D float array (height x width)
        """
        heatmap = self.get_heatmap(normalize=False)
        
        # Upsample with smooth interpolation
        full_heatmap = cv2.resize(
            heatmap,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR
        )
        
        if normalize and full_heatmap.max() > 0:
            full_heatmap = full_heatmap / full_heatmap.max()
            
        return full_heatmap
        
    def reset(self):
        """Clear accumulator for new session."""
        self.accumulator.fill(0)
        self.fixation_state = FixationState()
        self.total_points = 0
        self.total_fixations = 0
        self.fixation_durations = []
        
    def get_statistics(self) -> dict:
        """Get accumulator statistics."""
        stats = {
            'total_points': self.total_points,
            'total_fixations': self.total_fixations,
            'mean_fixation_duration_ms': (
                np.mean(self.fixation_durations) if self.fixation_durations else 0
            ),
            'max_intensity': float(self.accumulator.max()),
            'mean_intensity': float(self.accumulator.mean()),
        }
        return stats
        
    def save(self, path: str, include_metadata: bool = True):
        """
        Save heatmap as .npy with JSON metadata sidecar.
        
        Args:
            path: Output path (without extension)
            include_metadata: Save metadata JSON file
        """
        path = Path(path)
        
        # Save raw numpy array
        np.save(str(path) + '.npy', self.accumulator)
        
        if include_metadata:
            metadata = {
                'width': self.width,
                'height': self.height,
                'grid_width': self.grid_width,
                'grid_height': self.grid_height,
                'sigma': self.sigma,
                'downsample': self.downsample,
                'statistics': self.get_statistics(),
            }
            
            with open(str(path) + '_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
    def load(self, path: str):
        """
        Load heatmap from .npy file.
        
        Args:
            path: Path to .npy file
        """
        path = Path(path)
        
        if path.suffix != '.npy':
            path = Path(str(path) + '.npy')
            
        self.accumulator = np.load(str(path))
        
        # Try to load metadata
        metadata_path = Path(str(path).replace('.npy', '_metadata.json'))
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.sigma = metadata.get('sigma', self.sigma)
                self.downsample = metadata.get('downsample', self.downsample)


class FixationAccumulator(HeatmapAccumulator):
    """
    Extended accumulator that tracks individual fixations.
    
    Records fixation centers and durations for detailed analysis.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixations = []  # List of (x, y, duration, timestamp)
        
    def _detect_fixation(self, x: float, y: float, timestamp: float) -> bool:
        """Override to record fixation events."""
        was_fixating = self.fixation_state.is_fixating
        is_fixation = super()._detect_fixation(x, y, timestamp)
        
        # Record fixation when it ends
        if was_fixating and not self.fixation_state.is_fixating:
            duration = (timestamp - self.fixation_state.fixation_start) * 1000
            self.fixations.append((
                self.fixation_state.fixation_x,
                self.fixation_state.fixation_y,
                duration,
                self.fixation_state.fixation_start
            ))
            
        return is_fixation
        
    def get_fixations(self) -> list:
        """Get list of recorded fixations."""
        return self.fixations.copy()
        
    def reset(self):
        """Clear all data."""
        super().reset()
        self.fixations = []

