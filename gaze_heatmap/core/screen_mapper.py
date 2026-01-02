"""
Screen Mapper - Gaze vector to screen coordinate projection.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ScreenConfig:
    """Screen configuration parameters."""
    width: int
    height: int
    dpi: float = 96.0
    viewing_distance_cm: float = 60.0
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.width // 2, self.height // 2)
        
    @property
    def pixels_per_cm(self) -> float:
        return self.dpi / 2.54
        
    @property
    def viewing_distance_px(self) -> float:
        return self.viewing_distance_cm * self.pixels_per_cm


class ScreenMapper:
    """
    Maps gaze vectors and angles to screen coordinates.
    
    Provides geometric projection from 3D gaze direction to 2D screen position,
    independent of the calibration-based polynomial mapping.
    """
    
    def __init__(self, screen_config: ScreenConfig):
        """
        Initialize screen mapper.
        
        Args:
            screen_config: Screen configuration parameters
        """
        self.config = screen_config
        
    def gaze_angles_to_screen(
        self,
        pitch: float,
        yaw: float,
        head_pitch: float = 0.0,
        head_yaw: float = 0.0
    ) -> Tuple[int, int]:
        """
        Convert gaze angles to screen coordinates using geometric projection.
        
        Args:
            pitch: Vertical gaze angle in radians
            yaw: Horizontal gaze angle in radians
            head_pitch: Head pitch for compensation
            head_yaw: Head yaw for compensation
            
        Returns:
            Screen coordinates (x, y)
        """
        # Combine gaze and head angles
        combined_yaw = yaw + head_yaw * 0.3  # Partial head compensation
        combined_pitch = pitch + head_pitch * 0.3
        
        # Project to screen using viewing distance
        dx = self.config.viewing_distance_px * np.tan(combined_yaw)
        dy = self.config.viewing_distance_px * np.tan(combined_pitch)
        
        # Convert to screen coordinates (centered)
        x = int(self.config.center[0] + dx)
        y = int(self.config.center[1] + dy)
        
        # Clamp to screen bounds
        x = max(0, min(self.config.width - 1, x))
        y = max(0, min(self.config.height - 1, y))
        
        return x, y
        
    def gaze_vector_to_screen(
        self,
        gaze_vector: np.ndarray,
        head_pose: Optional[np.ndarray] = None
    ) -> Tuple[int, int]:
        """
        Convert 3D gaze vector to screen coordinates.
        
        Args:
            gaze_vector: 3D gaze direction vector (x, y, z)
            head_pose: Optional head pose (roll, pitch, yaw)
            
        Returns:
            Screen coordinates (x, y)
        """
        # Normalize gaze vector
        gaze_norm = gaze_vector / (np.linalg.norm(gaze_vector) + 1e-10)
        
        # Extract angles from gaze vector
        yaw = np.arctan2(gaze_norm[0], gaze_norm[2])
        pitch = np.arctan2(gaze_norm[1], gaze_norm[2])
        
        # Extract head pose if provided
        head_pitch = head_pose[1] if head_pose is not None else 0.0
        head_yaw = head_pose[2] if head_pose is not None else 0.0
        
        return self.gaze_angles_to_screen(pitch, yaw, head_pitch, head_yaw)
        
    def screen_to_angles(
        self,
        screen_pos: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Convert screen position to visual angles from center.
        
        Args:
            screen_pos: Screen coordinates (x, y)
            
        Returns:
            Tuple of (pitch, yaw) in radians
        """
        dx_px = screen_pos[0] - self.config.center[0]
        dy_px = screen_pos[1] - self.config.center[1]
        
        yaw = np.arctan2(dx_px, self.config.viewing_distance_px)
        pitch = np.arctan2(dy_px, self.config.viewing_distance_px)
        
        return pitch, yaw
        
    def calculate_visual_angle(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int]
    ) -> float:
        """
        Calculate visual angle between two screen positions.
        
        Args:
            pos1: First screen position
            pos2: Second screen position
            
        Returns:
            Visual angle in degrees
        """
        pitch1, yaw1 = self.screen_to_angles(pos1)
        pitch2, yaw2 = self.screen_to_angles(pos2)
        
        # Convert to 3D vectors
        v1 = np.array([
            np.sin(yaw1) * np.cos(pitch1),
            np.sin(pitch1),
            np.cos(yaw1) * np.cos(pitch1)
        ])
        v2 = np.array([
            np.sin(yaw2) * np.cos(pitch2),
            np.sin(pitch2),
            np.cos(yaw2) * np.cos(pitch2)
        ])
        
        # Angular difference
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        
        return np.degrees(angle_rad)
        
    def pixels_to_cm(self, pixels: float) -> float:
        """Convert pixels to centimeters."""
        return pixels / self.config.pixels_per_cm
        
    def cm_to_pixels(self, cm: float) -> float:
        """Convert centimeters to pixels."""
        return cm * self.config.pixels_per_cm
        
    def pixels_to_visual_angle(self, pixels: float) -> float:
        """
        Convert pixel distance to visual angle.
        
        Args:
            pixels: Distance in pixels
            
        Returns:
            Visual angle in degrees
        """
        cm = self.pixels_to_cm(pixels)
        return np.degrees(np.arctan2(cm, self.config.viewing_distance_cm))
        
    def visual_angle_to_pixels(self, degrees: float) -> float:
        """
        Convert visual angle to pixel distance.
        
        Args:
            degrees: Visual angle in degrees
            
        Returns:
            Distance in pixels
        """
        cm = self.config.viewing_distance_cm * np.tan(np.radians(degrees))
        return self.cm_to_pixels(cm)

