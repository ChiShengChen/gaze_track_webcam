"""
Geometry utilities for 3D→2D projection and angle calculations.
"""

import numpy as np
from typing import Tuple


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles (roll, pitch, yaw).
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
        
    return roll, pitch, yaw


def gaze_to_screen_angle(
    gaze_vector: np.ndarray,
    screen_center: Tuple[int, int],
    viewing_distance_px: float
) -> Tuple[float, float]:
    """
    Convert 3D gaze vector to screen-space angles.
    
    Coordinate system:
    - X: right (positive)
    - Y: down (positive)
    - Z: into screen (positive)
    
    Args:
        gaze_vector: 3D gaze direction vector
        screen_center: Screen center coordinates (x, y)
        viewing_distance_px: Viewing distance in pixels
        
    Returns:
        Tuple of (pitch, yaw) in radians
    """
    # Normalize gaze vector
    gaze_norm = normalize_vector(gaze_vector)
    
    # Extract angles
    # yaw = horizontal angle, pitch = vertical angle
    yaw = np.arctan2(gaze_norm[0], gaze_norm[2])    # horizontal
    pitch = np.arctan2(gaze_norm[1], gaze_norm[2])  # vertical
    
    return pitch, yaw


def calculate_angular_error(
    target_pos: Tuple[int, int],
    predicted_pos: Tuple[int, int],
    screen_center: Tuple[int, int],
    viewing_distance_cm: float,
    pixels_per_cm: float
) -> float:
    """
    Calculate angular error between target and predicted gaze points.
    
    Args:
        target_pos: Target screen position (x, y)
        predicted_pos: Predicted gaze position (x, y)
        screen_center: Screen center coordinates (x, y)
        viewing_distance_cm: Viewing distance in centimeters
        pixels_per_cm: Pixels per centimeter (DPI / 2.54)
        
    Returns:
        Angular error in degrees
    """
    # Convert positions to cm from screen center
    target_cm = np.array([
        (target_pos[0] - screen_center[0]) / pixels_per_cm,
        (target_pos[1] - screen_center[1]) / pixels_per_cm
    ])
    
    predicted_cm = np.array([
        (predicted_pos[0] - screen_center[0]) / pixels_per_cm,
        (predicted_pos[1] - screen_center[1]) / pixels_per_cm
    ])
    
    # Create 3D vectors (assuming eye at center, viewing_distance back)
    target_3d = np.array([target_cm[0], target_cm[1], viewing_distance_cm])
    predicted_3d = np.array([predicted_cm[0], predicted_cm[1], viewing_distance_cm])
    
    # Normalize
    target_3d = normalize_vector(target_3d)
    predicted_3d = normalize_vector(predicted_3d)
    
    # Angular error
    cos_angle = np.clip(np.dot(target_3d, predicted_3d), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    
    return np.degrees(angle_rad)


def screen_to_angle(
    screen_pos: Tuple[int, int],
    screen_center: Tuple[int, int],
    viewing_distance_cm: float,
    pixels_per_cm: float
) -> Tuple[float, float]:
    """
    Convert screen position to visual angle from center.
    
    Args:
        screen_pos: Screen position (x, y)
        screen_center: Screen center coordinates (x, y)
        viewing_distance_cm: Viewing distance in centimeters
        pixels_per_cm: Pixels per centimeter
        
    Returns:
        Tuple of (horizontal_angle, vertical_angle) in degrees
    """
    dx_cm = (screen_pos[0] - screen_center[0]) / pixels_per_cm
    dy_cm = (screen_pos[1] - screen_center[1]) / pixels_per_cm
    
    h_angle = np.degrees(np.arctan2(dx_cm, viewing_distance_cm))
    v_angle = np.degrees(np.arctan2(dy_cm, viewing_distance_cm))
    
    return h_angle, v_angle


def angle_to_screen(
    h_angle: float,
    v_angle: float,
    screen_center: Tuple[int, int],
    viewing_distance_cm: float,
    pixels_per_cm: float
) -> Tuple[int, int]:
    """
    Convert visual angles to screen position.
    
    Args:
        h_angle: Horizontal angle in degrees
        v_angle: Vertical angle in degrees
        screen_center: Screen center coordinates (x, y)
        viewing_distance_cm: Viewing distance in centimeters
        pixels_per_cm: Pixels per centimeter
        
    Returns:
        Screen position (x, y)
    """
    dx_cm = viewing_distance_cm * np.tan(np.radians(h_angle))
    dy_cm = viewing_distance_cm * np.tan(np.radians(v_angle))
    
    x = int(screen_center[0] + dx_cm * pixels_per_cm)
    y = int(screen_center[1] + dy_cm * pixels_per_cm)
    
    return x, y

