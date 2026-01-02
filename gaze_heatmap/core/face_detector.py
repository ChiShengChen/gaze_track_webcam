"""
Face Detector - MediaPipe Face Mesh wrapper for face and iris detection.
"""

import numpy as np
import cv2
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class FaceDetectionResult:
    """Container for face detection results."""
    landmarks: np.ndarray            # (468, 2) face landmarks
    landmarks_3d: np.ndarray         # (468, 3) 3D face landmarks
    left_eye_landmarks: np.ndarray   # Left eye landmark points
    right_eye_landmarks: np.ndarray  # Right eye landmark points
    left_iris: np.ndarray            # (5, 2) left iris landmarks
    right_iris: np.ndarray           # (5, 2) right iris landmarks
    face_bbox: Tuple[int, int, int, int]  # (x, y, w, h) face bounding box
    confidence: float


class FaceDetector:
    """
    MediaPipe Face Mesh detector with 468 landmarks and iris tracking.
    """
    
    # Key landmark indices (these are just integers, safe to define here)
    NOSE_TIP = 1
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    
    # Iris landmark indices (when refine_landmarks=True)
    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
    
    # Eye contour indices
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    def __init__(
        self,
        static_mode: bool = False,
        max_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7
    ):
        """
        Initialize face detector.
        
        Args:
            static_mode: If True, treats each image independently
            max_faces: Maximum number of faces to detect
            refine_landmarks: Enable iris landmark detection
            min_detection_confidence: Detection confidence threshold
            min_tracking_confidence: Tracking confidence threshold
        """
        # Initialize face mesh connections (must be done at runtime)
        self.FACE_OVAL = mp.solutions.face_mesh.FACEMESH_FACE_OVAL
        self.LEFT_EYE_CONN = mp.solutions.face_mesh.FACEMESH_LEFT_EYE
        self.RIGHT_EYE_CONN = mp.solutions.face_mesh.FACEMESH_RIGHT_EYE
        self.LEFT_IRIS_CONN = mp.solutions.face_mesh.FACEMESH_LEFT_IRIS
        self.RIGHT_IRIS_CONN = mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.refine_landmarks = refine_landmarks
        
    def detect(self, frame: np.ndarray) -> Optional[FaceDetectionResult]:
        """
        Detect face and extract landmarks.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            FaceDetectionResult or None if no face detected
        """
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract 2D landmarks
        landmarks_2d = np.array([
            [lm.x * w, lm.y * h] 
            for lm in face_landmarks.landmark
        ])
        
        # Extract 3D landmarks
        landmarks_3d = np.array([
            [lm.x * w, lm.y * h, lm.z * w]  # z scaled by width
            for lm in face_landmarks.landmark
        ])
        
        # Extract eye landmarks
        left_eye_landmarks = landmarks_2d[self.LEFT_EYE_INDICES]
        right_eye_landmarks = landmarks_2d[self.RIGHT_EYE_INDICES]
        
        # Extract iris landmarks
        if self.refine_landmarks and len(face_landmarks.landmark) > 468:
            left_iris = landmarks_2d[self.LEFT_IRIS_INDICES]
            right_iris = landmarks_2d[self.RIGHT_IRIS_INDICES]
        else:
            # Fallback: estimate iris from eye center
            left_iris = np.array([left_eye_landmarks.mean(axis=0)])
            right_iris = np.array([right_eye_landmarks.mean(axis=0)])
            
        # Calculate face bounding box
        x_coords = landmarks_2d[:, 0]
        y_coords = landmarks_2d[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        return FaceDetectionResult(
            landmarks=landmarks_2d,
            landmarks_3d=landmarks_3d,
            left_eye_landmarks=left_eye_landmarks,
            right_eye_landmarks=right_eye_landmarks,
            left_iris=left_iris,
            right_iris=right_iris,
            face_bbox=bbox,
            confidence=1.0  # MediaPipe doesn't expose confidence
        )
        
    def get_iris_center(self, result: FaceDetectionResult) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get iris centers from detection result.
        
        Returns:
            Tuple of (left_iris_center, right_iris_center)
        """
        left_center = result.left_iris.mean(axis=0)
        right_center = result.right_iris.mean(axis=0)
        return left_center, right_center
        
    def get_eye_aspect_ratio(self, result: FaceDetectionResult) -> Tuple[float, float]:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        
        Returns:
            Tuple of (left_ear, right_ear)
        """
        def calculate_ear(eye_landmarks: np.ndarray) -> float:
            # Vertical distances
            v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            # Horizontal distance
            h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            ear = (v1 + v2) / (2.0 * h + 1e-6)
            return ear
            
        # Use subset of eye landmarks for EAR calculation
        left_ear = calculate_ear(result.left_eye_landmarks[:6])
        right_ear = calculate_ear(result.right_eye_landmarks[:6])
        
        return left_ear, right_ear
        
    def draw_landmarks(
        self,
        frame: np.ndarray,
        result: FaceDetectionResult,
        draw_face: bool = True,
        draw_eyes: bool = True,
        draw_iris: bool = True
    ) -> np.ndarray:
        """
        Draw detected landmarks on frame.
        
        Args:
            frame: BGR image
            result: Detection result
            draw_face: Draw face oval
            draw_eyes: Draw eye contours
            draw_iris: Draw iris points
            
        Returns:
            Frame with drawn landmarks
        """
        output = frame.copy()
        
        if draw_face:
            # Draw face oval
            for connection in self.FACE_OVAL:
                pt1 = tuple(result.landmarks[connection[0]].astype(int))
                pt2 = tuple(result.landmarks[connection[1]].astype(int))
                cv2.line(output, pt1, pt2, (200, 200, 200), 1)
                
        if draw_eyes:
            # Draw eye contours
            for idx in self.LEFT_EYE_INDICES:
                pt = tuple(result.landmarks[idx].astype(int))
                cv2.circle(output, pt, 2, (0, 255, 0), -1)
            for idx in self.RIGHT_EYE_INDICES:
                pt = tuple(result.landmarks[idx].astype(int))
                cv2.circle(output, pt, 2, (0, 255, 0), -1)
                
        if draw_iris:
            # Draw iris points
            for pt in result.left_iris:
                cv2.circle(output, tuple(pt.astype(int)), 2, (255, 0, 255), -1)
            for pt in result.right_iris:
                cv2.circle(output, tuple(pt.astype(int)), 2, (255, 0, 255), -1)
                
            # Draw iris centers
            left_center, right_center = self.get_iris_center(result)
            cv2.circle(output, tuple(left_center.astype(int)), 4, (0, 0, 255), -1)
            cv2.circle(output, tuple(right_center.astype(int)), 4, (0, 0, 255), -1)
            
        return output
        
    def close(self):
        """Release resources."""
        self.face_mesh.close()

