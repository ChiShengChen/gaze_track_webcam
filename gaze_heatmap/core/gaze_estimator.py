"""
Gaze Estimator - Wrapper for ptgaze library with ETH-XGaze model.

Provides unified interface for gaze estimation with fallback to MediaPipe-based estimation.
"""

import os
# Fix OpenMP duplicate library issue on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2

# Check if ptgaze is available and functional
PTGAZE_AVAILABLE = False
PTGazeEstimator = None
_PTGAZE_ERROR = None

try:
    from ptgaze.gaze_estimator import GazeEstimator as PTGazeEstimator
    PTGAZE_AVAILABLE = True
except (ImportError, ValueError, Exception) as e:
    # ptgaze may be installed but not functional due to dependency issues
    _PTGAZE_ERROR = str(e)
    PTGAZE_AVAILABLE = False

# Check if L2CS is available
L2CS_AVAILABLE = False
L2CSPipeline = None
_L2CS_ERROR = None

try:
    from l2cs import Pipeline as L2CSPipeline
    import torch
    L2CS_AVAILABLE = True
except (ImportError, ValueError, Exception) as e:
    _L2CS_ERROR = str(e)
    L2CS_AVAILABLE = False

import mediapipe as mp


@dataclass
class GazeResult:
    """Container for gaze estimation results."""
    gaze_vector: np.ndarray          # (3,) normalized 3D gaze direction
    gaze_pitch: float                # vertical angle in radians
    gaze_yaw: float                  # horizontal angle in radians
    head_pose: np.ndarray            # (3,) roll, pitch, yaw in radians
    landmarks: Optional[np.ndarray]  # (468, 2) MediaPipe face landmarks
    left_eye_center: np.ndarray      # (2,) left eye center in image coords
    right_eye_center: np.ndarray     # (2,) right eye center in image coords
    confidence: float                # detection confidence score
    

class GazeEstimator:
    """
    Wrapper for gaze estimation with multiple model support.
    
    Supports:
    - L2CS-Net (recommended for better accuracy)
    - ETH-XGaze / MPIIGaze (via ptgaze)
    - MediaPipe (fallback)
    """
    
    def __init__(self, device: str = 'cpu', model: str = 'eth-xgaze'):
        """
        Initialize gaze estimator.
        
        Args:
            device: 'cpu' or 'cuda' for GPU acceleration
            model: 'l2cs' (recommended), 'eth-xgaze', 'mpiigaze', or 'mediapipe'
        """
        self.device = device
        self.model = model.lower()
        self.estimator_type = None
        self.use_ptgaze = False  # Initialize for compatibility
        
        # Try to initialize in order of preference
        if self.model == 'l2cs' and L2CS_AVAILABLE:
            try:
                self._init_l2cs()
            except:
                # L2CS failed, try ETH-XGaze as fallback
                if PTGAZE_AVAILABLE:
                    print("L2CS failed, falling back to ETH-XGaze")
                    self.model = 'eth-xgaze'
                    self._init_ptgaze()
                else:
                    print("L2CS failed, falling back to MediaPipe")
                    self.model = 'mediapipe'
                    self._init_mediapipe()
        elif self.model in ['eth-xgaze', 'mpiigaze'] and PTGAZE_AVAILABLE:
            self._init_ptgaze()
        elif L2CS_AVAILABLE:
            # Fallback to L2CS if available
            print(f"Model '{model}' not available, falling back to L2CS-Net")
            self.model = 'l2cs'
            try:
                self._init_l2cs()
            except:
                # L2CS failed, try ETH-XGaze
                if PTGAZE_AVAILABLE:
                    print("L2CS failed, falling back to ETH-XGaze")
                    self.model = 'eth-xgaze'
                    self._init_ptgaze()
                else:
                    print("L2CS failed, falling back to MediaPipe")
                    self.model = 'mediapipe'
                    self._init_mediapipe()
        elif PTGAZE_AVAILABLE:
            # Fallback to ptgaze if available
            print(f"Model '{model}' not available, falling back to ETH-XGaze")
            self.model = 'eth-xgaze'
            self._init_ptgaze()
        else:
            # Final fallback to MediaPipe
            print(f"Advanced models not available, using MediaPipe fallback")
            self.model = 'mediapipe'
            self._init_mediapipe()
    
    def _init_l2cs(self):
        """Initialize L2CS-Net estimator."""
        if L2CSPipeline is None:
            print("Warning: L2CS not available, falling back to MediaPipe")
            self.estimator_type = None
            self._init_mediapipe()
            return
        
        try:
            import torch
            from pathlib import Path
            
            # Determine device (L2CS requires torch.device object, not string)
            if self.device == 'cuda' and torch.cuda.is_available():
                torch_device = torch.device('cuda')
            elif self.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch_device = torch.device('mps')
            else:
                torch_device = torch.device('cpu')
            
            # Try to find model weights
            # L2CS needs pre-trained weights
            weights_path = None
            # Get the project root (gaze_heatmap directory)
            project_root = Path(__file__).parent.parent
            possible_paths = [
                # User-specified location (highest priority)
                project_root / 'l2cs_model' / 'L2CSNet_gaze360.pkl',
                # Standard locations
                Path.home() / '.l2cs' / 'models' / 'L2CSNet_gaze360.pkl',
                Path.home() / '.l2cs' / 'L2CSNet_gaze360.pkl',
                project_root / 'models' / 'L2CSNet_gaze360.pkl',
                Path('models') / 'L2CSNet_gaze360.pkl',
                Path('L2CSNet_gaze360.pkl'),
            ]
            
            for path in possible_paths:
                if path.exists():
                    weights_path = path  # Keep as Path object
                    break
            
            if weights_path is None:
                # Try to use default weights (L2CS may download automatically)
                # Check if there's a default location
                default_path = Path.home() / '.l2cs' / 'models' / 'L2CSNet_gaze360.pkl'
                if not default_path.parent.exists():
                    default_path.parent.mkdir(parents=True, exist_ok=True)
                weights_path = default_path
                print(f"Warning: L2CS weights not found at standard locations")
                print(f"Will try to download or use: {weights_path}")
                print("If it fails, download weights from: https://github.com/Ahmednull/L2CS-Net")
            
            # Initialize pipeline
            # Note: RetinaFace (face detector) doesn't support MPS, so we need special handling
            # For MPS devices, we'll use CPU for face detection but MPS for model inference
            if torch_device.type == 'mps':
                # RetinaFace doesn't support MPS, so use CPU for face detection
                # The model will still run on MPS for inference
                try:
                    self.estimator = L2CSPipeline(
                        weights=weights_path,
                        arch='ResNet50',
                        device=torch_device,
                        include_detector=True,
                        confidence_threshold=0.5
                    )
                except Exception as e:
                    # If MPS fails (e.g., RetinaFace issue), fall back to CPU
                    print(f"Warning: MPS initialization failed ({e}), using CPU instead")
                    torch_device = torch.device('cpu')
                    self.estimator = L2CSPipeline(
                        weights=weights_path,
                        arch='ResNet50',
                        device=torch_device,
                        include_detector=True,
                        confidence_threshold=0.5
                    )
            else:
                self.estimator = L2CSPipeline(
                    weights=weights_path,
                    arch='ResNet50',
                    device=torch_device,
                    include_detector=True,
                    confidence_threshold=0.5
                )
            
            self.estimator_type = 'l2cs'
            device_str = 'mps' if torch_device.type == 'mps' else ('cuda' if torch_device.type == 'cuda' else 'cpu')
            print(f"✅ 使用 L2CS-Net 模型 (device={device_str})")
            
        except Exception as e:
            print(f"Warning: L2CS initialization failed ({e})")
            print("Will try ETH-XGaze as fallback...")
            self.estimator_type = None
            # Don't initialize MediaPipe here, let the main logic handle fallback
            raise  # Re-raise to let main __init__ handle fallback
            
    def _init_ptgaze(self):
        """Initialize ptgaze estimator."""
        if PTGazeEstimator is None:
            # Fallback to MediaPipe if ptgaze not available
            self.use_ptgaze = False
            self._init_mediapipe()
            return
            
        try:
            from omegaconf import OmegaConf
            import pathlib
            
            # Get ptgaze package path for config
            import ptgaze
            package_root = pathlib.Path(ptgaze.__file__).parent
            
            # Load default config for ETH-XGaze
            config_path = package_root / 'data' / 'configs' / 'eth-xgaze.yaml'
            config = OmegaConf.load(config_path)
            
            # Update config with custom settings
            config.device = self.device
            config.face_detector.mediapipe_max_num_faces = 1
            
            # Resolve ${PACKAGE_ROOT} placeholders and expand ~ paths
            config_str = OmegaConf.to_yaml(config)
            config_str = config_str.replace('${PACKAGE_ROOT}', str(package_root))
            
            # Expand ~ to home directory
            home_dir = os.path.expanduser('~')
            config_str = config_str.replace('~/', f'{home_dir}/')
            
            config = OmegaConf.create(config_str)
            
            self.estimator = PTGazeEstimator(config)
            self.estimator_type = 'ptgaze'
            self.use_ptgaze = True
            print(f"✅ 使用 ETH-XGaze 模型 (device={self.device})")
            
        except Exception as e:
            print(f"Warning: ptgaze initialization failed ({e}), falling back to MediaPipe")
            self.estimator_type = None
            self.use_ptgaze = False
            self._init_mediapipe()
        
    def _init_mediapipe(self):
        """Initialize MediaPipe face mesh for fallback gaze estimation."""
        self.estimator_type = 'mediapipe'
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Landmark indices for eye corners and iris
        self.LEFT_EYE = {
            'outer': 33, 'inner': 133, 'upper': 159, 'lower': 145,
            'iris_center': 468, 'iris_left': 469, 'iris_right': 471
        }
        self.RIGHT_EYE = {
            'inner': 362, 'outer': 263, 'upper': 386, 'lower': 374,
            'iris_center': 473, 'iris_left': 474, 'iris_right': 476
        }
        
        # Iris landmark indices (from refine_landmarks)
        self._iris_indices = list(range(468, 478))
        
    def estimate(self, frame: np.ndarray) -> Optional[GazeResult]:
        """
        Process single frame and return gaze data.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            GazeResult with gaze data, or None if no face detected
        """
        if self.estimator_type == 'l2cs':
            return self._estimate_l2cs(frame)
        elif self.estimator_type == 'ptgaze' or self.use_ptgaze:
            return self._estimate_ptgaze(frame)
        else:
            return self._estimate_mediapipe(frame)
    
    def _estimate_l2cs(self, frame: np.ndarray) -> Optional[GazeResult]:
        """Estimate gaze using L2CS-Net."""
        try:
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # L2CS Pipeline includes face detection, just call step
            try:
                results = self.estimator.step(frame_rgb)
            except (ValueError, IndexError) as e:
                # No face detected or other processing error
                return None
            
            # Check if any faces were detected
            if results.pitch is None or len(results.pitch) == 0:
                return None
            
            # Get first detected face (pitch and yaw are arrays)
            pitch_deg = float(results.pitch[0])  # Convert to scalar
            yaw_deg = float(results.yaw[0])
            
            # Convert to radians
            pitch = np.radians(pitch_deg)
            yaw = np.radians(yaw_deg)
            
            # Convert angles to 3D gaze vector
            gaze_vector = np.array([
                np.sin(yaw),
                np.sin(pitch),
                np.cos(yaw) * np.cos(pitch)
            ])
            gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
            
            # Estimate head pose using MediaPipe (L2CS doesn't provide head pose)
            head_pose = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
            try:
                # Use MediaPipe to estimate head pose from face landmarks
                if not hasattr(self, '_mp_face_mesh'):
                    import mediapipe as mp
                    self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=False,
                        min_detection_confidence=0.5
                    )
                
                mp_results = self._mp_face_mesh.process(frame_rgb)
                if mp_results.multi_face_landmarks:
                    # Use face detector to estimate head pose
                    landmarks_3d = []
                    landmarks_2d = []
                    face_landmarks = mp_results.multi_face_landmarks[0]
                    
                    # Key face landmarks for head pose estimation (nose tip, chin, etc.)
                    # MediaPipe face mesh indices
                    key_indices = [1, 33, 61, 199, 291, 4, 51, 175, 2, 152]  # nose, eyes, mouth corners
                    
                    for idx in key_indices:
                        if idx < len(face_landmarks.landmark):
                            lm = face_landmarks.landmark[idx]
                            landmarks_2d.append([lm.x * w, lm.y * h])
                            # Use approximate 3D coordinates (normalized)
                            landmarks_3d.append([lm.x, lm.y, lm.z])
                    
                    if len(landmarks_2d) >= 4:
                        # Use PnP to estimate head pose
                        # Approximate 3D face model (normalized coordinates)
                        face_3d_model = np.array(landmarks_3d, dtype=np.float32)
                        face_2d = np.array(landmarks_2d, dtype=np.float32)
                        
                        # Camera matrix (approximate, using frame dimensions)
                        focal_length = w
                        center = (w / 2, h / 2)
                        camera_matrix = np.array([
                            [focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]
                        ], dtype=np.float32)
                        dist_coeffs = np.zeros((4, 1))
                        
                        # Solve PnP
                        success, rotation_vec, translation_vec = cv2.solvePnP(
                            face_3d_model, face_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                        )
                        
                        if success:
                            # Convert rotation vector to Euler angles
                            rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
                            # Extract Euler angles (ZYX convention: yaw, pitch, roll)
                            sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
                            singular = sy < 1e-6
                            
                            if not singular:
                                head_roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                                head_pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                                head_yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                            else:
                                head_roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                                head_pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                                head_yaw = 0
                            
                            head_pose = np.array([head_roll, head_pitch, head_yaw])
            except Exception as e:
                # If head pose estimation fails, use zero pose
                pass
            
            # Get bbox and landmarks from L2CS results if available
            bbox = None
            landmarks = None
            if results.bboxes is not None and len(results.bboxes) > 0:
                bbox = results.bboxes[0]  # [x, y, w, h] or similar format
            
            if results.landmarks is not None and len(results.landmarks) > 0:
                landmarks = results.landmarks[0]  # May be face landmarks
            
            # Get eye centers
            if bbox is not None:
                # Use bbox to estimate eye centers
                x, y = int(bbox[0]), int(bbox[1])
                face_w, face_h = int(bbox[2]), int(bbox[3]) if len(bbox) > 3 else face_w
                eye_y = y + int(face_h * 0.4)
                left_eye_center = np.array([x + int(face_w * 0.3), eye_y])
                right_eye_center = np.array([x + int(face_w * 0.7), eye_y])
            else:
                # Fallback: use MediaPipe for landmarks
                face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.5
                )
                mesh_results = face_mesh.process(frame_rgb)
                if mesh_results.multi_face_landmarks:
                    landmarks_list = mesh_results.multi_face_landmarks[0].landmark
                    landmarks = np.array([[lm.x * w, lm.y * h] for lm in landmarks_list])
                    if len(landmarks) >= 468:
                        left_eye_center = (landmarks[33] + landmarks[133]) / 2
                        right_eye_center = (landmarks[362] + landmarks[263]) / 2
                    else:
                        # Final fallback: center of frame
                        left_eye_center = np.array([w * 0.4, h * 0.5])
                        right_eye_center = np.array([w * 0.6, h * 0.5])
                else:
                    # Final fallback: center of frame
                    left_eye_center = np.array([w * 0.4, h * 0.5])
                    right_eye_center = np.array([w * 0.6, h * 0.5])
            
            # Get confidence from scores if available
            confidence = 0.9
            if results.scores is not None and len(results.scores) > 0:
                confidence = float(results.scores[0])
            
            return GazeResult(
                gaze_vector=gaze_vector,
                gaze_pitch=pitch,
                gaze_yaw=yaw,
                head_pose=head_pose,
                landmarks=landmarks,
                left_eye_center=left_eye_center,
                right_eye_center=right_eye_center,
                confidence=confidence
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
            
    def _estimate_ptgaze(self, frame: np.ndarray) -> Optional[GazeResult]:
        """Estimate gaze using ptgaze."""
        try:
            h, w = frame.shape[:2]
            
            # Undistort image using camera calibration
            undistorted = cv2.undistort(
                frame, 
                self.estimator.camera.camera_matrix,
                self.estimator.camera.dist_coefficients
            )
            
            # Detect faces
            faces = self.estimator.detect_faces(undistorted)
            
            if not faces:
                return None
                
            # Use first detected face
            face = faces[0]
            
            # Estimate gaze for this face
            self.estimator.estimate_gaze(undistorted, face)
            
            # Extract gaze vector and angles
            gaze_vector = face.gaze_vector
            if gaze_vector is None:
                return None
                
            # Get angles from normalized_gaze_angles (pitch, yaw)
            if face.normalized_gaze_angles is not None:
                gaze_pitch, gaze_yaw = face.normalized_gaze_angles
            else:
                gaze_pitch = np.arcsin(-gaze_vector[1])
                gaze_yaw = np.arctan2(-gaze_vector[0], -gaze_vector[2])
            
            # Extract head pose
            if face.head_pose_rot is not None:
                head_pose = face.head_pose_rot.as_euler('xyz')
            else:
                head_pose = np.array([0.0, 0.0, 0.0])
            
            # Get eye centers from face landmarks (indices for mediapipe 468-point model)
            # Note: ptgaze uses 468-point model without iris landmarks (468-477)
            # Use eye corner landmarks instead: left=33, right=263
            landmarks = face.landmarks
            if landmarks is not None and len(landmarks) >= 68:
                # MediaPipe 468-point model eye indices
                # Left eye center from landmarks 33, 133
                # Right eye center from landmarks 362, 263
                left_eye_center = (landmarks[33] + landmarks[133]) / 2
                right_eye_center = (landmarks[362] + landmarks[263]) / 2
            else:
                # Approximate from face bbox
                bbox = face.bbox
                left_eye_center = np.array([bbox[0, 0] + (bbox[1, 0] - bbox[0, 0]) * 0.3, 
                                           bbox[0, 1] + (bbox[1, 1] - bbox[0, 1]) * 0.3])
                right_eye_center = np.array([bbox[0, 0] + (bbox[1, 0] - bbox[0, 0]) * 0.7, 
                                            bbox[0, 1] + (bbox[1, 1] - bbox[0, 1]) * 0.3])
            
            return GazeResult(
                gaze_vector=gaze_vector,
                gaze_pitch=gaze_pitch,
                gaze_yaw=gaze_yaw,
                head_pose=head_pose,
                landmarks=landmarks,
                left_eye_center=left_eye_center,
                right_eye_center=right_eye_center,
                confidence=1.0
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
            
    def _estimate_mediapipe(self, frame: np.ndarray) -> Optional[GazeResult]:
        """Estimate gaze using MediaPipe face mesh with iris tracking."""
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Extract all landmarks as numpy array
        all_landmarks = np.array([
            [lm.x * w, lm.y * h] for lm in landmarks
        ])
        
        # Get iris centers
        try:
            left_iris = np.array([
                landmarks[self.LEFT_EYE['iris_center']].x * w,
                landmarks[self.LEFT_EYE['iris_center']].y * h
            ])
            right_iris = np.array([
                landmarks[self.RIGHT_EYE['iris_center']].x * w,
                landmarks[self.RIGHT_EYE['iris_center']].y * h
            ])
        except (IndexError, AttributeError):
            # Fallback to eye center calculation
            left_iris = self._calculate_eye_center(landmarks, self.LEFT_EYE, w, h)
            right_iris = self._calculate_eye_center(landmarks, self.RIGHT_EYE, w, h)
            
        # Calculate eye boxes for normalization
        left_box = self._get_eye_box(landmarks, self.LEFT_EYE, w, h)
        right_box = self._get_eye_box(landmarks, self.RIGHT_EYE, w, h)
        
        # Estimate gaze direction from iris position relative to eye socket
        gaze_vector, gaze_pitch, gaze_yaw = self._compute_gaze_from_iris(
            left_iris, right_iris, left_box, right_box
        )
        
        # Estimate head pose from facial landmarks
        head_pose = self._estimate_head_pose(landmarks, w, h)
        
        return GazeResult(
            gaze_vector=gaze_vector,
            gaze_pitch=gaze_pitch,
            gaze_yaw=gaze_yaw,
            head_pose=head_pose,
            landmarks=all_landmarks,
            left_eye_center=left_iris,
            right_eye_center=right_iris,
            confidence=0.9  # MediaPipe doesn't provide confidence for iris
        )
        
    def _calculate_eye_center(self, landmarks, eye_indices: dict, w: int, h: int) -> np.ndarray:
        """Calculate eye center from corner landmarks."""
        inner = np.array([landmarks[eye_indices['inner']].x * w,
                         landmarks[eye_indices['inner']].y * h])
        outer = np.array([landmarks[eye_indices['outer']].x * w,
                         landmarks[eye_indices['outer']].y * h])
        upper = np.array([landmarks[eye_indices['upper']].x * w,
                         landmarks[eye_indices['upper']].y * h])
        lower = np.array([landmarks[eye_indices['lower']].x * w,
                         landmarks[eye_indices['lower']].y * h])
        return (inner + outer + upper + lower) / 4
        
    def _get_eye_box(self, landmarks, eye_indices: dict, w: int, h: int) -> dict:
        """Get eye bounding box coordinates."""
        return {
            'inner': np.array([landmarks[eye_indices['inner']].x * w,
                              landmarks[eye_indices['inner']].y * h]),
            'outer': np.array([landmarks[eye_indices['outer']].x * w,
                              landmarks[eye_indices['outer']].y * h]),
            'upper': np.array([landmarks[eye_indices['upper']].x * w,
                              landmarks[eye_indices['upper']].y * h]),
            'lower': np.array([landmarks[eye_indices['lower']].x * w,
                              landmarks[eye_indices['lower']].y * h]),
        }
        
    def _compute_gaze_from_iris(
        self,
        left_iris: np.ndarray,
        right_iris: np.ndarray,
        left_box: dict,
        right_box: dict
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute gaze direction from iris positions relative to eye sockets.
        
        Returns normalized gaze vector and pitch/yaw angles.
        """
        # Calculate normalized iris position within each eye
        def normalize_iris(iris: np.ndarray, box: dict) -> np.ndarray:
            eye_width = max(1e-3, np.linalg.norm(box['outer'] - box['inner']))
            eye_height = max(1e-3, np.linalg.norm(box['upper'] - box['lower']))
            
            # Position relative to eye center
            eye_center = (box['inner'] + box['outer']) / 2
            dx = (iris[0] - eye_center[0]) / eye_width
            dy = (iris[1] - eye_center[1]) / eye_height
            
            return np.array([dx, dy])
            
        left_norm = normalize_iris(left_iris, left_box)
        right_norm = normalize_iris(right_iris, right_box)
        
        # Average both eyes
        avg_norm = (left_norm + right_norm) / 2
        
        # Convert to angles (approximate mapping)
        # Typical eye can rotate ~30 degrees each direction
        max_angle = np.radians(30)
        yaw = avg_norm[0] * max_angle * 2    # horizontal
        pitch = avg_norm[1] * max_angle * 2  # vertical
        
        # Construct 3D gaze vector
        gaze_vector = np.array([
            np.sin(yaw),
            np.sin(pitch),
            np.cos(yaw) * np.cos(pitch)
        ])
        gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
        
        return gaze_vector, pitch, yaw
        
    def _estimate_head_pose(self, landmarks, w: int, h: int) -> np.ndarray:
        """
        Estimate head pose from facial landmarks using PnP.
        
        Returns (roll, pitch, yaw) in radians.
        """
        # 6-point model for head pose estimation
        # Nose tip, chin, left/right eye corners, left/right mouth corners
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # Corresponding 2D points from MediaPipe landmarks
        image_points = np.array([
            [landmarks[1].x * w, landmarks[1].y * h],     # Nose tip
            [landmarks[152].x * w, landmarks[152].y * h], # Chin
            [landmarks[33].x * w, landmarks[33].y * h],   # Left eye left corner
            [landmarks[263].x * w, landmarks[263].y * h], # Right eye right corner
            [landmarks[61].x * w, landmarks[61].y * h],   # Left mouth corner
            [landmarks[291].x * w, landmarks[291].y * h]  # Right mouth corner
        ], dtype=np.float64)
        
        # Camera matrix (approximate)
        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return np.array([0.0, 0.0, 0.0])
            
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Extract Euler angles
        try:
            from ..utils.geometry import rotation_matrix_to_euler
        except ImportError:
            from utils.geometry import rotation_matrix_to_euler
        roll, pitch, yaw = rotation_matrix_to_euler(rotation_matrix)
        
        return np.array([roll, pitch, yaw])
        
    def close(self):
        """Release resources."""
        # Close MediaPipe face mesh if it was initialized
        if self.estimator_type == 'mediapipe' and hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        # L2CS and ptgaze don't need explicit cleanup

