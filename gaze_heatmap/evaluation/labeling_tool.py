"""
Labeling Tool - Manual ground-truth annotation interface.

Provides UI for annotating Areas of Interest (AOI) on heatmaps
and evaluating attention distribution accuracy.
"""

import numpy as np
import cv2
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime


@dataclass
class AOIAnnotation:
    """Area of Interest annotation."""
    aoi_id: str
    type: str  # 'rectangle', 'polygon', 'point'
    coords: List[Tuple[int, int]]
    expected_attention: float  # 0-1 scale
    actual_attention: float = 0.0
    correct: bool = False
    label: str = ""
    
    def to_dict(self) -> dict:
        return {
            'aoi_id': self.aoi_id,
            'type': self.type,
            'coords': self.coords,
            'expected_attention': self.expected_attention,
            'actual_attention': self.actual_attention,
            'correct': self.correct,
            'label': self.label,
        }


@dataclass
class SessionAnnotation:
    """Complete session annotation."""
    session_id: str
    stimulus: str
    heatmap_path: str
    annotations: List[AOIAnnotation] = field(default_factory=list)
    overall_score: float = 0.0
    annotator: str = ""
    timestamp: str = ""
    notes: str = ""
    
    def to_dict(self) -> dict:
        return {
            'session_id': self.session_id,
            'stimulus': self.stimulus,
            'heatmap_path': self.heatmap_path,
            'annotations': [a.to_dict() for a in self.annotations],
            'overall_score': self.overall_score,
            'annotator': self.annotator,
            'timestamp': self.timestamp,
            'notes': self.notes,
        }
        
    def save(self, path: str):
        """Save annotation to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, path: str) -> 'SessionAnnotation':
        """Load annotation from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
            
        annotations = [
            AOIAnnotation(**a) for a in data.get('annotations', [])
        ]
        
        return cls(
            session_id=data['session_id'],
            stimulus=data['stimulus'],
            heatmap_path=data['heatmap_path'],
            annotations=annotations,
            overall_score=data.get('overall_score', 0.0),
            annotator=data.get('annotator', ''),
            timestamp=data.get('timestamp', ''),
            notes=data.get('notes', ''),
        )


class LabelingTool:
    """
    Manual annotation interface for heatmap evaluation.
    
    Features:
    - Load and display heatmap with stimulus
    - Draw AOI regions (rectangles, polygons)
    - Calculate attention within AOIs
    - Save/load annotations
    """
    
    def __init__(
        self,
        sessions_dir: str,
        output_dir: str,
        attention_threshold: float = 0.5
    ):
        """
        Initialize labeling tool.
        
        Args:
            sessions_dir: Directory containing session data
            output_dir: Directory for saving annotations
            attention_threshold: Threshold for correct/incorrect classification
        """
        self.sessions_dir = Path(sessions_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.attention_threshold = attention_threshold
        
        # Current session data
        self.current_session: Optional[SessionAnnotation] = None
        self.heatmap: Optional[np.ndarray] = None
        self.screenshot: Optional[np.ndarray] = None
        self.overlay: Optional[np.ndarray] = None
        
        # Drawing state
        self.drawing = False
        self.current_points: List[Tuple[int, int]] = []
        self.current_type = 'rectangle'
        
    def load_session(self, session_id: str):
        """
        Load session data for annotation.
        
        Args:
            session_id: Session identifier
        """
        session_dir = self.sessions_dir / session_id
        
        if not session_dir.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
            
        # Load heatmap
        heatmap_path = session_dir / "heatmap.npy"
        if heatmap_path.exists():
            self.heatmap = np.load(str(heatmap_path))
        else:
            raise FileNotFoundError("Heatmap not found")
            
        # Load screenshot
        screenshot_path = session_dir / "screenshot.png"
        if screenshot_path.exists():
            self.screenshot = cv2.imread(str(screenshot_path))
            
        # Load or create overlay
        overlay_path = session_dir / "heatmap_overlay.png"
        if overlay_path.exists():
            self.overlay = cv2.imread(str(overlay_path))
        elif self.screenshot is not None:
            self._create_overlay()
            
        # Check for existing annotation
        annotation_path = self.output_dir / f"{session_id}_annotation.json"
        if annotation_path.exists():
            self.current_session = SessionAnnotation.load(str(annotation_path))
        else:
            self.current_session = SessionAnnotation(
                session_id=session_id,
                stimulus=str(screenshot_path) if screenshot_path.exists() else "",
                heatmap_path=str(heatmap_path),
                timestamp=datetime.now().isoformat()
            )
            
    def _create_overlay(self):
        """Create heatmap overlay on screenshot."""
        if self.screenshot is None or self.heatmap is None:
            return
            
        try:
            from ..heatmap.renderer import HeatmapRenderer
        except ImportError:
            from heatmap.renderer import HeatmapRenderer
        
        renderer = HeatmapRenderer()
        
        # Resize heatmap to match screenshot
        h, w = self.screenshot.shape[:2]
        heatmap_full = cv2.resize(self.heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize
        if heatmap_full.max() > 0:
            heatmap_full = heatmap_full / heatmap_full.max()
            
        self.overlay = renderer.overlay_transparent(heatmap_full, self.screenshot)
        
    def compute_aoi_attention(
        self,
        heatmap: np.ndarray,
        aoi_type: str,
        coords: List[Tuple[int, int]]
    ) -> float:
        """
        Calculate attention score within AOI from heatmap.
        
        Args:
            heatmap: 2D attention heatmap (normalized 0-1)
            aoi_type: 'rectangle' or 'polygon'
            coords: AOI coordinates
            
        Returns:
            Attention score (0-1)
        """
        h, w = heatmap.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if aoi_type == 'rectangle':
            # coords: [(x1, y1), (x2, y2)]
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            
            # Scale to heatmap size
            x1 = int(x1 * w / self.screenshot.shape[1]) if self.screenshot is not None else x1
            y1 = int(y1 * h / self.screenshot.shape[0]) if self.screenshot is not None else y1
            x2 = int(x2 * w / self.screenshot.shape[1]) if self.screenshot is not None else x2
            y2 = int(y2 * h / self.screenshot.shape[0]) if self.screenshot is not None else y2
            
            mask[y1:y2, x1:x2] = 255
            
        elif aoi_type == 'polygon':
            # coords: list of (x, y) points
            pts = np.array(coords, dtype=np.int32)
            
            # Scale to heatmap size
            if self.screenshot is not None:
                scale_x = w / self.screenshot.shape[1]
                scale_y = h / self.screenshot.shape[0]
                pts = (pts * [scale_x, scale_y]).astype(np.int32)
                
            cv2.fillPoly(mask, [pts], 255)
            
        # Calculate attention within AOI
        aoi_attention = heatmap[mask > 0].sum()
        total_attention = heatmap.sum()
        
        if total_attention > 0:
            return float(aoi_attention / total_attention)
        return 0.0
        
    def add_annotation(
        self,
        aoi_id: str,
        aoi_type: str,
        coords: List[Tuple[int, int]],
        expected_attention: float,
        label: str = ""
    ):
        """
        Add AOI annotation to current session.
        
        Args:
            aoi_id: Unique identifier for AOI
            aoi_type: 'rectangle' or 'polygon'
            coords: AOI coordinates
            expected_attention: Expected attention level (0-1)
            label: Optional label/description
        """
        if self.current_session is None or self.heatmap is None:
            raise RuntimeError("No session loaded")
            
        # Calculate actual attention
        actual_attention = self.compute_aoi_attention(
            self.heatmap, aoi_type, coords
        )
        
        # Determine if correct
        correct = actual_attention >= expected_attention * self.attention_threshold
        
        annotation = AOIAnnotation(
            aoi_id=aoi_id,
            type=aoi_type,
            coords=coords,
            expected_attention=expected_attention,
            actual_attention=actual_attention,
            correct=correct,
            label=label
        )
        
        self.current_session.annotations.append(annotation)
        
        # Update overall score
        self._update_overall_score()
        
    def _update_overall_score(self):
        """Update overall session score."""
        if not self.current_session or not self.current_session.annotations:
            return
            
        correct_count = sum(1 for a in self.current_session.annotations if a.correct)
        self.current_session.overall_score = correct_count / len(self.current_session.annotations)
        
    def run_ui(self, window_name: str = "Labeling Tool"):
        """
        Launch annotation interface.
        
        Controls:
        - Left click: Add point
        - Right click: Complete shape
        - 'r': Rectangle mode
        - 'p': Polygon mode
        - 'c': Clear current drawing
        - 's': Save annotations
        - 'q': Quit
        """
        if self.overlay is None:
            print("No overlay available. Load a session first.")
            return
            
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        print("\n=== Labeling Tool Controls ===")
        print("Left click: Add point")
        print("Right click: Complete shape")
        print("'r': Rectangle mode")
        print("'p': Polygon mode")
        print("'c': Clear current drawing")
        print("'s': Save annotations")
        print("'q': Quit")
        print("==============================\n")
        
        while True:
            display = self.overlay.copy()
            
            # Draw existing annotations
            for ann in self.current_session.annotations if self.current_session else []:
                self._draw_annotation(display, ann)
                
            # Draw current shape being drawn
            if len(self.current_points) > 0:
                self._draw_current_shape(display)
                
            # Show mode
            mode_text = f"Mode: {self.current_type.upper()}"
            cv2.putText(display, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                       
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.current_type = 'rectangle'
                self.current_points = []
            elif key == ord('p'):
                self.current_type = 'polygon'
                self.current_points = []
            elif key == ord('c'):
                self.current_points = []
            elif key == ord('s'):
                self.save_annotations()
                print("Annotations saved!")
                
        cv2.destroyWindow(window_name)
        
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            
            if self.current_type == 'rectangle' and len(self.current_points) == 2:
                self._complete_annotation()
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_type == 'polygon' and len(self.current_points) >= 3:
                self._complete_annotation()
                
    def _complete_annotation(self):
        """Complete current annotation."""
        if not self.current_points:
            return
            
        # Prompt for details
        aoi_id = f"aoi_{len(self.current_session.annotations) + 1}" if self.current_session else "aoi_1"
        
        print(f"\nCompleting {self.current_type} annotation: {aoi_id}")
        
        try:
            expected = float(input("Expected attention (0-1): ") or "0.5")
            label = input("Label (optional): ") or ""
        except ValueError:
            expected = 0.5
            label = ""
            
        self.add_annotation(
            aoi_id=aoi_id,
            aoi_type=self.current_type,
            coords=self.current_points.copy(),
            expected_attention=expected,
            label=label
        )
        
        print(f"Added annotation. Actual attention: {self.current_session.annotations[-1].actual_attention:.2f}")
        
        self.current_points = []
        
    def _draw_annotation(self, image: np.ndarray, annotation: AOIAnnotation):
        """Draw annotation on image."""
        color = (0, 255, 0) if annotation.correct else (0, 0, 255)
        
        if annotation.type == 'rectangle':
            pt1 = annotation.coords[0]
            pt2 = annotation.coords[1]
            cv2.rectangle(image, pt1, pt2, color, 2)
            
        elif annotation.type == 'polygon':
            pts = np.array(annotation.coords, dtype=np.int32)
            cv2.polylines(image, [pts], True, color, 2)
            
        # Draw label
        if annotation.coords:
            label_pos = annotation.coords[0]
            text = f"{annotation.aoi_id}: {annotation.actual_attention:.0%}"
            cv2.putText(image, text, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
    def _draw_current_shape(self, image: np.ndarray):
        """Draw shape currently being drawn."""
        color = (255, 255, 0)
        
        for pt in self.current_points:
            cv2.circle(image, pt, 5, color, -1)
            
        if len(self.current_points) >= 2:
            if self.current_type == 'rectangle':
                cv2.rectangle(image, self.current_points[0],
                            self.current_points[1], color, 2)
            else:
                pts = np.array(self.current_points, dtype=np.int32)
                cv2.polylines(image, [pts], False, color, 2)
                
    def save_annotations(self):
        """Save current annotations."""
        if self.current_session is None:
            return
            
        path = self.output_dir / f"{self.current_session.session_id}_annotation.json"
        self.current_session.save(str(path))
        
    def export_annotations(self, path: Optional[str] = None):
        """Export annotations to specified path."""
        if self.current_session is None:
            return
            
        if path is None:
            path = str(self.output_dir / f"{self.current_session.session_id}_annotation.json")
            
        self.current_session.save(path)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get annotation summary for current session."""
        if self.current_session is None:
            return {}
            
        return {
            'session_id': self.current_session.session_id,
            'num_annotations': len(self.current_session.annotations),
            'overall_score': self.current_session.overall_score,
            'annotations': [a.to_dict() for a in self.current_session.annotations],
        }

