"""
Heatmap Exporter - Save heatmaps in various formats.

Supports PNG, NPY, and JSON metadata export.
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .accumulator import HeatmapAccumulator
from .renderer import HeatmapRenderer


class HeatmapExporter:
    """
    Export heatmaps in various formats with metadata.
    """
    
    def __init__(
        self,
        output_dir: str = "./data/heatmaps",
        renderer: Optional[HeatmapRenderer] = None
    ):
        """
        Initialize exporter.
        
        Args:
            output_dir: Default output directory
            renderer: HeatmapRenderer for visual exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.renderer = renderer or HeatmapRenderer()
        
    def export(
        self,
        accumulator: HeatmapAccumulator,
        name: str,
        formats: list = ['png', 'npy', 'json'],
        background: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Path]:
        """
        Export heatmap in multiple formats.
        
        Args:
            accumulator: HeatmapAccumulator to export
            name: Base filename (without extension)
            formats: List of formats to export ('png', 'npy', 'json', 'overlay')
            background: Background image for overlay export
            metadata: Additional metadata to include
            
        Returns:
            Dictionary of format -> output path
        """
        output_paths = {}
        
        # Get heatmap data
        heatmap = accumulator.get_heatmap(normalize=True)
        full_heatmap = accumulator.get_full_resolution_heatmap(normalize=True)
        
        # Build metadata
        export_metadata = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'screen_size': {
                'width': accumulator.width,
                'height': accumulator.height
            },
            'grid_size': {
                'width': accumulator.grid_width,
                'height': accumulator.grid_height
            },
            'parameters': {
                'sigma': accumulator.sigma,
                'downsample': accumulator.downsample,
                'decay_rate': accumulator.decay_rate,
            },
            'statistics': accumulator.get_statistics(),
        }
        
        if metadata:
            export_metadata.update(metadata)
            
        # Export each format
        for fmt in formats:
            if fmt == 'png':
                path = self._export_png(full_heatmap, name)
                output_paths['png'] = path
                
            elif fmt == 'npy':
                path = self._export_npy(heatmap, name)
                output_paths['npy'] = path
                
            elif fmt == 'json':
                path = self._export_json(export_metadata, name)
                output_paths['json'] = path
                
            elif fmt == 'overlay' and background is not None:
                path = self._export_overlay(full_heatmap, background, name)
                output_paths['overlay'] = path
                
            elif fmt == 'full_npy':
                path = self._export_npy(full_heatmap, name + '_full')
                output_paths['full_npy'] = path
                
        return output_paths
        
    def _export_png(self, heatmap: np.ndarray, name: str) -> Path:
        """Export as colored PNG image."""
        path = self.output_dir / f"{name}_heatmap.png"
        
        # Render to color
        colored = self.renderer.render(heatmap)
        cv2.imwrite(str(path), colored)
        
        return path
        
    def _export_npy(self, heatmap: np.ndarray, name: str) -> Path:
        """Export as numpy array."""
        path = self.output_dir / f"{name}.npy"
        np.save(str(path), heatmap)
        return path
        
    def _export_json(self, metadata: dict, name: str) -> Path:
        """Export metadata as JSON."""
        path = self.output_dir / f"{name}_metadata.json"
        
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        return path
        
    def _export_overlay(
        self,
        heatmap: np.ndarray,
        background: np.ndarray,
        name: str
    ) -> Path:
        """Export heatmap overlaid on background."""
        path = self.output_dir / f"{name}_overlay.png"
        
        overlaid = self.renderer.overlay_transparent(heatmap, background)
        cv2.imwrite(str(path), overlaid)
        
        return path
        
    def export_session(
        self,
        accumulator: HeatmapAccumulator,
        session_id: str,
        gaze_data: list,
        screenshot: Optional[np.ndarray] = None,
        config: Optional[dict] = None
    ) -> Path:
        """
        Export complete session data.
        
        Creates a session directory with all relevant files:
        - metadata.json: Session info
        - gaze_data.csv: Raw gaze timeseries
        - heatmap.npy: Accumulated heatmap
        - heatmap.png: Visual heatmap
        - heatmap_overlay.png: Heatmap on screenshot (if provided)
        - screenshot.png: Screen capture (if provided)
        
        Args:
            accumulator: HeatmapAccumulator
            session_id: Session identifier
            gaze_data: List of gaze samples
            screenshot: Screen capture image
            config: Session configuration
            
        Returns:
            Session directory path
        """
        session_dir = self.output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Export heatmap
        heatmap = accumulator.get_heatmap(normalize=True)
        full_heatmap = accumulator.get_full_resolution_heatmap(normalize=True)
        
        np.save(session_dir / "heatmap.npy", heatmap)
        
        colored = self.renderer.render(full_heatmap)
        cv2.imwrite(str(session_dir / "heatmap.png"), colored)
        
        # Export screenshot and overlay
        if screenshot is not None:
            cv2.imwrite(str(session_dir / "screenshot.png"), screenshot)
            
            overlaid = self.renderer.overlay_transparent(full_heatmap, screenshot)
            cv2.imwrite(str(session_dir / "heatmap_overlay.png"), overlaid)
            
        # Export gaze data as CSV
        if gaze_data:
            self._export_gaze_csv(gaze_data, session_dir / "gaze_data.csv")
            
        # Export metadata
        metadata = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'screen_resolution': [accumulator.width, accumulator.height],
            'config': config or {},
            'statistics': accumulator.get_statistics(),
        }
        
        with open(session_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return session_dir
        
    def _export_gaze_csv(self, gaze_data: list, path: Path):
        """Export gaze data as CSV."""
        import csv
        
        if not gaze_data:
            return
            
        # Infer columns from first sample
        sample = gaze_data[0]
        if isinstance(sample, dict):
            columns = list(sample.keys())
        else:
            # Assume tuple format
            columns = ['timestamp_ms', 'gaze_x', 'gaze_y', 'smoothed_x', 'smoothed_y',
                      'gaze_pitch', 'gaze_yaw', 'confidence', 'is_fixation']
            columns = columns[:len(sample)]
            
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            
            for row in gaze_data:
                if isinstance(row, dict):
                    writer.writerow([row.get(col, '') for col in columns])
                else:
                    writer.writerow(row)
                    
    def load_session(self, session_id: str) -> dict:
        """
        Load session data.
        
        Returns:
            Dictionary with session data
        """
        session_dir = self.output_dir / session_id
        
        if not session_dir.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
            
        data = {}
        
        # Load metadata
        metadata_path = session_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data['metadata'] = json.load(f)
                
        # Load heatmap
        heatmap_path = session_dir / "heatmap.npy"
        if heatmap_path.exists():
            data['heatmap'] = np.load(str(heatmap_path))
            
        # Load screenshot
        screenshot_path = session_dir / "screenshot.png"
        if screenshot_path.exists():
            data['screenshot'] = cv2.imread(str(screenshot_path))
            
        # Load gaze data
        gaze_path = session_dir / "gaze_data.csv"
        if gaze_path.exists():
            import csv
            with open(gaze_path, 'r') as f:
                reader = csv.DictReader(f)
                data['gaze_data'] = list(reader)
                
        return data

