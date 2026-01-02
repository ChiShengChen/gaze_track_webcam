"""
Heatmap Renderer - Visualization and overlay rendering.

Converts heatmap data to visual representations with colormap
and transparency support.
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class HeatmapRenderer:
    """
    Renders heatmaps as visual images with colormaps and overlays.
    """
    
    # Available colormap options
    COLORMAPS = {
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'plasma': cv2.COLORMAP_PLASMA,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'inferno': cv2.COLORMAP_INFERNO,
        'turbo': cv2.COLORMAP_TURBO,
        'rainbow': cv2.COLORMAP_RAINBOW,
    }
    
    def __init__(
        self,
        colormap: str = 'jet',
        min_alpha: float = 0.0,
        max_alpha: float = 0.7,
        gamma: float = 0.7,
        blur_sigma: float = 5.0
    ):
        """
        Initialize renderer.
        
        Args:
            colormap: Colormap name ('jet', 'hot', 'plasma', etc.)
            min_alpha: Minimum alpha for low intensity regions
            max_alpha: Maximum alpha for high intensity regions
            gamma: Gamma correction for intensity (< 1 = brighter)
            blur_sigma: Gaussian blur sigma for smoothing
        """
        self.colormap_name = colormap
        self.colormap = self.COLORMAPS.get(colormap, cv2.COLORMAP_JET)
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.gamma = gamma
        self.blur_sigma = blur_sigma
        
    def render(
        self,
        heatmap: np.ndarray,
        size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Render heatmap to color image.
        
        Args:
            heatmap: 2D float array (normalized 0-1)
            size: Output size (width, height), None = same as input
            
        Returns:
            BGR color image
        """
        # Apply gamma correction
        heatmap_gamma = np.power(heatmap, self.gamma)
        
        # Convert to 8-bit
        heatmap_u8 = (heatmap_gamma * 255).astype(np.uint8)
        
        # Apply Gaussian blur for smoothing
        if self.blur_sigma > 0:
            heatmap_u8 = cv2.GaussianBlur(heatmap_u8, (0, 0), self.blur_sigma)
            
        # Apply colormap
        colored = cv2.applyColorMap(heatmap_u8, self.colormap)
        
        # Resize if needed
        if size is not None:
            colored = cv2.resize(colored, size, interpolation=cv2.INTER_LINEAR)
            
        return colored
        
    def render_with_alpha(
        self,
        heatmap: np.ndarray,
        size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Render heatmap with per-pixel alpha channel.
        
        Args:
            heatmap: 2D float array (normalized 0-1)
            size: Output size (width, height)
            
        Returns:
            BGRA image with alpha channel
        """
        # Render color
        colored = self.render(heatmap, size)
        
        # Compute alpha based on intensity
        if size is not None:
            heatmap_resized = cv2.resize(heatmap, size, interpolation=cv2.INTER_LINEAR)
        else:
            heatmap_resized = heatmap
            
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * np.power(heatmap_resized, self.gamma)
        alpha = (alpha * 255).astype(np.uint8)
        
        # Combine to BGRA
        bgra = cv2.cvtColor(colored, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha
        
        return bgra
        
    def overlay(
        self,
        heatmap: np.ndarray,
        background: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay heatmap on background image.
        
        Args:
            heatmap: 2D float array (normalized 0-1)
            background: BGR background image
            alpha: Overall blend alpha
            
        Returns:
            Blended BGR image
        """
        h, w = background.shape[:2]
        
        # Render heatmap to match background size
        colored = self.render(heatmap, (w, h))
        
        # Create mask from heatmap intensity
        if len(heatmap.shape) == 2:
            heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            heatmap_resized = heatmap
            
        # Per-pixel blending based on intensity
        mask = np.power(heatmap_resized, self.gamma)
        mask = self.min_alpha + (self.max_alpha - self.min_alpha) * mask
        mask = mask * alpha
        
        # Expand mask to 3 channels
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
            
        # Blend
        blended = background.astype(np.float32) * (1 - mask) + colored.astype(np.float32) * mask
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return blended
        
    def overlay_transparent(
        self,
        heatmap: np.ndarray,
        background: np.ndarray
    ) -> np.ndarray:
        """
        Overlay heatmap with transparency based on intensity.
        
        Low intensity regions are fully transparent,
        high intensity regions are more opaque.
        
        Args:
            heatmap: 2D float array (normalized 0-1)
            background: BGR background image
            
        Returns:
            Blended BGR image
        """
        h, w = background.shape[:2]
        
        # Render heatmap with alpha
        heatmap_bgra = self.render_with_alpha(heatmap, (w, h))
        
        # Extract color and alpha
        heatmap_bgr = heatmap_bgra[:, :, :3]
        alpha = heatmap_bgra[:, :, 3:4].astype(np.float32) / 255
        
        # Blend
        blended = background.astype(np.float32) * (1 - alpha) + heatmap_bgr.astype(np.float32) * alpha
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return blended
        
    def render_crosshair(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        size: int = 20,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw crosshair at gaze point.
        
        Args:
            image: BGR image to draw on
            x, y: Crosshair center
            size: Crosshair arm length
            color: BGR color
            thickness: Line thickness
            
        Returns:
            Image with crosshair
        """
        output = image.copy()
        
        # Horizontal line
        cv2.line(output, (x - size, y), (x + size, y), color, thickness)
        # Vertical line
        cv2.line(output, (x, y - size), (x, y + size), color, thickness)
        # Center dot
        cv2.circle(output, (x, y), 3, color, -1)
        
        return output
        
    def render_gaze_trail(
        self,
        image: np.ndarray,
        points: list,
        max_points: int = 50,
        start_color: Tuple[int, int, int] = (255, 0, 0),
        end_color: Tuple[int, int, int] = (0, 0, 255),
        max_thickness: int = 4
    ) -> np.ndarray:
        """
        Draw fading trail of recent gaze points.
        
        Args:
            image: BGR image to draw on
            points: List of (x, y) points (oldest first)
            max_points: Maximum points to draw
            start_color: Color for oldest points
            end_color: Color for newest points
            max_thickness: Maximum line thickness
            
        Returns:
            Image with gaze trail
        """
        output = image.copy()
        
        # Take last max_points
        points = points[-max_points:]
        n = len(points)
        
        if n < 2:
            return output
            
        for i in range(n - 1):
            # Interpolate color and thickness
            t = i / (n - 1)
            color = tuple(int(start_color[j] * (1 - t) + end_color[j] * t) for j in range(3))
            thickness = max(1, int(max_thickness * t))
            
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))
            
            cv2.line(output, pt1, pt2, color, thickness)
            
        return output
        
    def set_colormap(self, colormap: str):
        """Change colormap."""
        self.colormap_name = colormap
        self.colormap = self.COLORMAPS.get(colormap, cv2.COLORMAP_JET)

