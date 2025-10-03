#!/usr/bin/env python3
"""
Advanced Eye Tracking System - Full-screen overlay heatmap display
Uses PyQt6 to implement semi-transparent, click-through overlay effects
"""

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
import csv
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np
from sklearn.linear_model import Ridge

import sounddevice as sd
import vosk

# PyQt6 imports
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import QTimer, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider
from PyQt6.QtGui import QPixmap, QPainter, QImage
from PyQt6.QtCore import Qt

# 從主程式匯入共用類別
from gaze_tracker import (
    GazeFeatureExtractor, KeywordListener, Calibrator, Heatmap,
    get_screen_resolution, grid_points, dist, clamp
)

# -----------------------------
# PyQt6 Overlay Window
# -----------------------------
class OverlayWindow(QWidget):
    """Full-screen semi-transparent overlay window"""
    
    def __init__(self, screen_w, screen_h):
        super().__init__()
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.heatmap_image = None
        self.opacity = 0.6
        
        # Set window properties
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        
        # Set geometry
        self.setGeometry(0, 0, screen_w, screen_h)
        
        # Set style
        self.setStyleSheet("background: transparent;")
        
    def update_heatmap(self, heatmap_bgr):
        """Update heatmap"""
        if heatmap_bgr is None:
            return
            
        # Convert BGR to RGB
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = heatmap_rgb.shape
        bytes_per_line = ch * w
        
        # Create QImage
        q_image = QImage(heatmap_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.heatmap_image = QPixmap.fromImage(q_image)
        self.update()
        
    def set_opacity(self, opacity):
        """Set opacity"""
        self.opacity = max(0.0, min(1.0, opacity))
        self.update()
        
    def paintEvent(self, event):
        """Paint event"""
        if self.heatmap_image is None:
            return
            
        painter = QPainter(self)
        painter.setOpacity(self.opacity)
        painter.drawPixmap(0, 0, self.heatmap_image)

# -----------------------------
# Control Panel
# -----------------------------
class ControlPanel(QWidget):
    """Control panel window"""
    
    def __init__(self, overlay_window):
        super().__init__()
        self.overlay_window = overlay_window
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Eye Tracking Control Panel")
        self.setFixedSize(300, 200)
        
        layout = QVBoxLayout()
        
        # Opacity control
        opacity_label = QLabel("Opacity:")
        opacity_slider = QSlider(Qt.Orientation.Horizontal)
        opacity_slider.setRange(0, 100)
        opacity_slider.setValue(60)
        opacity_slider.valueChanged.connect(self.on_opacity_changed)
        
        # Control buttons
        self.toggle_btn = QPushButton("Hide Overlay")
        self.toggle_btn.clicked.connect(self.toggle_overlay)
        
        self.calibrate_btn = QPushButton("Recalibrate")
        self.calibrate_btn.clicked.connect(self.start_calibration)
        
        self.quit_btn = QPushButton("Quit")
        self.quit_btn.clicked.connect(self.quit_app)
        
        # Status label
        self.status_label = QLabel("Status: Running")
        
        # Add to layout
        layout.addWidget(opacity_label)
        layout.addWidget(opacity_slider)
        layout.addWidget(self.toggle_btn)
        layout.addWidget(self.calibrate_btn)
        layout.addWidget(self.quit_btn)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
        self.overlay_visible = True
        
    def on_opacity_changed(self, value):
        """Opacity changed"""
        opacity = value / 100.0
        self.overlay_window.set_opacity(opacity)
        
    def toggle_overlay(self):
        """Toggle overlay display"""
        if self.overlay_visible:
            self.overlay_window.hide()
            self.toggle_btn.setText("Show Overlay")
            self.overlay_visible = False
        else:
            self.overlay_window.show()
            self.toggle_btn.setText("Hide Overlay")
            self.overlay_visible = True
            
    def start_calibration(self):
        """Start recalibration"""
        self.status_label.setText("Status: Preparing calibration...")
        # This can trigger calibration process
        # Actual implementation needs integration with main program
        
    def quit_app(self):
        """Quit application"""
        QApplication.quit()

# -----------------------------
# Eye Tracking Worker Thread
# -----------------------------
class GazeTrackingThread(QThread):
    """Eye tracking worker thread"""
    
    heatmap_updated = pyqtSignal(np.ndarray)
    status_updated = pyqtSignal(str)
    
    def __init__(self, vosk_model_path, camera_index=0, rows=3, cols=3):
        super().__init__()
        self.vosk_model_path = vosk_model_path
        self.camera_index = camera_index
        self.rows = rows
        self.cols = cols
        
        # Initialize components
        self.cap = None
        self.feat_extractor = None
        self.kw_listener = None
        self.calib = None
        self.heatmap = None
        self.ema_xy = None
        self.alpha = 0.3  # Higher response speed
        
        self.running = False
        self.calibration_mode = True
        
    def run(self):
        """Execute main loop"""
        try:
            self.setup_components()
            self.run_calibration()
            self.run_inference()
        except Exception as e:
            self.status_updated.emit(f"Error: {str(e)}")
        finally:
            self.cleanup()
            
    def setup_components(self):
        """Initialize components"""
        self.status_updated.emit("Initializing components...")
        
        # Get screen resolution
        sw, sh = get_screen_resolution()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera, please check permissions")
        
        # Set camera parameters
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Test camera
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise RuntimeError("Cannot read from camera")
        
        # Initialize feature extractor
        self.feat_extractor = GazeFeatureExtractor()
        
        # Initialize voice detection
        self.kw_listener = KeywordListener(self.vosk_model_path, ["here", "這裡"])
        self.kw_listener.start()
        
        # Initialize calibrator
        self.calib = Calibrator(sw, sh)
        
        # Initialize heatmap
        self.heatmap = Heatmap(sw, sh, downsample=4, sigma=7)
        
        self.status_updated.emit("Components initialized")
        
    def run_calibration(self):
        """Execute calibration"""
        self.status_updated.emit("Starting calibration...")
        
        sw, sh = get_screen_resolution()
        points = grid_points(sw, sh, rows=self.rows, cols=self.cols, margin=0.12)
        
        for i, (tx, ty) in enumerate(points, 1):
            self.status_updated.emit(f"Calibration point {i}/{len(points)}: ({tx}, {ty})")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                feat, dbg = self.feat_extractor.process(frame)
                ts = self.kw_listener.pop_trigger()
                
                if ts is not None and feat is not None:
                    self.calib.add(feat, (tx, ty))
                    self.status_updated.emit(f"Recorded calibration point {i}")
                    break
                    
                # Check if should exit
                if not self.running:
                    return
                    
        # Train model
        self.status_updated.emit("Training regression model...")
        self.calib.fit()
        self.status_updated.emit("Calibration completed!")
        
    def run_inference(self):
        """Execute inference"""
        self.status_updated.emit("Starting real-time tracking...")
        self.calibration_mode = False
        
        # Open CSV file
        csv_file = open("gaze_points.csv", "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "x", "y"])
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            feat, dbg = self.feat_extractor.process(frame)
            if feat is not None:
                x, y = self.calib.predict(feat)
                if self.ema_xy is None:
                    self.ema_xy = np.array([x, y], dtype=np.float32)
                else:
                    self.ema_xy = (1-self.alpha)*self.ema_xy + self.alpha*np.array([x, y], dtype=np.float32)
                hx, hy = int(self.ema_xy[0]), int(self.ema_xy[1])
                self.heatmap.add_point(hx, hy)
                
                # Send heatmap update
                heatmap_img = self.heatmap.render()
                self.heatmap_updated.emit(heatmap_img)
                
            # Voice trigger recording
            ts = self.kw_listener.pop_trigger()
            if ts is not None and self.ema_xy is not None:
                csv_writer.writerow([ts, int(self.ema_xy[0]), int(self.ema_xy[1])])
                csv_file.flush()
                self.status_updated.emit(f"Recorded gaze point: ({int(self.ema_xy[0])}, {int(self.ema_xy[1])})")
                
        csv_file.close()
        
    def cleanup(self):
        """Cleanup resources"""
        if self.kw_listener:
            self.kw_listener.stop()
            self.kw_listener.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            
    def stop(self):
        """Stop tracking"""
        self.running = False

# -----------------------------
# Main Application
# -----------------------------
class GazeTrackingApp(QApplication):
    """Main eye tracking application"""
    
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        # Get screen resolution
        self.screen_w, self.screen_h = get_screen_resolution()
        
        # Create overlay window
        self.overlay_window = OverlayWindow(self.screen_w, self.screen_h)
        
        # Create control panel
        self.control_panel = ControlPanel(self.overlay_window)
        
        # Create tracking thread
        self.tracking_thread = GazeTrackingThread(
            self.args.vosk_model,
            self.args.camera,
            self.args.rows,
            self.args.cols
        )
        
        self.setup_connections()
        self.show_windows()
        
    def setup_connections(self):
        """Setup signal connections"""
        # Connect tracking thread signals
        self.tracking_thread.heatmap_updated.connect(self.overlay_window.update_heatmap)
        self.tracking_thread.status_updated.connect(self.control_panel.status_label.setText)
        
        # Connect control panel signals
        self.control_panel.quit_btn.clicked.connect(self.quit)
        
    def show_windows(self):
        """Show windows"""
        self.overlay_window.show()
        self.control_panel.show()
        
    def start_tracking(self):
        """Start tracking"""
        self.tracking_thread.running = True
        self.tracking_thread.start()
        
    def stop_tracking(self):
        """Stop tracking"""
        self.tracking_thread.stop()
        self.tracking_thread.wait()

# -----------------------------
# Main Program Entry
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Advanced Eye Tracking System - Full-screen overlay version")
    parser.add_argument("--vosk-model", default="./vosk-model-small-en-us-0.15", help="Vosk model folder path")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--rows", type=int, default=3, help="Calibration points rows")
    parser.add_argument("--cols", type=int, default=3, help="Calibration points columns")
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.vosk_model):
        print(f"Error: Vosk model path not found: {args.vosk_model}")
        print("Please run ./setup.sh to download models first")
        sys.exit(1)
    
    # Create application
    app = GazeTrackingApp(sys.argv)
    
    # Start tracking
    app.start_tracking()
    
    # Execute application
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        app.stop_tracking()
        sys.exit(0)

if __name__ == "__main__":
    main()
