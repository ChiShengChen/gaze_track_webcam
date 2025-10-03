#!/usr/bin/env python3
"""
Eye Tracking System - Real-time gaze tracking using webcam, voice calibration and heatmap display
Supports macOS, uses Mediapipe for facial feature extraction and Vosk for voice keyword detection
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

import sounddevice as sd
import vosk

# -----------------------------
# 常數定義
# -----------------------------
LEFT_EYE_LANDMARKS = dict(
    outer=33, inner=133, upper=159, lower=145
)
RIGHT_EYE_LANDMARKS = dict(
    inner=362, outer=263, upper=386, lower=374
)

def dist(a, b):
    """Calculate distance between two points"""
    return math.hypot(a[0]-b[0], a[1]-b[1])

def clamp(v, lo, hi):
    """Clamp value within range"""
    return max(lo, min(hi, v))

# -----------------------------
# Gaze Feature Extractor
# -----------------------------
class GazeFeatureExtractor:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # Need iris details
            min_detection_confidence=0.7,  # Higher detection confidence
            min_tracking_confidence=0.7    # Higher tracking confidence
        )
        self.iris_indices = None
        self._iod0 = None  # Calibration head distance ratio baseline

    def _get_iris_indices(self):
        """Get iris feature point indices"""
        try:
            conns = mp.solutions.face_mesh.FACEMESH_IRISES
            idxs = set()
            for a, b in conns:
                idxs.add(a)
                idxs.add(b)
            return sorted(list(idxs))
        except Exception:
            # Fallback: Common iris point indices
            return list(range(468, 478))

    def process(self, frame_bgr) -> Tuple[Optional[np.ndarray], dict]:
        """Process image frame and extract gaze features"""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(frame_rgb)
        dbg = {}
        
        if not res.multi_face_landmarks:
            return None, dbg

        lm = res.multi_face_landmarks[0].landmark
        if self.iris_indices is None:
            self.iris_indices = self._get_iris_indices()

        # Extract iris points
        pts = np.array([[lm[i].x*w, lm[i].y*h] for i in self.iris_indices], dtype=np.float32)
        if pts.shape[0] < 6:
            return None, dbg
            
        # Separate left and right eyes by x-coordinate
        xs = pts[:, 0]
        midx = np.median(xs)
        left_iris = pts[xs < midx]
        right_iris = pts[xs >= midx]
        lc = left_iris.mean(axis=0)
        rc = right_iris.mean(axis=0)

        # Eye socket reference points
        L = LEFT_EYE_LANDMARKS
        R = RIGHT_EYE_LANDMARKS
        pL = {k: np.array([lm[i].x*w, lm[i].y*h]) for k, i in L.items()}
        pR = {k: np.array([lm[i].x*w, lm[i].y*h]) for k, i in R.items()}

        # Eye width and height
        lw = max(1e-3, dist(pL['outer'], pL['inner']))
        lh = max(1e-3, dist(pL['upper'], pL['lower']))
        rw = max(1e-3, dist(pR['outer'], pR['inner']))
        rh = max(1e-3, dist(pR['upper'], pR['lower']))

        # Normalized coordinates (relative to eye socket)
        nlx = (lc[0] - pL['outer'][0]) / lw
        nly = (lc[1] - min(pL['upper'][1], pL['lower'][1])) / lh
        nrx = (rc[0] - pR['inner'][0]) / rw
        nry = (rc[1] - min(pR['upper'][1], pR['lower'][1])) / rh

        # Inter-ocular distance ratio (head distance compensation)
        left_center = 0.5*(pL['outer'] + pL['inner'])
        right_center = 0.5*(pR['outer'] + pR['inner'])
        iod = dist(left_center, right_center)
        if self._iod0 is None:
            self._iod0 = iod
        iod_ratio = clamp(iod / (self._iod0 + 1e-6), 0.6, 1.6)

        # Feature vector: [left_eye_x, left_eye_y, right_eye_x, right_eye_y, iod_ratio]
        feat = np.array([nlx, nly, nrx, nry, iod_ratio], dtype=np.float32)

        # Debug information
        dbg['left_iris'] = tuple(map(int, lc))
        dbg['right_iris'] = tuple(map(int, rc))
        dbg['left_box'] = (tuple(map(int, pL['outer'])), tuple(map(int, pL['inner'])),
                           tuple(map(int, pL['upper'])), tuple(map(int, pL['lower'])))
        dbg['right_box'] = (tuple(map(int, pR['outer'])), tuple(map(int, pR['inner'])),
                            tuple(map(int, pR['upper'])), tuple(map(int, pR['lower'])))
        return feat, dbg

# -----------------------------
# Voice Keyword Detection
# -----------------------------
class KeywordListener(threading.Thread):
    def __init__(self, model_path: str, keywords: List[str] = ["here", "這裡"], samplerate=16000):
        super().__init__(daemon=True)
        self.model = vosk.Model(model_path)
        # Use vocabulary to narrow recognition scope
        self.rec = vosk.KaldiRecognizer(self.model, samplerate, json.dumps(keywords))
        self.samplerate = samplerate
        self.q = queue.Queue()
        self.running = False
        self.trigger_q = queue.Queue(maxsize=32)

    def callback(self, indata, frames, t, status):
        """Audio callback function"""
        if status:
            pass  # Handle audio status
        self.q.put(bytes(indata))

    def run(self):
        """Execute voice detection"""
        self.running = True
        with sd.RawInputStream(samplerate=self.samplerate, blocksize=8000,
                               dtype='int16', channels=1, callback=self.callback):
            while self.running:
                try:
                    data = self.q.get(timeout=0.3)
                except queue.Empty:
                    continue
                if self.rec.AcceptWaveform(data):
                    res = json.loads(self.rec.Result())
                    text = res.get("text", "").strip().lower()
                    if text in ("here", "這裡"):
                        ts = time.time()
                        try:
                            self.trigger_q.put_nowait(ts)
                        except queue.Full:
                            pass

    def pop_trigger(self) -> Optional[float]:
        """Get trigger timestamp"""
        try:
            return self.trigger_q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Stop detection"""
        self.running = False

# -----------------------------
# Calibrator
# -----------------------------
@dataclass
class Calibrator:
    screen_w: int
    screen_h: int
    X: List[np.ndarray] = field(default_factory=list)
    Y: List[Tuple[float, float]] = field(default_factory=list)
    W: List[float] = field(default_factory=list)
    pipe_x: Optional[Pipeline] = None
    pipe_y: Optional[Pipeline] = None
    deg: int = 2
    alpha: float = 1.0

    def add(self, feat: np.ndarray, target_xy: Tuple[float, float], weight: Optional[float] = None):
        """Add calibration point with optional weighting"""
        self.X.append(feat.astype(np.float32))
        self.Y.append(target_xy)
        if weight is None:
            # Edge weighting: closer to edges → higher weight
            x, y = target_xy
            nx, ny = x / self.screen_w, y / self.screen_h
            edge = 1.0 - 2.0 * min(nx, 1-nx, ny, 1-ny)  # 0 at center, 1 at edges
            w = 1.0 + 1.5 * max(0.0, edge)              # γ=1.5 adjustable
        else:
            w = weight
        if not hasattr(self, 'W'):
            self.W = []
        self.W.append(w)

    def fit(self):
        """Train polynomial regression model with edge weighting"""
        if len(self.X) < 12:
            raise RuntimeError("Too few calibration samples (at least 12, recommend 16-25)")
        X = np.vstack(self.X)
        Y = np.array(self.Y, dtype=np.float32)
        
        # Create polynomial features pipeline
        self.pipe_x = Pipeline([
            ("poly", PolynomialFeatures(degree=self.deg, include_bias=True)),
            ("reg", Ridge(alpha=self.alpha))
        ])
        self.pipe_y = Pipeline([
            ("poly", PolynomialFeatures(degree=self.deg, include_bias=True)),
            ("reg", Ridge(alpha=self.alpha))
        ])
        
        # Fit with sample weights
        weights = np.array(self.W) if self.W else None
        self.pipe_x.fit(X, Y[:, 0], reg__sample_weight=weights)
        self.pipe_y.fit(X, Y[:, 1], reg__sample_weight=weights)

    def predict(self, feat: np.ndarray) -> Tuple[int, int]:
        """Predict screen coordinates"""
        if self.pipe_x is None or self.pipe_y is None:
            return (self.screen_w//2, self.screen_h//2)
        x = float(self.pipe_x.predict(feat.reshape(1, -1))[0])
        y = float(self.pipe_y.predict(feat.reshape(1, -1))[0])
        x = int(clamp(x, 0, self.screen_w-1))
        y = int(clamp(y, 0, self.screen_h-1))
        return x, y

# -----------------------------
# Heatmap Management
# -----------------------------
class Heatmap:
    def __init__(self, screen_w, screen_h, downsample=6, sigma=9):
        self.w = screen_w
        self.h = screen_h
        self.ds = downsample
        self.grid_w = max(8, self.w//self.ds)
        self.grid_h = max(8, self.h//self.ds)
        self.acc = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.sigma = sigma

    def add_point(self, x, y):
        """Add gaze point"""
        gx = clamp(x//self.ds, 0, self.grid_w-1)
        gy = clamp(y//self.ds, 0, self.grid_h-1)
        self.acc[gy, gx] += 2.0  # Increase weight for higher sensitivity

    def render(self):
        """Render heatmap"""
        # Reduce decay to maintain heat zones longer
        self.acc *= 0.998
        arr = cv2.GaussianBlur(self.acc, (0, 0), self.sigma)
        if arr.max() > 0:
            # Increase contrast to make heat zones more visible
            arr = np.power(arr / arr.max(), 0.7) * 255.0
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        color = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
        vis = cv2.resize(color, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return vis

# -----------------------------
# Utility Functions
# -----------------------------
def get_screen_resolution():
    """Get screen resolution"""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080

def grid_points(sw, sh, rows=3, cols=3, margin=0.12):
    """Generate calibration point grid"""
    xs = np.linspace(margin, 1.0-margin, cols)
    ys = np.linspace(margin, 1.0-margin, rows)
    pts = [(int(x*sw), int(y*sh)) for y in ys for x in xs]
    return pts

# -----------------------------
# Main Program
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Eye Tracking System")
    parser.add_argument("--vosk-model", default="./vosk-model-small-en-us-0.15", help="Vosk model folder path")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--rows", type=int, default=3, help="Calibration points rows")
    parser.add_argument("--cols", type=int, default=3, help="Calibration points columns")
    parser.add_argument("--show-cam-debug", action="store_true", help="Show camera debug window")
    parser.add_argument("--cam-mirror", action="store_true", help="Mirror camera horizontally")
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.vosk_model):
        print(f"Error: Vosk model path not found: {args.vosk_model}")
        print("Please run ./setup.sh to download models first")
        sys.exit(1)

    # Get screen resolution
    sw, sh = get_screen_resolution()
    print(f"Screen resolution: {sw}x{sh}")

    # Initialize voice detection
    print("Initializing voice detection...")
    kw = KeywordListener(args.vosk_model, keywords=["here", "這裡"])
    kw.start()

    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(args.camera)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Cannot open camera, please check:")
        print("1. Camera is not being used by other programs")
        print("2. System Preferences > Security & Privacy > Camera is authorized")
        print("3. Try restarting the program")
        sys.exit(1)
    
    # Set camera parameters
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Test camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from camera")
        cap.release()
        sys.exit(1)
    
    print("Camera initialized successfully")
    feat_extractor = GazeFeatureExtractor()

    # Initialize calibrator
    calib = Calibrator(sw, sh)
    points = grid_points(sw, sh, rows=args.rows, cols=args.cols, margin=0.12)

    print(f"Starting calibration, {len(points)} points total...")
    
    # Calibration phase
    win_cal = "calibration"
    cv2.namedWindow(win_cal, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_cal, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for i, (tx, ty) in enumerate(points, 1):
        print(f"Calibration point {i}/{len(points)}: ({tx}, {ty})")
        while True:
            # Display calibration point
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            cv2.circle(canvas, (tx, ty), 12, (0, 255, 0), thickness=-1)
            cv2.putText(canvas, f"Look at the dot and say 'here' / 「這裡」  ({i}/{len(points)})",
                        (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
            cv2.imshow(win_cal, canvas)

            # Process camera frame
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Apply mirroring if requested
            if args.cam_mirror:
                frame = cv2.flip(frame, 1)
                
            feat, dbg = feat_extractor.process(frame)
            ts = kw.pop_trigger()
            
            if ts is not None and feat is not None:
                # Multi-frame averaging: collect 0.4 seconds of features
                print(f"Collecting samples for point {i}...")
                samples = []
                t_end = time.time() + 0.4
                while time.time() < t_end:
                    ret2, frame2 = cap.read()
                    if not ret2:
                        continue
                    # Apply mirroring if requested
                    if args.cam_mirror:
                        frame2 = cv2.flip(frame2, 1)
                    feat2, _ = feat_extractor.process(frame2)
                    if feat2 is not None:
                        samples.append(feat2)
                
                if len(samples) >= 6:
                    mean_feat = np.mean(samples, axis=0)
                    calib.add(mean_feat, (tx, ty))
                    print(f"Recorded calibration point {i} with {len(samples)} samples")
                else:
                    # Fallback: use single sample
                    calib.add(feat, (tx, ty))
                    print(f"Recorded calibration point {i} with single sample")
                
                # Visual feedback
                cv2.circle(canvas, (tx, ty), 16, (255, 128, 0), thickness=-1)
                cv2.imshow(win_cal, canvas)
                cv2.waitKey(200)
                break

            if cv2.waitKey(1) & 0xFF == 27:  # ESC cancel
                print("Calibration cancelled")
                kw.stop()
                kw.join(timeout=1.0)
                cap.release()
                cv2.destroyAllWindows()
                return

    cv2.destroyWindow(win_cal)
    print("Training regression model...")
    calib.fit()
    print("Calibration completed!")

    # Inference phase
    heat = Heatmap(sw, sh, downsample=4, sigma=7)  # Higher resolution, less blur
    ema_xy = None
    alpha = 0.3  # Higher response speed
    
    # Open CSV file for recording
    csv_file = open("gaze_points.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "x", "y"])

    if args.show_cam_debug:
        cv2.namedWindow("camera", cv2.WINDOW_NORMAL)

    cv2.namedWindow("gaze_heatmap", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("gaze_heatmap", 640, 360)

    print("Starting inference mode. Say 'here' to record points, press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Apply mirroring if requested
        if args.cam_mirror:
            frame = cv2.flip(frame, 1)
            
        feat, dbg = feat_extractor.process(frame)
        if feat is not None:
            x, y = calib.predict(feat)
            if ema_xy is None:
                ema_xy = np.array([x, y], dtype=np.float32)
            else:
                ema_xy = (1-alpha)*ema_xy + alpha*np.array([x, y], dtype=np.float32)
            hx, hy = int(ema_xy[0]), int(ema_xy[1])
            heat.add_point(hx, hy)

        # Display heatmap
        vis = heat.render()
        cv2.imshow("gaze_heatmap", vis)

        # Optional: Show camera debug window
        if args.show_cam_debug and dbg:
            dbg_img = frame.copy()
            cv2.circle(dbg_img, dbg.get('left_iris', (0, 0)), 4, (0, 255, 0), -1)
            cv2.circle(dbg_img, dbg.get('right_iris', (0, 0)), 4, (0, 255, 0), -1)
            for p in dbg.get('left_box', []):
                cv2.circle(dbg_img, p, 3, (255, 0, 0), -1)
            for p in dbg.get('right_box', []):
                cv2.circle(dbg_img, p, 3, (0, 0, 255), -1)
            cv2.imshow("camera", dbg_img)

        # Voice trigger recording
        ts = kw.pop_trigger()
        if ts is not None and ema_xy is not None:
            csv_writer.writerow([ts, int(ema_xy[0]), int(ema_xy[1])])
            csv_file.flush()
            print(f"Recorded gaze point: ({int(ema_xy[0])}, {int(ema_xy[1])})")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup resources
    kw.stop()
    kw.join(timeout=1.0)
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("Program ended")

if __name__ == "__main__":
    main()
