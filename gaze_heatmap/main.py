#!/usr/bin/env python3
"""
Gaze Heatmap Application - CLI Entry Point

Commands:
    calibrate - Run calibration procedure
    record    - Record gaze session with heatmap
    evaluate  - Evaluate tracking accuracy
    label     - Annotate recorded heatmaps
    demo      - Live demo with heatmap overlay
"""

import argparse
import sys
import time
import cv2
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime

# Support both direct execution and module execution
try:
    from core.gaze_estimator import GazeEstimator
    from core.calibration import CalibrationSystem
    from core.smoother import create_smoother
    from core.screen_mapper import ScreenConfig
    from heatmap.accumulator import HeatmapAccumulator
    from heatmap.renderer import HeatmapRenderer
    from heatmap.exporter import HeatmapExporter
    from evaluation.error_metrics import ErrorMetricsCalculator, OnlineEvaluator
    from evaluation.labeling_tool import LabelingTool
    from evaluation.benchmark import Benchmark, BenchmarkConfig
    from utils.logger import setup_logging, get_logger
except ImportError:
    from .core.gaze_estimator import GazeEstimator
    from .core.calibration import CalibrationSystem
    from .core.smoother import create_smoother
    from .core.screen_mapper import ScreenConfig
    from .heatmap.accumulator import HeatmapAccumulator
    from .heatmap.renderer import HeatmapRenderer
    from .heatmap.exporter import HeatmapExporter
    from .evaluation.error_metrics import ErrorMetricsCalculator, OnlineEvaluator
    from .evaluation.labeling_tool import LabelingTool
    from .evaluation.benchmark import Benchmark, BenchmarkConfig
    from .utils.logger import setup_logging, get_logger


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent / config_path
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Return default config
        return {
            'gaze': {'model': 'mediapipe', 'device': 'cpu', 'camera_id': 0},
            'calibration': {'num_points': 9, 'point_duration_ms': 1500},
            'smoothing': {'method': 'kalman'},
            'heatmap': {'sigma': 50.0, 'downsample': 4},
            'evaluation': {'viewing_distance_cm': 60.0, 'monitor_dpi': 96.0},
            'output': {'base_dir': './data'},
        }


def get_screen_resolution() -> tuple:
    """Get primary screen resolution."""
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


def setup_camera(config: dict) -> cv2.VideoCapture:
    """Initialize camera with config settings."""
    gaze_config = config.get('gaze', {})
    
    cap = cv2.VideoCapture(gaze_config.get('camera_id', 0))
    
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Check permissions and camera availability.")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, gaze_config.get('frame_width', 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, gaze_config.get('frame_height', 720))
    cap.set(cv2.CAP_PROP_FPS, gaze_config.get('fps', 30))
    
    return cap


def run_calibration(config: dict, args):
    """Run calibration procedure."""
    logger = get_logger(__name__)
    logger.info("Starting calibration...")
    
    screen_w, screen_h = get_screen_resolution()
    logger.info(f"Screen resolution: {screen_w}x{screen_h}")
    
    # Initialize components
    gaze_config = config.get('gaze', {})
    cal_config = config.get('calibration', {})
    
    gaze_estimator = GazeEstimator(
        device=gaze_config.get('device', 'cpu'),
        model=gaze_config.get('model', 'mediapipe')
    )
    
    calibration = CalibrationSystem(
        screen_width=screen_w,
        screen_height=screen_h,
        num_points=cal_config.get('num_points', 9),
        point_duration_ms=cal_config.get('point_duration_ms', 1500),
        collection_window_ms=cal_config.get('collection_window_ms', 500),
        polynomial_degree=cal_config.get('polynomial_degree', 2),
        margin=cal_config.get('margin', 0.1),
    )
    
    cap = setup_camera(config)
    
    try:
        profile = calibration.run_calibration(gaze_estimator, cap)
        
        # Validate if requested
        if args.validate:
            logger.info("Validating calibration...")
            error = calibration.validate_calibration(gaze_estimator, cap, num_validation_points=5)
            logger.info(f"Validation error: {error:.1f} pixels")
        
        # Save profile
        output_path = args.output or "calibration.yaml"
        output_dir = Path(config.get('output', {}).get('calibrations_dir', './data/calibrations'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        full_path = output_dir / output_path
        calibration.save_profile(str(full_path))
        logger.info(f"Calibration saved to: {full_path}")
        
    except KeyboardInterrupt:
        logger.info("Calibration cancelled")
    finally:
        cap.release()
        gaze_estimator.close()
        cv2.destroyAllWindows()


def run_recording(config: dict, args):
    """Record gaze session with heatmap generation."""
    logger = get_logger(__name__)
    logger.info("Starting recording session...")
    
    screen_w, screen_h = get_screen_resolution()
    
    # Initialize components
    gaze_config = config.get('gaze', {})
    smooth_config = config.get('smoothing', {})
    heat_config = config.get('heatmap', {})
    
    gaze_estimator = GazeEstimator(
        device=gaze_config.get('device', 'cpu'),
        model=gaze_config.get('model', 'mediapipe')
    )
    
    calibration = CalibrationSystem(screen_w, screen_h)
    
    # Load calibration
    if not args.calibration:
        logger.error("Calibration file required. Run 'calibrate' first.")
        return
    
    # Search for calibration file in multiple locations
    cal_path = Path(args.calibration)
    if not cal_path.exists():
        # Try in calibrations directory
        alt_path = Path(config.get('output', {}).get('calibrations_dir', './data/calibrations')) / args.calibration
        if alt_path.exists():
            cal_path = alt_path
        else:
            logger.error(f"Calibration file not found: {args.calibration}")
            logger.error(f"Also tried: {alt_path}")
            return
            
    calibration.load_profile(str(cal_path))
    logger.info(f"Loaded calibration: {cal_path}")
    
    # Create smoother
    smoother = create_smoother(
        method=smooth_config.get('method', 'kalman'),
        **smooth_config.get(smooth_config.get('method', 'kalman'), {})
    )
    
    # Create heatmap accumulator
    heatmap = HeatmapAccumulator(
        width=screen_w,
        height=screen_h,
        sigma=heat_config.get('sigma', 50.0),
        downsample=heat_config.get('downsample', 4),
        decay_rate=heat_config.get('decay_rate', 0.0),
        fixation_velocity_threshold=heat_config.get('fixation_velocity_threshold', 100),
        fixation_duration_threshold=heat_config.get('fixation_duration_threshold', 100),
    )
    
    renderer = HeatmapRenderer(colormap=heat_config.get('colormap', 'jet'))
    
    cap = setup_camera(config)
    
    # Take screenshot at start
    screenshot = None
    if config.get('output', {}).get('save_screenshot', True):
        try:
            import pyautogui
            screenshot = np.array(pyautogui.screenshot())
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        except ImportError:
            logger.warning("pyautogui not installed, skipping screenshot")
    
    # Recording data
    gaze_data = []
    session_id = args.output or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    duration = args.duration or 60
    
    logger.info(f"Recording for {duration} seconds...")
    logger.info("Press 'q' to stop early")
    
    cv2.namedWindow("heatmap", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("heatmap", 640, 360)
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame_count += 1
            timestamp = time.time()
            
            gaze_result = gaze_estimator.estimate(frame)
            
            if gaze_result is not None:
                # Map to screen
                raw_x, raw_y = calibration.map_gaze_to_screen(
                    gaze_result.gaze_pitch,
                    gaze_result.gaze_yaw,
                    gaze_result.head_pose[1],
                    gaze_result.head_pose[2]
                )
                
                # Smooth
                smooth_x, smooth_y = smoother.update(raw_x, raw_y, timestamp)
                
                # Detect fixation
                velocity = smoother.get_velocity()
                speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                is_fixation = speed < heat_config.get('fixation_velocity_threshold', 100)
                
                # Accumulate
                heatmap.add_gaze_point(smooth_x, smooth_y, timestamp, is_fixation)
                
                # Record data
                if config.get('output', {}).get('save_raw_gaze', True):
                    gaze_data.append({
                        'timestamp_ms': (timestamp - start_time) * 1000,
                        'gaze_x': raw_x,
                        'gaze_y': raw_y,
                        'smoothed_x': smooth_x,
                        'smoothed_y': smooth_y,
                        'gaze_pitch': gaze_result.gaze_pitch,
                        'gaze_yaw': gaze_result.gaze_yaw,
                        'confidence': gaze_result.confidence,
                        'is_fixation': is_fixation,
                    })
                    
            # Display heatmap
            heatmap_vis = heatmap.get_full_resolution_heatmap()
            heatmap_colored = renderer.render(heatmap_vis)
            cv2.imshow("heatmap", heatmap_colored)
            
            # Show elapsed time
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            if frame_count % 30 == 0:
                logger.info(f"Recording: {elapsed:.1f}s / {duration}s ({remaining:.0f}s remaining)")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Recording stopped")
    finally:
        cap.release()
        gaze_estimator.close()
        cv2.destroyAllWindows()
        
    # Export session
    logger.info("Saving session data...")
    
    exporter = HeatmapExporter(
        output_dir=config.get('output', {}).get('sessions_dir', './data/sessions')
    )
    
    session_dir = exporter.export_session(
        accumulator=heatmap,
        session_id=session_id,
        gaze_data=gaze_data,
        screenshot=screenshot,
        config=config
    )
    
    logger.info(f"Session saved to: {session_dir}")
    
    # Print statistics
    stats = heatmap.get_statistics()
    logger.info(f"Statistics:")
    logger.info(f"  Total points: {stats['total_points']}")
    logger.info(f"  Total fixations: {stats['total_fixations']}")
    logger.info(f"  Mean fixation duration: {stats['mean_fixation_duration_ms']:.0f}ms")


def run_evaluation(config: dict, args):
    """Evaluate tracking accuracy."""
    logger = get_logger(__name__)
    logger.info("Starting evaluation...")
    
    screen_w, screen_h = get_screen_resolution()
    
    # Initialize components
    gaze_config = config.get('gaze', {})
    eval_config = config.get('evaluation', {})
    
    gaze_estimator = GazeEstimator(
        device=gaze_config.get('device', 'cpu'),
        model=gaze_config.get('model', 'mediapipe')
    )
    
    calibration = CalibrationSystem(screen_w, screen_h)
    
    # Load calibration
    if not args.calibration:
        logger.error("Calibration file required. Run 'calibrate' first.")
        return
    
    # Search for calibration file in multiple locations
    cal_path = Path(args.calibration)
    if not cal_path.exists():
        alt_path = Path(config.get('output', {}).get('calibrations_dir', './data/calibrations')) / args.calibration
        if alt_path.exists():
            cal_path = alt_path
        else:
            logger.error(f"Calibration file not found: {args.calibration}")
            return
            
    calibration.load_profile(str(cal_path))
    logger.info(f"Loaded calibration: {cal_path}")
    
    cap = setup_camera(config)
    
    # Run evaluation
    evaluator = OnlineEvaluator(
        screen_width=screen_w,
        screen_height=screen_h,
        num_test_points=args.num_points or eval_config.get('num_test_points', 20),
        point_duration_ms=eval_config.get('point_duration_ms', 1500),
        collection_window_ms=eval_config.get('collection_window_ms', 500),
        monitor_dpi=eval_config.get('monitor_dpi', 96.0),
        viewing_distance_cm=eval_config.get('viewing_distance_cm', 60.0),
    )
    
    try:
        result = evaluator.run_evaluation(gaze_estimator, calibration, cap)
        
        # Print results
        logger.info("\n=== Evaluation Results ===")
        logger.info(f"Angular Error:")
        logger.info(f"  Mean: {result.mean_angular_error:.2f}°")
        logger.info(f"  Std:  {result.std_angular_error:.2f}°")
        logger.info(f"  Median: {result.median_angular_error:.2f}°")
        logger.info(f"  95th percentile: {result.percentile_95_angular:.2f}°")
        logger.info(f"\nScreen Error:")
        logger.info(f"  Mean: {result.mean_screen_error_px:.1f} px")
        logger.info(f"  Mean: {result.mean_screen_error_cm:.2f} cm")
        logger.info(f"\nPrecision: {result.precision:.2f}°")
        
        # Save report
        if args.output:
            output_dir = Path(config.get('output', {}).get('base_dir', './data'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            report_path = output_dir / args.output
            evaluator.calculator.export_report(str(report_path))
            logger.info(f"\nReport saved to: {report_path}")
            
    except KeyboardInterrupt:
        logger.info("Evaluation cancelled")
    finally:
        cap.release()
        gaze_estimator.close()
        cv2.destroyAllWindows()


def run_labeling(config: dict, args):
    """Run labeling tool for heatmap annotation."""
    logger = get_logger(__name__)
    
    sessions_dir = config.get('output', {}).get('sessions_dir', './data/sessions')
    output_dir = config.get('output', {}).get('base_dir', './data') + '/annotations'
    
    tool = LabelingTool(
        sessions_dir=sessions_dir,
        output_dir=output_dir
    )
    
    if args.session:
        try:
            tool.load_session(args.session)
            logger.info(f"Loaded session: {args.session}")
            tool.run_ui()
        except FileNotFoundError as e:
            logger.error(str(e))
    else:
        # List available sessions
        sessions_path = Path(sessions_dir)
        if sessions_path.exists():
            sessions = [d.name for d in sessions_path.iterdir() if d.is_dir()]
            logger.info("Available sessions:")
            for s in sessions:
                logger.info(f"  - {s}")
            logger.info("\nUse: python main.py label --session <session_id>")
        else:
            logger.error("No sessions found. Run 'record' first.")


def run_demo(config: dict, args):
    """Live demo with real-time heatmap overlay."""
    logger = get_logger(__name__)
    logger.info("Starting demo...")
    
    screen_w, screen_h = get_screen_resolution()
    
    # Initialize components
    gaze_config = config.get('gaze', {})
    smooth_config = config.get('smoothing', {})
    heat_config = config.get('heatmap', {})
    display_config = config.get('display', {})
    
    gaze_estimator = GazeEstimator(
        device=gaze_config.get('device', 'cpu'),
        model=gaze_config.get('model', 'mediapipe')
    )
    
    calibration = CalibrationSystem(screen_w, screen_h)
    
    # Load or run calibration
    cal_path = None
    if args.calibration:
        cal_path = Path(args.calibration)
        if not cal_path.exists():
            alt_path = Path(config.get('output', {}).get('calibrations_dir', './data/calibrations')) / args.calibration
            if alt_path.exists():
                cal_path = alt_path
            else:
                cal_path = None
                
    if cal_path:
        calibration.load_profile(str(cal_path))
        logger.info(f"Loaded calibration: {cal_path}")
    else:
        logger.info("No calibration found, running calibration first...")
        cap = setup_camera(config)
        try:
            calibration.run_calibration(gaze_estimator, cap)
        except KeyboardInterrupt:
            logger.info("Calibration cancelled")
            cap.release()
            return
        cap.release()
    
    # Create components
    smoother = create_smoother(
        method=smooth_config.get('method', 'kalman'),
        **smooth_config.get(smooth_config.get('method', 'kalman'), {})
    )
    
    heatmap = HeatmapAccumulator(
        width=screen_w,
        height=screen_h,
        sigma=heat_config.get('sigma', 50.0),
        downsample=heat_config.get('downsample', 4),
    )
    
    renderer = HeatmapRenderer(
        colormap=heat_config.get('colormap', 'jet'),
        max_alpha=display_config.get('heatmap_opacity', 0.6)
    )
    
    cap = setup_camera(config)
    
    # Create windows
    cv2.namedWindow("heatmap", cv2.WINDOW_NORMAL)
    
    # Check if fullscreen display is requested
    use_fullscreen = display_config.get('fullscreen_demo', False)
    if use_fullscreen:
        cv2.resizeWindow("heatmap", screen_w, screen_h)
        cv2.moveWindow("heatmap", 0, 0)
        # Note: We don't use WINDOW_FULLSCREEN to avoid exit issues
        logger.info(f"Using fullscreen display ({screen_w}x{screen_h})")
    else:
        cv2.resizeWindow("heatmap", 800, 450)
        logger.info("Using windowed display (800x450)")
    
    if display_config.get('show_camera_feed', False):
        cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("camera", 640, 480)
    
    gaze_trail = []
    
    logger.info("Demo running. Press 'q' to quit, 'r' to reset heatmap, 'c' to recalibrate")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            timestamp = time.time()
            gaze_result = gaze_estimator.estimate(frame)
            
            # Debug: track detection rate
            if not hasattr(run_demo, '_debug_count'):
                run_demo._debug_count = {'total': 0, 'detected': 0, 'last_log': time.time()}
            
            run_demo._debug_count['total'] += 1
            
            if gaze_result is not None:
                run_demo._debug_count['detected'] += 1
                
                # Map to screen
                raw_x, raw_y = calibration.map_gaze_to_screen(
                    gaze_result.gaze_pitch,
                    gaze_result.gaze_yaw,
                    gaze_result.head_pose[1],
                    gaze_result.head_pose[2]
                )
                
                # Smooth
                smooth_x, smooth_y = smoother.update(raw_x, raw_y, timestamp)
                
                # Debug: log coordinates periodically
                if time.time() - run_demo._debug_count['last_log'] > 2.0:
                    detection_rate = run_demo._debug_count['detected'] / run_demo._debug_count['total'] * 100
                    logger.debug(f"Detection rate: {detection_rate:.1f}% | Raw: ({raw_x:.1f}, {raw_y:.1f}) | Smooth: ({smooth_x:.1f}, {smooth_y:.1f})")
                    run_demo._debug_count['last_log'] = time.time()
                    run_demo._debug_count['total'] = 0
                    run_demo._debug_count['detected'] = 0
                
                # Add to heatmap
                heatmap.add_gaze_point(smooth_x, smooth_y, timestamp)
                
                # Track trail
                gaze_trail.append((smooth_x, smooth_y))
                if len(gaze_trail) > display_config.get('trail_length', 50):
                    gaze_trail.pop(0)
            else:
                # Debug: warn if no detection for a while
                if run_demo._debug_count['total'] > 30 and run_demo._debug_count['detected'] == 0:
                    logger.warning("No face detected for 30 frames. Check camera and lighting.")
                    run_demo._debug_count['total'] = 0
                    
                # Show camera with debug info
                if display_config.get('show_camera_feed', False):
                    debug_frame = frame.copy()
                    cv2.circle(debug_frame, 
                              tuple(gaze_result.left_eye_center.astype(int)), 
                              4, (0, 255, 0), -1)
                    cv2.circle(debug_frame,
                              tuple(gaze_result.right_eye_center.astype(int)),
                              4, (0, 255, 0), -1)
                    cv2.imshow("camera", debug_frame)
                    
            # Render heatmap
            heatmap_vis = heatmap.get_full_resolution_heatmap()
            heatmap_colored = renderer.render(heatmap_vis)
            
            # Draw crosshair
            if gaze_trail and display_config.get('show_gaze_crosshair', True):
                # Get latest gaze point
                latest_x, latest_y = gaze_trail[-1]
                
                # Convert screen coordinates to heatmap image coordinates
                h, w = heatmap_colored.shape[:2]
                crosshair_x = int(latest_x * w / screen_w)
                crosshair_y = int(latest_y * h / screen_h)
                
                # Clamp to image bounds
                crosshair_x = max(0, min(w - 1, crosshair_x))
                crosshair_y = max(0, min(h - 1, crosshair_y))
                
                # Draw crosshair
                heatmap_colored = renderer.render_crosshair(
                    heatmap_colored,
                    crosshair_x,
                    crosshair_y
                )
                
            cv2.imshow("heatmap", heatmap_colored)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                heatmap.reset()
                gaze_trail.clear()
                smoother.reset()
                logger.info("Heatmap reset")
            elif key == ord('c'):
                logger.info("Recalibrating...")
                try:
                    calibration.run_calibration(gaze_estimator, cap)
                except KeyboardInterrupt:
                    pass
                    
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        gaze_estimator.close()
        cv2.destroyAllWindows()
        logger.info("Demo ended")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Gaze Heatmap Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  calibrate  Run calibration procedure
  record     Record gaze session with heatmap
  evaluate   Evaluate tracking accuracy  
  label      Annotate recorded heatmaps
  demo       Live demo with heatmap overlay

Examples:
  %(prog)s calibrate --output my_calibration.yaml
  %(prog)s record --calibration my_calibration.yaml --duration 60
  %(prog)s evaluate --calibration my_calibration.yaml --num-points 20
  %(prog)s demo --calibration my_calibration.yaml
        """
    )
    
    parser.add_argument('--config', default='config.yaml',
                        help='Configuration file path')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Calibrate command
    cal_parser = subparsers.add_parser('calibrate', help='Run calibration')
    cal_parser.add_argument('--output', '-o', help='Output calibration file')
    cal_parser.add_argument('--validate', action='store_true',
                           help='Validate calibration after completion')
    
    # Record command
    rec_parser = subparsers.add_parser('record', help='Record gaze session')
    rec_parser.add_argument('--calibration', '-c', required=True,
                           help='Calibration file to use')
    rec_parser.add_argument('--duration', '-d', type=int, default=60,
                           help='Recording duration in seconds')
    rec_parser.add_argument('--output', '-o', help='Session name')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate accuracy')
    eval_parser.add_argument('--calibration', '-c', required=True,
                            help='Calibration file to use')
    eval_parser.add_argument('--num-points', '-n', type=int,
                            help='Number of test points')
    eval_parser.add_argument('--output', '-o', help='Report output path')
    
    # Label command
    label_parser = subparsers.add_parser('label', help='Annotate heatmaps')
    label_parser.add_argument('--session', '-s', help='Session to annotate')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Live demo')
    demo_parser.add_argument('--calibration', '-c',
                            help='Calibration file (optional, will calibrate if not provided)')
    
    args = parser.parse_args()
    
    # Setup logging
    import logging
    level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=level)
    
    # Load config
    config = load_config(args.config)
    
    # Run command
    if args.command == 'calibrate':
        run_calibration(config, args)
    elif args.command == 'record':
        run_recording(config, args)
    elif args.command == 'evaluate':
        run_evaluation(config, args)
    elif args.command == 'label':
        run_labeling(config, args)
    elif args.command == 'demo':
        run_demo(config, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

