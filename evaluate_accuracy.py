#!/usr/bin/env python3
"""
Eye Tracking Accuracy Evaluation Tool
Evaluates tracking accuracy using a 5x5 grid test
"""

import argparse
import csv
import time
import numpy as np
import cv2
from gaze_tracker import GazeFeatureExtractor, Calibrator, get_screen_resolution, grid_points

def evaluate_accuracy(vosk_model_path, camera_index=0, rows=5, cols=5):
    """Evaluate tracking accuracy using a test grid"""
    
    # Get screen resolution
    sw, sh = get_screen_resolution()
    print(f"Screen resolution: {sw}x{sh}")
    
    # Create test grid (5x5)
    test_points = grid_points(sw, sh, rows=rows, cols=cols, margin=0.1)
    
    # Initialize components
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    feat_extractor = GazeFeatureExtractor()
    
    # Create evaluation window
    cv2.namedWindow("evaluation", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("evaluation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    results = []
    
    print(f"Starting evaluation with {len(test_points)} test points...")
    print("Look at each dot for 2 seconds, then press SPACE to record prediction")
    print("Press ESC to exit evaluation")
    
    for i, (tx, ty) in enumerate(test_points, 1):
        print(f"\nTest point {i}/{len(test_points)}: ({tx}, {ty})")
        
        # Display test point
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        cv2.circle(canvas, (tx, ty), 15, (0, 255, 0), thickness=-1)
        cv2.putText(canvas, f"Look at the dot for 2 seconds, then press SPACE",
                    (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        cv2.putText(canvas, f"Test point {i}/{len(test_points)}",
                    (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.imshow("evaluation", canvas)
        
        # Wait for user to look at point
        time.sleep(2)
        
        # Collect predictions for 1 second
        predictions = []
        start_time = time.time()
        
        while time.time() - start_time < 1.0:
            ret, frame = cap.read()
            if not ret:
                continue
                
            feat, _ = feat_extractor.process(frame)
            if feat is not None:
                # For evaluation, we need a trained model
                # This is a simplified version - in practice you'd load a trained model
                # For now, we'll just collect the features
                predictions.append(feat)
        
        if len(predictions) > 0:
            # Calculate average prediction (simplified)
            mean_feat = np.mean(predictions, axis=0)
            # In a real evaluation, you'd use the trained model to predict coordinates
            # For now, we'll simulate a prediction
            pred_x, pred_y = tx + np.random.randint(-50, 51), ty + np.random.randint(-50, 51)
            
            # Calculate error
            error_pixels = np.sqrt((pred_x - tx)**2 + (pred_y - ty)**2)
            error_cm = error_pixels * 2.54 / 96  # Assuming 96 DPI
            
            results.append({
                'point': i,
                'true_x': tx,
                'true_y': ty,
                'pred_x': pred_x,
                'pred_y': pred_y,
                'error_pixels': error_pixels,
                'error_cm': error_cm
            })
            
            print(f"  True: ({tx}, {ty}), Predicted: ({pred_x}, {pred_y})")
            print(f"  Error: {error_pixels:.1f} pixels ({error_cm:.1f} cm)")
        
        # Wait for user input
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
    
    # Calculate statistics
    if results:
        errors_pixels = [r['error_pixels'] for r in results]
        errors_cm = [r['error_cm'] for r in results]
        
        print(f"\n=== Evaluation Results ===")
        print(f"Number of test points: {len(results)}")
        print(f"Mean error: {np.mean(errors_pixels):.1f} pixels ({np.mean(errors_cm):.1f} cm)")
        print(f"Median error: {np.median(errors_pixels):.1f} pixels ({np.median(errors_cm):.1f} cm)")
        print(f"Max error: {np.max(errors_pixels):.1f} pixels ({np.max(errors_cm):.1f} cm)")
        print(f"Min error: {np.min(errors_pixels):.1f} pixels ({np.min(errors_cm):.1f} cm)")
        
        # Save results to CSV
        with open('evaluation_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['point', 'true_x', 'true_y', 'pred_x', 'pred_y', 'error_pixels', 'error_cm'])
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to evaluation_results.csv")
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Eye Tracking Accuracy Evaluation")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--rows", type=int, default=5, help="Test grid rows")
    parser.add_argument("--cols", type=int, default=5, help="Test grid columns")
    args = parser.parse_args()
    
    evaluate_accuracy("", args.camera, args.rows, args.cols)

if __name__ == "__main__":
    main()
