# verify_linecam.py
import cv2
import time
import os
import sys

from utils.camera_handler import CameraHandler
from utils.config_live import LIVE_CONFIG

def main():
    print("=" * 60)
    print("  MindVision Line-Scan Camera Verification Tool")
    print("=" * 60)
    
    # Try to open MindVision camera
    print("[Test] Attempting to open MindVision camera via CameraHandler wrapper...")
    cam = CameraHandler('mindvision', resolution=(640, 480))
    
    success = cam.start()
    if not success:
        print("[Error] Failed to start MindVision camera.")
        print("[Info] Make sure a MindVision camera is connected via GigE/USB and drivers are installed.")
        print("[Info] Falling back to standard webcam index 0 for basic handler verification...")
        cam = CameraHandler(0, resolution=(640, 480))
        success = cam.start()
        if not success:
            print("[Error] Failed to open fallback camera as well. Exiting.")
            return

    print("[Success] Camera handler started successfully!")
    print("[Info] Press 'q' to exit the preview window.")
    
    window_name = "Integrated Camera Handler Stitched Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    
    last_time = time.time()
    frame_count = 0
    fps = 0.0
    
    try:
        while True:
            ret, frame = cam.read()
            if ret and frame is not None:
                # Calculate FPS
                frame_count += 1
                now = time.time()
                if now - last_time >= 1.0:
                    fps = frame_count / (now - last_time)
                    frame_count = 0
                    last_time = now
                
                # Draw FPS indicator
                cv2.putText(
                    frame, 
                    f"Test Tool FPS: {fps:.1f}", 
                    (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0), 
                    2
                )
                
                cv2.imshow(window_name, frame)
            else:
                # Wait for frames to initialize
                time.sleep(0.01)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("[Clean] Stopping camera...")
        cam.stop()
        cv2.destroyAllWindows()
        print("[Clean] Finished.")

if __name__ == "__main__":
    main()
