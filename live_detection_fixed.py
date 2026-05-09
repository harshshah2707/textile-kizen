# live_detection_fixed.py
import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
from utils.camera_handler import CameraHandler
from utils.frame_queue import FrameQueue
from alert_system import AlertSystem
from logger import InspectionLogger
from performance_monitor import PerformanceMonitor
from config import DEFECT_CLASSES, BEST_MODEL_PATH

class SimpleAccurateDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.camera = CameraHandler(0, (1280, 720))
        self.alert = AlertSystem()
        self.logger = InspectionLogger()
        self.monitor = PerformanceMonitor()
        
        self.conf_threshold = 0.5
        self.edge_margin = 20
        self.min_area = 100
        self.max_area_ratio = 0.3
        
        self.is_running = False
        self.frame_queue = FrameQueue(max_size=2)
        
        # Simple temporal persistence (ID-less)
        self.prev_detections = []
        self.persistence_threshold = 2
        self.alert_persistence = 0

    def start(self):
        if not self.camera.start(): return
        self.is_running = True
        threading.Thread(target=self._inference_loop, daemon=True).start()
        self._display_loop()

    def _inference_loop(self):
        while self.is_running:
            success, frame = self.camera.read()
            if not success: continue
            
            self.monitor.start_inference()
            results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
            self.monitor.end_inference()
            
            valid_rects = []
            frame_h, frame_w = frame.shape[:2]
            
            for r in results:
                for box in r.boxes:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = b
                    w, h = x2 - x1, y2 - y1
                    area = w * h
                    
                    # 1. Edge Filter (Reject if touching borders)
                    if x1 < self.edge_margin or y1 < self.edge_margin or \
                       x2 > frame_w - self.edge_margin or y2 > frame_h - self.edge_margin:
                        continue
                        
                    # 2. Size Filter
                    if area < self.min_area or area > (frame_w * frame_h * self.max_area_ratio):
                        continue
                        
                    valid_rects.append({'bbox': b, 'conf': float(box.conf[0]), 'cls': int(box.cls[0])})

            # 3. Simple Temporal Persistence (Check if we had detections recently)
            if len(valid_rects) > 0:
                self.alert_persistence = min(10, self.alert_persistence + 2)
                if self.alert_persistence >= self.persistence_threshold:
                    self.alert.trigger()
            else:
                self.alert_persistence = max(0, self.alert_persistence - 1)

            # Draw
            annotated = frame.copy()
            for d in valid_rects:
                x1, y1, x2, y2 = d['bbox']
                label = f"{DEFECT_CLASSES[d['cls']]} {d['conf']:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if self.alert_persistence >= self.persistence_threshold:
                cv2.putText(annotated, "⚠️ DEFECT DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(annotated, "✓ SCANNING - NORMAL", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            self.frame_queue.put(annotated)

    def _display_loop(self):
        while self.is_running:
            self.monitor.update_fps()
            frame = self.frame_queue.get()
            if frame is not None:
                stats = self.monitor.get_stats()
                cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (frame.shape[1]-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.imshow("TextileGuard FIXED", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
        
        self.camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SimpleAccurateDetection(str(BEST_MODEL_PATH))
    app.start()
