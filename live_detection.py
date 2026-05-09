# live_detection.py
import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
from utils.config_live import LIVE_CONFIG
from utils.camera_handler import CameraHandler
from utils.frame_queue import FrameQueue
from defect_tracker import DefectTracker
from alert_system import AlertSystem
from logger import InspectionLogger
from performance_monitor import PerformanceMonitor
from textile_detector import TextileDetector
from defect_validator import DefectValidator
from false_positive_filter import FalsePositiveFilter
from config import DEFECT_CLASSES, VIS_CONFIG

class LiveDetectionSystem:
    def __init__(self):
        self.config = LIVE_CONFIG
        self.model = YOLO(self.config['detection']['model_path'])
        
        # Load Classifier
        self.use_classifier = self.config['detection']['use_classifier']
        if self.use_classifier:
            self.classifier = YOLO(self.config['detection']['classifier_path'])
            print(f"Loaded classifier: {self.config['detection']['classifier_path']}")
        self.camera = CameraHandler(self.config['camera_id'], self.config['resolution'])
        self.tracker = DefectTracker(
            max_distance=self.config['tracking']['max_distance'],
            max_age=self.config['tracking']['max_age']
        )
        self.alert = AlertSystem(
            enable_sound=self.config['alerts']['enable_sound'],
            frequency=self.config['alerts']['sound_frequency'],
            duration=self.config['alerts']['sound_duration']
        )
        self.logger = InspectionLogger(
            log_dir=self.config['logging']['log_dir'],
            frame_dir=self.config['logging']['frame_dir']
        )
        self.monitor = PerformanceMonitor(smoothing=self.config['performance']['fps_smoothing'])
        
        self.textile_detector = TextileDetector()
        self.defect_validator = DefectValidator()
        self.fp_filter = FalsePositiveFilter()
        
        self.frame_queue = FrameQueue(max_size=2)
        self.is_running = False
        self.is_paused = False
        self.show_debug = True
        self.conf_threshold = self.config['confidence_threshold']
        self.alert_persistence = 0
        self.frame_id = 0
        self.total_defects_session = 0
        self.detected_ids = set()

    def start(self):
        if not self.camera.start():
            return
            
        self.is_running = True
        # Start processing thread
        self.process_thread = threading.Thread(target=self._run_inference_loop, daemon=True)
        self.process_thread.start()
        
        self._run_display_loop()

    def _run_inference_loop(self):
        try:
            while self.is_running:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                    
                success, frame = self.camera.read()
                if not success:
                    time.sleep(0.01)
                    continue
                    
                self.monitor.start_inference()
                
                # STEP 1: Pre-filter (Is it textile?)
                textile_results = self.textile_detector.is_textile(frame)
                roi_mask = textile_results['roi_mask']
                
                if self.config['detection'].get('strict_mode', False) and not textile_results['is_textile']:
                    self.monitor.end_inference()
                    annotated_frame = frame.copy()
                    cv2.putText(annotated_frame, "STATUS: NO TEXTILE DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    self.frame_queue.put(annotated_frame)
                    continue

                # STEP 2: YOLO Inference
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=0.45,
                    device=self.config['detection']['device'],
                    verbose=False,
                    imgsz=self.config['detection']['imgsz']
                )
                
                self.monitor.end_inference()
                
                # Extract and Validate detections
                rects, confs, clss = [], [], []
                
                for r in results:
                    for box in r.boxes:
                        b = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        # 1. Validation (Contrast, ROI, Size)
                        val = self.defect_validator.validate_defect(b, frame, roi_mask)
                        if not val['is_valid']: continue
                        
                        # 2. False Positive Filter (Sharp edges, patterns)
                        fp = self.fp_filter.filter_detections(b, frame)
                        if not fp['is_valid']: continue
                        
                        # 3. Edge Margin Filter
                        frame_h, frame_w = frame.shape[:2]
                        margin = 5
                        if b[0] < margin or b[1] < margin or b[2] > frame_w - margin or b[3] > frame_h - margin:
                            continue
                        
                        # 4. Secondary Classifier (Real-world Kaggle Knowledge)
                        if self.use_classifier:
                            x1, y1, x2, y2 = map(int, b)
                            pad = 10
                            crop = frame[max(0, y1-pad):min(frame_h, y2+pad), max(0, x1-pad):min(frame_w, x2+pad)]
                            if crop.size > 0:
                                cls_res = self.classifier.predict(crop, imgsz=64, verbose=False)[0]
                                top1_idx = cls_res.probs.top1
                                top1_name = self.classifier.names[top1_idx]
                                top1_conf = float(cls_res.probs.top1conf)
                                
                                if top1_name == 'good' or top1_conf < self.config['detection']['classifier_conf']:
                                    continue
                            
                        rects.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
                        confs.append(conf)
                        clss.append(cls)
                
                # Update Tracker
                self.tracker.update(rects, confs, clss)
                active_defects = self.tracker.get_active_defects(min_hits=self.config['tracking']['min_hits'])
                
                # Handle Alerts & Logging
                if len(active_defects) > 0:
                    self.alert.trigger()
                    self.alert_persistence = self.config['alerts']['alert_persistence']
                    
                    for d in active_defects:
                        if d['id'] not in self.detected_ids:
                            self.detected_ids.add(d['id'])
                            self.total_defects_session += 1
                    
                    self.logger.log(self.frame_id, len(active_defects), active_defects)
                
                # Prepare annotated frame
                annotated_frame = self._draw_annotations(frame.copy(), active_defects)
                self.frame_queue.put(annotated_frame)
                
                self.frame_id += 1
                
        except Exception as e:
            print(f"CRITICAL ERROR in Inference Loop: {e}")
            self.is_running = False

    def _draw_annotations(self, frame, defects):
        # Draw defects
        for d in defects:
            x1, y1, x2, y2 = d['bbox']
            label = f"ID:{d['id']} {DEFECT_CLASSES[d['class']]} {d['conf']:.2f}"
            
            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), (0, 0, 255), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw alert overlay
        if self.alert_persistence > 0:
            self.alert.draw_alert(frame, len(defects), self.alert_persistence)
            self.alert_persistence -= 1
        else:
            # Green status if clear
            cv2.putText(frame, "STATUS: SCANNING - NORMAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Performance Overlay
        stats = self.monitor.get_stats()
        cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (frame.shape[1]-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Latency: {stats['latency']:.1f}ms", (frame.shape[1]-200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"GPU Mem: {stats['gpu_mem']:.1f}GB", (frame.shape[1]-200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Session Stats
        cv2.putText(frame, f"Session Defects: {self.total_defects_session}", (20, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {self.conf_threshold:.2f}", (20, frame.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def _run_display_loop(self):
        print("Textile Defect Detection Live System Started.")
        print("Controls: [SPACE]: Pause, [S]: Save, [+/-]: Threshold, [Q]: Quit")
        
        while self.is_running:
            self.monitor.update_fps()
            
            frame = self.frame_queue.get()
            if frame is not None:
                cv2.imshow("TextileGuard AI - Live Inspection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
            elif key == ord(' '):
                self.is_paused = not self.is_paused
                print(f"System {'PAUSED' if self.is_paused else 'RESUMED'}")
            elif key == ord('s'):
                if frame is not None:
                    path = self.logger.save_frame(frame, len(self.tracker.objects))
                    print(f"Frame saved to {path}")
            elif key == ord('+') or key == ord('='):
                self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
            elif key == ord('-') or key == ord('_'):
                self.conf_threshold = max(0.1, self.conf_threshold - 0.05)
            elif key == ord('c'):
                self.use_classifier = not self.use_classifier
                print(f"Classifier validation: {'ENABLED' if self.use_classifier else 'DISABLED'}")

        self.camera.stop()
        cv2.destroyAllWindows()
        print("System shutdown complete.")

if __name__ == "__main__":
    system = LiveDetectionSystem()
    system.start()
