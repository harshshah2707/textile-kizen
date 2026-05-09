"""
Textile Defect Detection - Inference Engine
===========================================
Handles image, video, and webcam detection using trained YOLOv8 model.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from config import BEST_MODEL_PATH, INFERENCE_CONFIG, DEFECT_CLASSES, VIS_CONFIG, QC_THRESHOLDS

class TextileDefectDetector:
    def __init__(self, model_path=None):
        path = model_path or str(BEST_MODEL_PATH)
        self.model = YOLO(path)
        self.config = INFERENCE_CONFIG

    def detect(self, source, show=False):
        results = self.model.predict(
            source=source,
            conf=self.config["confidence_threshold"],
            iou=self.config["iou_threshold"],
            device=self.config["device"]
        )
        return results

    def visualize(self, frame, results):
        defect_count = 0
        for result in results:
            boxes = result.boxes
            defect_count += len(boxes)
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{DEFECT_CLASSES[cls]} {conf:.2f}"
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), VIS_CONFIG["box_color"], VIS_CONFIG["box_thickness"])
                
                # Draw label
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, VIS_CONFIG["font_scale"], VIS_CONFIG["font_thickness"])
                cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), VIS_CONFIG["box_color"], -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, VIS_CONFIG["font_scale"], VIS_CONFIG["label_text_color"], VIS_CONFIG["font_thickness"])

        # Status overlay
        status = "PASS" if defect_count == 0 else "FAIL"
        color = VIS_CONFIG["status_pass_color"] if defect_count == 0 else VIS_CONFIG["status_fail_color"]
        cv2.putText(frame, f"STATUS: {status} ({defect_count} defects)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame, defect_count

    def run_webcam(self):
        cap = cv2.VideoCapture(0)
        print("Webcam started. Press 'q' to exit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            results = self.detect(frame)
            frame, count = self.visualize(frame, results)
            
            cv2.imshow("Textile Defect Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    detector = TextileDefectDetector()
    if len(sys.argv) > 1 and sys.argv[1] == "webcam":
        detector.run_webcam()
    else:
        print("Usage: python inference.py webcam")
