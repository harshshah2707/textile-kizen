# logger.py
import csv
import os
import time
import cv2
from datetime import datetime

class InspectionLogger:
    def __init__(self, log_dir="logs", frame_dir="saved_frames"):
        self.log_dir = log_dir
        self.frame_dir = frame_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(frame_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"inspection_{timestamp}.csv")
        self.is_logging = True
        
        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Frame_ID", "Defect_Count", "Detections"])

    def log(self, frame_id, defect_count, detections):
        if not self.is_logging: return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        det_str = "; ".join([f"ID:{d['id']}-Cls:{d['class']}-Conf:{d['conf']:.2f}" for d in detections])
        
        try:
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, frame_id, defect_count, det_str])
        except:
            pass

    def save_frame(self, frame, defect_count):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"defect_{timestamp}_cnt{defect_count}.jpg"
        filepath = os.path.join(self.frame_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath
