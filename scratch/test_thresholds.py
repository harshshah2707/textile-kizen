import os
import glob
import cv2
from ultralytics import YOLO

def main():
    weights_path = "runs/textile_detection/defect_model_pro_v1/weights/best.pt"
    model = YOLO(weights_path)
    
    val_images = glob.glob("datasets/lusitano_yolo/images/val/defect_*.jpg")
    print(f"Testing {len(val_images)} validation patches...")
    
    thresholds = [0.01, 0.05, 0.10, 0.20]
    
    for conf in thresholds:
        total_detections = 0
        for path in val_images:
            img = cv2.imread(path)
            results = model.predict(source=img, conf=conf, verbose=False)
            for r in results:
                total_detections += len(r.boxes)
        print(f"Conf={conf:.2f}: Total Detections across all patches = {total_detections}")

if __name__ == "__main__":
    main()
