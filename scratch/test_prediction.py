import os
import cv2
from pathlib import Path
from ultralytics import YOLO

def test_prediction():
    model_path = 'runs/textile_detection/defect_model_pro_v1/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = YOLO(model_path)
    val_images_dir = Path('datasets/multiclass_yolo/images/val')
    if not val_images_dir.exists():
        print(f"Error: Directory not found at {val_images_dir}")
        return
        
    val_images = list(val_images_dir.glob('*.jpg'))
    print(f"Found {len(val_images)} validation images.")
    
    # Test on first 10 validation images
    for i, img_path in enumerate(val_images[:10]):
        print(f"\n--- Image {i+1}: {img_path.name} ---")
        img = cv2.imread(str(img_path))
        if img is None:
            print("Failed to load image.")
            continue
            
        # Run prediction at 0.15 threshold
        results = model.predict(source=img, conf=0.15, verbose=False)
        print(f"Detections at conf=0.15:")
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(f"  Class: {model.names[cls]} ({cls}), Conf: {conf:.3f}, BBox: {[round(x,1) for x in xyxy]}")

if __name__ == '__main__':
    test_prediction()
