import os
import sys
import cv2
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from ultralytics import YOLO
from utils.bbox_refiner import refine_defect_bbox

def test_refiner():
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
    print(f"Testing refiner on 5 validation images...")
    
    for i, img_path in enumerate(val_images[:5]):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        results = model.predict(source=img, conf=0.15, verbose=False)
        print(f"\n--- Image {i+1}: {img_path.name} ---")
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            refined = refine_defect_bbox(img, xyxy, cls)
            
            w_orig = xyxy[2] - xyxy[0]
            h_orig = xyxy[3] - xyxy[1]
            w_ref = refined[2] - refined[0]
            h_ref = refined[3] - refined[1]
            
            print(f"  Class: {model.names[cls]}, Conf: {conf:.3f}")
            print(f"    Original BBox: {[round(x,1) for x in xyxy]} (Size: {round(w_orig,1)}x{round(h_orig,1)})")
            print(f"    Refined  BBox: {refined} (Size: {w_ref}x{h_ref})")

if __name__ == '__main__':
    test_refiner()
