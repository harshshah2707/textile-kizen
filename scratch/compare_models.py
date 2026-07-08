import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from ultralytics import YOLO

def compare_models():
    model_paths = {
        "defect_model_pro_v1": "runs/textile_detection/defect_model_pro_v1/weights/best.pt",
        "multiclass_v1": "runs/textile_detection/multiclass_v1/weights/best.pt",
    }
    
    val_images_dir = Path('datasets/multiclass_yolo/images/val')
    if not val_images_dir.exists():
        print(f"Error: Directory not found at {val_images_dir}")
        return
        
    val_images = list(val_images_dir.glob('*.jpg'))[:50]
    print(f"Comparing models on {len(val_images)} validation images...")
    
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"{name}: file not found at {path}")
            continue
            
        model = YOLO(path)
        total_detections = 0
        total_conf = 0.0
        images_with_defects = 0
        
        for img_path in val_images:
            results = model.predict(source=str(img_path), conf=0.25, verbose=False)
            boxes = results[0].boxes
            if len(boxes) > 0:
                total_detections += len(boxes)
                total_conf += sum([float(b.conf[0]) for b in boxes])
                images_with_defects += 1
                
        avg_conf = total_conf / total_detections if total_detections > 0 else 0.0
        print(f"\n{name} Results:")
        print(f"  Images with detections: {images_with_defects}/{len(val_images)}")
        print(f"  Total defects detected: {total_detections}")
        print(f"  Average confidence: {avg_conf:.3f}")

if __name__ == '__main__':
    compare_models()
