import os
from pathlib import Path
from ultralytics import YOLO
import cv2

def main():
    model_path = 'runs/textile_detection/defect_model_pro_v1/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Model path not found: {model_path}")
        # Try fallback models
        model_path = 'runs/textile_detection/multiclass_v1/weights/best.pt'
        if not os.path.exists(model_path):
            print(f"Fallback model path not found: {model_path}")
            model_path = 'runs/textile_detection/defect_model_v1/weights/best.pt'
            if not os.path.exists(model_path):
                print("No trained models found.")
                return

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    print(f"Model classes: {model.names}")

    # Search for validation images in different dataset directories
    val_dirs = [
        'datasets/multiclass_yolo/images/val',
        'datasets/pro_yolo/images/val',
        'datasets/synthetic/images/val',
        'datasets/custom_real/val/images'
    ]

    found_images = []
    for d in val_dirs:
        p = Path(d)
        if p.exists():
            imgs = list(p.glob('*.jpg')) + list(p.glob('*.png')) + list(p.glob('*.jpeg'))
            if imgs:
                print(f"Found {len(imgs)} validation images in: {d}")
                found_images.extend((d, img) for img in imgs)

    if not found_images:
        print("No validation images found anywhere.")
        return

    # Run inference on the first 5 validation images from each available folder
    processed_folders = set()
    for d, img_path in found_images:
        if d in processed_folders and len([x for x in processed_folders if x == d]) >= 5:
            continue
        
        print("\n" + "="*50)
        print(f"Testing image: {img_path}")
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Failed to read image: {img_path}")
            continue

        print(f"Image shape: {frame.shape}")
        
        # Test with multiple confidence thresholds
        for conf in [0.1, 0.25, 0.5]:
            results = model.predict(source=frame, conf=conf, verbose=False, device='cpu')
            boxes = results[0].boxes
            print(f"  Conf: {conf:.2f} -> Detections: {len(boxes)}")
            for i, box in enumerate(boxes):
                c = int(box.cls[0])
                label = model.names.get(c, str(c))
                score = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                print(f"    [{i}] Class: {label} ({c}), Conf: {score:.3f}, BBox: {[round(x, 1) for x in bbox]}")
        
        processed_folders.add(d)

if __name__ == '__main__':
    main()
