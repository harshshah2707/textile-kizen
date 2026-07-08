import os
import glob
import random
import cv2
from ultralytics import YOLO

def main():
    model_path = "runs/textile_detection/multiclass_v1/weights/best.pt"
    if not os.path.exists(model_path):
        model_path = "runs/textile_detection/defect_model_pro_v1/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"[Verify] Model not found at {model_path}")
        return

    print(f"[Verify] Loading model from {model_path}...")
    model = YOLO(model_path)
    
    val_images_dir = "datasets/multiclass_yolo/images/val"
    out_dir = "runs/multiclass_val_predictions"
    os.makedirs(out_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(val_images_dir, "*.jpg"))
    if not image_paths:
        print("[Verify] No validation images found.")
        return
        
    print(f"[Verify] Found {len(image_paths)} validation images.")
    # Pick 5 random validation images
    random.seed(42)
    selected = random.sample(image_paths, min(10, len(image_paths)))
    
    for img_path in selected:
        print(f"[Verify] Running prediction on {os.path.basename(img_path)}...")
        results = model(img_path, conf=0.25)
        for r in results:
            annotated = r.plot()
            out_path = os.path.join(out_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, annotated)
            print(f"  -> Saved prediction preview to {out_path}")

if __name__ == "__main__":
    main()
