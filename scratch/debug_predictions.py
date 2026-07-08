import os
import glob
import cv2
from ultralytics import YOLO

def main():
    weights_path = "runs/textile_detection/lusitano_finetune/weights/best.pt"
    if not os.path.exists(weights_path):
        weights_path = "runs/textile_detection/lusitano_finetune/weights/last.pt"
    
    if not os.path.exists(weights_path):
        print("Model weights not found!")
        return
        
    print(f"Loading weights from {weights_path}...")
    model = YOLO(weights_path)
    
    # Check 5 training patch images with defects
    train_images = glob.glob("datasets/lusitano_yolo/images/train/defect_*.jpg")[:5]
    print(f"Running debug inference on {len(train_images)} training images...")
    
    for path in train_images:
        basename = os.path.basename(path)
        print(f"\nImage: {basename}")
        
        # Read the ground truth label file
        label_path = path.replace("images", "labels").replace(".jpg", ".txt")
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                print("  Ground Truth:", f.read().strip())
        
        img = cv2.imread(path)
        # Use a very low confidence threshold to print all candidates
        results = model.predict(source=img, conf=0.0001, verbose=False)
        
        candidates = []
        for r in results:
            for box in r.boxes:
                b = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                candidates.append((cls, conf, [int(x) for x in b]))
                
        # Sort candidates by confidence descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        print(f"  Total raw detections (conf>=0.0001): {len(candidates)}")
        print("  Top 5 candidates:")
        for c in candidates[:5]:
            print(f"    Class {c[0]} (conf={c[1]:.5f}), Bbox={c[2]}")

if __name__ == "__main__":
    main()
