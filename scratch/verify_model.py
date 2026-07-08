import os
import cv2
import glob
from ultralytics import YOLO

def main():
    weights_path = "runs/textile_detection/defect_model_pro_v1/weights/best.pt"
    print(f"[Verify] Loading model from: {weights_path}")
    
    if not os.path.exists(weights_path):
        print(f"[Verify] Error: Model weights not found at {weights_path}")
        return
        
    model = YOLO(weights_path)
    
    # Create predictions output folder
    out_dir = "scratch/predictions"
    os.makedirs(out_dir, exist_ok=True)
    
    # Grab a few validation patch images
    raw_defects = glob.glob("datasets/lusitano_yolo/images/val/defect_*.jpg")
    if not raw_defects:
        print("[Verify] Error: No validation patch images found in datasets/lusitano_yolo/images/val/")
        return
        
    # Take up to 3 images for verification
    test_images = raw_defects[:3]
    print(f"[Verify] Running model inference on {len(test_images)} test images...")
    
    for idx, path in enumerate(test_images):
        basename = os.path.basename(path)
        print(f"Processing: {basename}")
        
        # Load image
        img = cv2.imread(path)
        if img is None:
            continue
            
        # Run prediction
        # Use conf=0.25 to see standard detections
        results = model.predict(source=img, conf=0.25, verbose=False)
        
        # Annotate image with detections
        annotated_img = img.copy()
        detections_found = 0
        
        for r in results:
            for box in r.boxes:
                b = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Draw green rectangle for detection
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                label = f"Defect: {conf:.2f}"
                cv2.putText(annotated_img, label, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                print(f"  Detected Defect: bbox=[{x1}, {y1}, {x2}, {y2}], conf={conf:.2f}")
                detections_found += 1
                
        # Save annotated image
        out_path = os.path.join(out_dir, f"annotated_{basename}")
        cv2.imwrite(out_path, annotated_img)
        print(f"  Saved annotated prediction to {out_path} (Detections: {detections_found})")
        
    print("\n[Verify] Verification complete! Outputs saved to scratch/predictions/")

if __name__ == "__main__":
    main()
