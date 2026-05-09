"""
Textile Defect Detection - Batch Image Processing
=================================================
Processes multiple images in a folder and outputs results.
"""

import os
import cv2
import json
from pathlib import Path
from inference import TextileDefectDetector
from config import OUTPUT_DIR, DATASET_DIR

def run_batch_test(source_dir, output_dir):
    detector = TextileDefectDetector()
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
    print(f"Found {len(image_files)} images in {source_dir}")
    
    batch_results = []
    
    for img_file in image_files:
        print(f"Processing {img_file.name}...")
        results = detector.detect(str(img_file))
        
        img = cv2.imread(str(img_file))
        vis_img, count = detector.visualize(img, results)
        
        # Save annotated image
        save_path = output_path / f"res_{img_file.name}"
        cv2.imwrite(str(save_path), vis_img)
        
        # Record data
        batch_results.append({
            "image": img_file.name,
            "defect_count": count,
            "status": "PASS" if count == 0 else "FAIL"
        })
        
    # Save JSON report
    with open(output_path / "batch_results.json", "w") as f:
        json.dump(batch_results, f, indent=4)
        
    print(f"Batch processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    # Test on validation set by default
    val_dir = DATASET_DIR / "images" / "val"
    if val_dir.exists():
        run_batch_test(val_dir, OUTPUT_DIR / "batch_val")
    else:
        print(f"Validation directory not found at {val_dir}. Generate dataset first.")
