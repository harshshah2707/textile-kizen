import cv2
import numpy as np
import os
import glob
from pathlib import Path
from tqdm import tqdm
import random

# Paths
BASE_DATA_DIR = "datasets/Multi-Class Fabric Defect Detection Dataset"
OUT_DIR = "datasets/pro_yolo"
CLASSES = ['hole', 'stain', 'lines', 'Needle mark', 'Pinched fabric']

def find_defect_bbox(img):
    """
    Advanced defect localization using multi-thresholding and edge detection.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. Edge Detection (Good for holes and lines)
    edges = cv2.Canny(blurred, 30, 100)
    
    # 2. Adaptive Thresholding (Good for stains and shadows)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Combine both
    combined = cv2.bitwise_or(edges, thresh)
    
    # Clean up
    kernel = np.ones((5,5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Get largest contour that is roughly in the middle 60% of the image 
    # (industrial datasets usually center the defect)
    H, W = img.shape[:2]
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter noise and huge background objects
        if 5 < w < W*0.8 and 5 < h < H*0.8:
            valid_contours.append(cnt)
            
    if not valid_contours:
        return None
        
    cnt = max(valid_contours, key=cv2.contourArea)
    return cv2.boundingRect(cnt)

def generate_pro_dataset():
    print("Generating ADVANCED Pro YOLO Dataset...")
    # ... existing dir setup ...
    img_out = Path(OUT_DIR) / "images" / "train"
    lbl_out = Path(OUT_DIR) / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    
    val_img_out = Path(OUT_DIR) / "images" / "val"
    val_lbl_out = Path(OUT_DIR) / "labels" / "val"
    val_img_out.mkdir(parents=True, exist_ok=True)
    val_lbl_out.mkdir(parents=True, exist_ok=True)

    # All classes from the new dataset
    folder_map = {
        'hole': 'hole',
        'stain': 'stain',
        'lines': 'lines',
        'Needle mark': 'Needle mark',
        'Pinched fabric': 'Pinched fabric'
    }
    
    processed_count = 0
    
    for cls_idx, (cls_name, folder_name) in enumerate(folder_map.items()):
        folder_path = os.path.join(BASE_DATA_DIR, folder_name)
        if not os.path.exists(folder_path): continue
            
        images = glob.glob(os.path.join(folder_path, "*.jpg"))
        print(f"Processing '{cls_name}'...")
        
        for i, img_path in enumerate(tqdm(images)):
            img = cv2.imread(img_path)
            if img is None: continue
            
            bbox = find_defect_bbox(img)
            if bbox is None: continue
                
            x, y, w, h = bbox
            H, W = img.shape[:2]
            
            xc, yc, bw, bh = (x + w/2)/W, (y + h/2)/H, w/W, h/H
            
            is_val = random.random() < 0.2
            target_img_dir = val_img_out if is_val else img_out
            target_lbl_dir = val_lbl_out if is_val else lbl_out
            
            out_name = f"pro_{cls_name}_{i}"
            cv2.imwrite(str(target_img_dir / f"{out_name}.jpg"), img)
            with open(target_lbl_dir / f"{out_name}.txt", "w") as f:
                # USE THE CORRECT CLASS INDEX
                f.write(f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                
            processed_count += 1

    # Update YAML with ALL classes
    yaml_content = {
        'path': os.path.abspath(OUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(folder_map),
        'names': list(folder_map.keys())
    }
    import yaml
    with open('pro_data.yaml', 'w') as f:
        yaml.dump(yaml_content, f)

    print(f"\nDone! Generated {processed_count} labeled images in {OUT_DIR}")
    

if __name__ == "__main__":
    generate_pro_dataset()
