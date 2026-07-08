import os
import cv2
import numpy as np
import random
import glob
import yaml
from pathlib import Path
from tqdm import tqdm

# Input paths
RAW_DIR = "scratch/lusitano_raw"
RAW_DEFECTS = os.path.join(RAW_DIR, "defects")
RAW_NORMALS = os.path.join(RAW_DIR, "non-defects")

# Output dataset path
OUT_DIR = "datasets/lusitano_yolo"
train_img_dir = os.path.join(OUT_DIR, "images/train")
train_lbl_dir = os.path.join(OUT_DIR, "labels/train")
val_img_dir = os.path.join(OUT_DIR, "images/val")
val_lbl_dir = os.path.join(OUT_DIR, "labels/val")

PATCH_SIZE = 640

def find_defect_bbox(img):
    """
    Locates the bounding box of a defect using edge and threshold analysis on normalized image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge Detection
    edges = cv2.Canny(blurred, 30, 100)
    
    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    combined = cv2.bitwise_or(edges, thresh)
    kernel = np.ones((5,5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    H, W = img.shape[:2]
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 8 < w < W * 0.8 and 8 < h < H * 0.8:
            valid_contours.append(cnt)
            
    if not valid_contours:
        return None
        
    cnt = max(valid_contours, key=cv2.contourArea)
    return cv2.boundingRect(cnt)

def main():
    print("[Prepare] Re-creating directories...")
    # Clear existing to avoid mixup
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        if os.path.exists(d):
            shutil_path = Path(d)
            for f in shutil_path.glob('*'):
                if f.is_file(): f.unlink()
        os.makedirs(d, exist_ok=True)
    
    random.seed(42)
    
    defect_images = glob.glob(os.path.join(RAW_DEFECTS, "*.jpg"))
    normal_images = glob.glob(os.path.join(RAW_NORMALS, "*.jpg"))
    
    print(f"[Prepare] Found {len(defect_images)} raw defects and {len(normal_images)} raw normals.")
    
    patches = []
    
    print("[Prepare] Processing defect images with contrast stretching...")
    for idx, path in enumerate(tqdm(defect_images)):
        raw_img = cv2.imread(path)
        if raw_img is None:
            continue
            
        # Apply min-max normalization to stretch contrast to [0, 255]
        img = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX)
        
        bbox = find_defect_bbox(img)
        H, W = img.shape[:2]
        
        if bbox is None:
            cx, cy = W // 2, H // 2
            w_box, h_box = 50, 50
            x_box, y_box = cx - 25, cy - 25
        else:
            x_box, y_box, w_box, h_box = bbox
            cx = x_box + w_box // 2
            cy = y_box + h_box // 2
            
        # Crop centered at defect, with a random shift for spatial augmentation
        shift_x = random.randint(-80, 80)
        shift_y = random.randint(-40, 40)
        
        px1 = int(np.clip(cx - PATCH_SIZE // 2 + shift_x, 0, W - PATCH_SIZE))
        py1 = int(np.clip(cy - PATCH_SIZE // 2 + shift_y, 0, H - PATCH_SIZE))
        px2 = px1 + PATCH_SIZE
        py2 = py1 + PATCH_SIZE
        
        crop = img[py1:py2, px1:px2]
        
        # Calculate new bbox coordinates inside the crop
        new_x1 = max(0, x_box - px1)
        new_y1 = max(0, y_box - py1)
        new_x2 = min(PATCH_SIZE, x_box + w_box - px1)
        new_y2 = min(PATCH_SIZE, y_box + h_box - py1)
        
        new_w = new_x2 - new_x1
        new_h = new_y2 - new_y1
        
        if new_w > 3 and new_h > 3:
            new_cx = new_x1 + new_w / 2
            new_cy = new_y1 + new_h / 2
            
            x_norm = new_cx / PATCH_SIZE
            y_norm = new_cy / PATCH_SIZE
            w_norm = new_w / PATCH_SIZE
            h_norm = new_h / PATCH_SIZE
            
            label_text = f"0 {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n"
            patches.append((crop, label_text, f"defect_{idx}"))
        else:
            patches.append((crop, None, f"defect_{idx}_bg"))
            
    print("[Prepare] Processing normal images with contrast stretching...")
    for idx, path in enumerate(tqdm(normal_images)):
        raw_img = cv2.imread(path)
        if raw_img is None:
            continue
            
        img = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX)
        
        H, W = img.shape[:2]
        px1 = random.randint(0, W - PATCH_SIZE)
        py1 = random.randint(0, H - PATCH_SIZE)
        crop = img[py1:py1+PATCH_SIZE, px1:px1+PATCH_SIZE]
        
        patches.append((crop, None, f"normal_{idx}"))
        
    # Shuffle and split into Train (80%) and Val (20%)
    random.shuffle(patches)
    split_idx = int(len(patches) * 0.8)
    train_patches = patches[:split_idx]
    val_patches = patches[split_idx:]
    
    print(f"[Prepare] Splitting dataset: {len(train_patches)} train, {len(val_patches)} val.")
    
    def save_patches(patch_list, img_dir, lbl_dir):
        for crop, label_text, name in patch_list:
            cv2.imwrite(os.path.join(img_dir, f"{name}.jpg"), crop)
            label_path = os.path.join(lbl_dir, f"{name}.txt")
            if label_text is not None:
                with open(label_path, "w") as f:
                    f.write(label_text)
            else:
                open(label_path, "w").close()
                
    save_patches(train_patches, train_img_dir, train_lbl_dir)
    save_patches(val_patches, val_img_dir, val_lbl_dir)
    
    # Write dataset YAML (5 classes for pre-trained compatibility)
    yaml_content = {
        "path": os.path.abspath(OUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "nc": 5,
        "names": ["hole", "stain", "lines", "Needle mark", "Pinched fabric"]
    }
    
    yaml_path = os.path.join(OUT_DIR, "lusitano_data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    print(f"[Prepare] Complete! Normalized dataset saved in {OUT_DIR}")

if __name__ == "__main__":
    main()
