import cv2
import numpy as np
import os
import glob
from pathlib import Path
import random

# Paths
IMG_DIR = "datasets/custom_real/images"
LBL_DIR = "datasets/custom_real/labels"
OUT_IMG_DIR = "datasets/custom_real/train/images"
OUT_LBL_DIR = "datasets/custom_real/train/labels"

# Ensure output dirs exist
Path(OUT_IMG_DIR).mkdir(parents=True, exist_ok=True)
Path(OUT_LBL_DIR).mkdir(parents=True, exist_ok=True)

def rotate_image_and_boxes(image, boxes, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h))
    
    new_boxes = []
    for box in boxes:
        cls, x_c, y_c, bw, bh = map(float, box.split())
        
        # Convert YOLO to pixel coordinates
        px_c, py_c = x_c * w, y_c * h
        
        # Rotate center point
        point = np.array([px_c, py_c, 1])
        new_point = M.dot(point)
        
        # Normalize back
        nx_c, ny_c = new_point[0] / w, new_point[1] / h
        
        # Keep width/height the same (approximation for small angles)
        # or flip if angle is 90/270
        if angle in [90, 270]:
            nbw, nbh = bh, bw
        else:
            nbw, nbh = bw, bh
            
        # Ensure within bounds
        if 0 < nx_c < 1 and 0 < ny_c < 1:
            new_boxes.append(f"{int(cls)} {nx_c:.6f} {ny_c:.6f} {nbw:.6f} {nbh:.6f}")
            
    return rotated_img, new_boxes

def augment_data(multiplier=100):
    images = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    print(f"Augmenting {len(images)} images by factor of {multiplier}...")

    for img_path in images:
        img_name = Path(img_path).stem
        lbl_path = os.path.join(LBL_DIR, f"{img_name}.txt")
        
        if not os.path.exists(lbl_path): continue
        
        img = cv2.imread(img_path)
        with open(lbl_path, "r") as f:
            boxes = f.read().splitlines()
            
        for i in range(multiplier):
            # Random transformations
            angle = random.choice([0, 90, 180, 270, random.randint(-15, 15)])
            brightness = random.uniform(0.6, 1.4)
            
            aug_img, aug_boxes = rotate_image_and_boxes(img, boxes, angle)
            
            # Brightness
            aug_img = cv2.convertScaleAbs(aug_img, alpha=brightness, beta=0)
            
            # Add noise
            if random.random() > 0.5:
                noise = np.random.normal(0, 5, aug_img.shape).astype(np.uint8)
                aug_img = cv2.add(aug_img, noise)

            # Save
            out_name = f"{img_name}_aug_{i}"
            cv2.imwrite(os.path.join(OUT_IMG_DIR, f"{out_name}.jpg"), aug_img)
            with open(os.path.join(OUT_LBL_DIR, f"{out_name}.txt"), "w") as f:
                f.write("\n".join(aug_boxes))

    print(f"Augmentation complete! Total images in train: {len(glob.glob(os.path.join(OUT_IMG_DIR, '*.jpg')))}")

if __name__ == "__main__":
    augment_data(multiplier=100) # 4 images * 100 = 400 images
