# analyze_kaggle.py
import os
import cv2
import numpy as np
from pathlib import Path

dataset_path = 'datasets/kaggle_textile'
images_path = os.path.join(dataset_path, 'images')

def analyze():
    print("="*60)
    print("  KAGGLE DATASET ANALYSIS")
    print("="*60)
    
    if not os.path.exists(images_path):
        print(f"ERROR: Dataset not found at {images_path}")
        return

    # Count images
    image_files = list(Path(images_path).glob('*.jpg')) + list(Path(images_path).glob('*.png'))
    print(f"Total images found: {len(image_files)}")

    # Check image sizes
    sizes = []
    for img_path in image_files[:50]:
        img = cv2.imread(str(img_path))
        if img is not None:
            sizes.append(img.shape)

    if sizes:
        print(f"Image resolutions found: {set(sizes)}")
        avg_h = np.mean([s[0] for s in sizes])
        avg_w = np.mean([s[1] for s in sizes])
        print(f"Average resolution: {int(avg_w)}x{int(avg_h)}")

    # Check labels
    labels_path = os.path.join(dataset_path, 'labels')
    if os.path.exists(labels_path):
        label_files = list(Path(labels_path).glob('*.txt'))
        print(f"Total label files: {len(label_files)}")
        if label_files:
            with open(label_files[0], 'r') as f:
                print(f"\nSample label (first 100 chars):\n{f.read(100)}")
    else:
        print("\nWARNING: No 'labels' folder found. Conversion may be required.")

if __name__ == "__main__":
    analyze()
