import h5py
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm

def extract_h5_recursive(h5_path, output_dir):
    """
    Recursively extracts images from a complex .h5 structure.
    Expected structure: class_name -> angle -> (N, 1, H, W) images
    """
    print(f"Extracting from {h5_path}...")
    if not os.path.exists(h5_path):
        print(f"Error: {h5_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_path, 'r') as f:
        # Loop through classes (top-level keys)
        for class_name in f.keys():
            class_group = f[class_name]
            if not isinstance(class_group, h5py.Group):
                continue
                
            print(f"Processing class: {class_name}")
            
            # Loop through angles (sub-keys)
            for angle_name in class_group.keys():
                dataset = class_group[angle_name]
                if not isinstance(dataset, h5py.Dataset):
                    continue
                
                print(f"  Extracting {angle_name}...")
                data = dataset[:] # Load into memory for speed if it fits
                
                # Create output directory for this class
                class_out_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_out_dir, exist_ok=True)
                
                num_images = data.shape[0]
                for i in range(num_images):
                    # Data is (N, 1, H, W) or (N, H, W)
                    if len(data.shape) == 4:
                        img = data[i, 0, :, :] # Take the first channel
                    else:
                        img = data[i, :, :]
                        
                    # Normalize and convert to BGR
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                        
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
                    # Save image
                    img_name = f"{class_name}_{angle_name}_patch_{i:04d}.jpg"
                    cv2.imwrite(os.path.join(class_out_dir, img_name), img_bgr)

if __name__ == "__main__":
    # We will prioritize the 64x64 train dataset
    train_h5 = "datasets/kaggle_textile/matchingtDATASET_train_64.h5"
    test_h5 = "datasets/kaggle_textile/matchingtDATASET_test_64.h5"
    
    output_base = "datasets/kaggle_textile/images"
    
    if os.path.exists(train_h5):
        extract_h5_recursive(train_h5, os.path.join(output_base, "train"))
    
    if os.path.exists(test_h5):
        extract_h5_recursive(test_h5, os.path.join(output_base, "val"))
        
    print("\nExtraction complete! Images are in datasets/kaggle_textile/images")
