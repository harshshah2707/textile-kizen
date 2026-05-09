# split_dataset.py
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_data(source_path, output_path, train_ratio=0.7, val_ratio=0.15):
    images_src = Path(source_path) / "images"
    labels_src = Path(source_path) / "labels"
    
    if not images_src.exists() or not labels_src.exists():
        print("Error: Source images or labels folder missing.")
        return

    # Get all image files
    all_images = list(images_src.glob("*.jpg")) + list(images_src.glob("*.png"))
    random.shuffle(all_images)
    
    total = len(all_images)
    train_idx = int(total * train_ratio)
    val_idx = int(total * (train_ratio + val_ratio))
    
    splits = {
        'train': all_images[:train_idx],
        'val': all_images[train_idx:val_idx],
        'test': all_images[val_idx:]
    }
    
    for split_name, split_files in splits.items():
        # Create directories
        (Path(output_path) / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (Path(output_path) / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        
        print(f"Copying {split_name} files...")
        for img_p in tqdm(split_files):
            # Copy image
            shutil.copy(img_p, Path(output_path) / "images" / split_name / img_p.name)
            
            # Copy corresponding label
            lbl_name = img_p.stem + ".txt"
            lbl_p = labels_src / lbl_name
            if lbl_p.exists():
                shutil.copy(lbl_p, Path(output_path) / "labels" / split_name / lbl_name)

    print("\nSplit Complete!")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

if __name__ == "__main__":
    split_data('datasets/kaggle_textile', 'datasets/kaggle_yolo_split')
