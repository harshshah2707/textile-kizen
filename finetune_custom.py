from ultralytics import YOLO
import yaml
import os
import shutil
from pathlib import Path

def setup_finetune():
    # 1. Prepare Dataset
    base_path = os.path.abspath('datasets/custom_real')
    train_img_dir = os.path.join(base_path, 'train', 'images')
    train_lbl_dir = os.path.join(base_path, 'train', 'labels')
    val_img_dir = os.path.join(base_path, 'val', 'images')
    val_lbl_dir = os.path.join(base_path, 'val', 'labels')
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
        
    # Copy files (simple split: first 3 for train, last 1 for val)
    images = sorted(list(Path(base_path).glob('images/*.jpg')))
    labels = sorted(list(Path(base_path).glob('labels/*.txt')))
    
    for i in range(len(images)):
        target_dir = 'train' if i < len(images)-1 else 'val'
        shutil.copy(images[i], os.path.join(base_path, target_dir, 'images'))
        shutil.copy(labels[i], os.path.join(base_path, target_dir, 'labels'))

    # 2. Create YAML
    data_config = {
        'path': base_path,
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': ['defect']
    }
    
    with open('custom_data.yaml', 'w') as f:
        yaml.dump(data_config, f)

    # 3. Load Model
    # Start from the synthetic model we already have
    model_path = 'runs/textile_detection/defect_model_v1/weights/best.pt'
    if os.path.exists(model_path):
        print(f"Starting from synthetic model: {model_path}")
        model = YOLO(model_path)
    else:
        print("Synthetic model not found. Starting from base yolov8s.pt")
        model = YOLO('yolov8s.pt')

    # 4. Train
    print("\nStarting Fine-tuning on your REAL images...")
    model.train(
        data='custom_data.yaml',
        epochs=50,
        imgsz=640,
        batch=4,           # Small batch for small dataset
        device=0,
        project='runs/textile_detection',
        name='defect_model_real_custom',
        exist_ok=True,
        lr0=0.001,         # Lower learning rate for fine-tuning
        augment=True       # Enable augmentation to prevent overfitting
    )
    
    print("\nFine-tuning complete!")
    print("New model saved at: runs/textile_detection/defect_model_real_custom/weights/best.pt")

if __name__ == "__main__":
    setup_finetune()
