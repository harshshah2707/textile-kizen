# finetune_on_kaggle.py
from ultralytics import YOLO
import yaml
import os

def run_finetuning():
    # 1. Create data.yaml
    data_config = {
        'path': os.path.abspath('datasets/kaggle_yolo_split'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': ['defect']
    }
    
    with open('kaggle_data.yaml', 'w') as f:
        yaml.dump(data_config, f)

    # 2. Load existing model (Start from synthetic knowledge)
    model_path = 'runs/textile_detection/defect_model_v1/weights/best.pt'
    if not os.path.exists(model_path):
        print("Synthetic model not found. Starting from base yolov8s.pt")
        model = YOLO('yolov8s.pt')
    else:
        model = YOLO(model_path)

    # 3. Fine-tune
    print("Starting fine-tuning on REAL Kaggle data...")
    model.train(
        data='kaggle_data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        lr0=0.001,          # Lower learning rate for fine-tuning
        patience=15,         # Early stopping
        project='runs/textile_detection',
        name='defect_model_kaggle',
        device=0,            # Use GPU
        augment=True,
        mosaic=1.0,
        mixup=0.1            # Add mixup for better generalization
    )

if __name__ == "__main__":
    run_finetuning()
