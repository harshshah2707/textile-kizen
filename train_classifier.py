from ultralytics import YOLO
import os

def train_classifier():
    # 1. Load a pretrained YOLOv8 classification model
    # We use 'n' (nano) for maximum speed on the GTX 1650
    model = YOLO('yolov8n-cls.pt') 

    print("Starting Training Classifier on Kaggle Textile Patches...")
    
    # 2. Train the model
    # The 'data' parameter for classification is just the root directory 
    # of your train/val folders.
    results = model.train(
        data=os.path.abspath('datasets/kaggle_textile/images'),
        epochs=20,
        imgsz=64,           # The patches are 64x64
        batch=64,           # Larger batch size possible for small images
        device=0,           # Use GPU
        project='runs/textile_classification',
        name='fabric_classifier_v1',
        exist_ok=True,
        workers=4
    )
    
    print("\nTraining Complete!")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_classifier()
