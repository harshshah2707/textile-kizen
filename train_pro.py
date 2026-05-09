from ultralytics import YOLO
import os

def train_pro_model():
    # 1. Load the best existing model as a starting point
    # We'll start from the synthetic model
    model_path = 'runs/textile_detection/defect_model_v1/weights/best.pt'
    if os.path.exists(model_path):
        print(f"Starting from synthetic model: {model_path}")
        model = YOLO(model_path)
    else:
        model = YOLO('yolov8s.pt')

    # 3. Train on the 756 high-quality real images
    # We specify the data yaml which has nc: 5. 
    # To avoid the 'class count 1' error, we start fresh or ensure the head is reset.
    model.train(
        data='pro_data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project='runs/textile_detection',
        name='defect_model_pro_v1',
        exist_ok=True,
        lr0=0.01,           # Slightly higher for resetting the head
        patience=20,
        augment=True
    )
    
    print("\nPro Training Complete!")
    print("New model saved at: runs/textile_detection/defect_model_pro_v1/weights/best.pt")

if __name__ == "__main__":
    train_pro_model()
