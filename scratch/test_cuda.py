import cv2
import numpy as np
import torch
from ultralytics import YOLO

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    model = YOLO("runs/textile_detection/defect_model_pro_v1/weights/best.pt")
    
    # Dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    try:
        print("Running prediction on CUDA with half=False...")
        results = model.predict(source=frame, device='cuda', half=False, imgsz=320)
        print("Success on CUDA!")
    except Exception as e:
        print(f"CUDA Error: {e}")
        try:
            print("Falling back to CPU...")
            results = model.predict(source=frame, device='cpu', imgsz=320)
            print("Success on CPU!")
        except Exception as ex:
            print(f"CPU Error: {ex}")

if __name__ == "__main__":
    main()
