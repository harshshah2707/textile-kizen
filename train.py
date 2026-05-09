"""
Textile Defect Detection - Training Script
=========================================
Trains YOLOv8 small model on synthetic textile data.
Optimized for GTX 1650 (4GB VRAM).
"""

from ultralytics import YOLO
from config import TRAINING_CONFIG, DATASET_DIR, BEST_MODEL_PATH

class TextileDefectTrainer:
    def __init__(self, config=None):
        self.config = config or TRAINING_CONFIG
        self.model = YOLO("yolov8s.pt")  # Load pretrained YOLOv8 small

    def train(self):
        print(f"Starting training on {DATASET_DIR / 'dataset.yaml'}...")
        results = self.model.train(
            data=str(DATASET_DIR / "dataset.yaml"),
            epochs=self.config["epochs"],
            imgsz=self.config["imgsz"],
            batch=self.config["batch_size"],
            device=self.config["device"],
            project=self.config["project"],
            name=self.config["name"],
            exist_ok=self.config["exist_ok"],
            patience=self.config["patience"],
            augment=self.config["augment"]
        )
        print(f"Training complete. Best model saved at: {BEST_MODEL_PATH}")
        return results

if __name__ == "__main__":
    trainer = TextileDefectTrainer()
    trainer.train()
