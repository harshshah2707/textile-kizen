"""
Final training script for Lusitano textile defect detection.
Strategy:
  - nc=1 (single class: "defect") to match dataset labels
  - Start from yolov8s.pt (COCO pre-trained for feature extraction)
  - 60 epochs with cosine LR schedule
  - Heavy augmentation: mosaic, flips, scale, HSV
  - AMP disabled (GTX 1650 NaN fix)
  - Deploy best.pt to production path
"""
import os
import shutil
import torch
from ultralytics import YOLO

DATASET_YAML = "datasets/lusitano_yolo/lusitano_data.yaml"
PROJECT_DIR  = "runs/textile_detection"
RUN_NAME     = "lusitano_v2"
PRODUCTION_WEIGHTS = "runs/textile_detection/defect_model_pro_v1/weights/best.pt"

def main():
    print("[Train] Starting Lusitano v2 training...")

    # Always start from COCO yolov8s — clean head for nc=1
    weights = "yolov8s.pt"
    print(f"[Train] Base weights: {weights}")

    model = YOLO(weights)

    device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[Train] Running on CPU")

    results = model.train(
        data=DATASET_YAML,
        epochs=60,
        imgsz=640,
        batch=8,
        device=device,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,

        # Optimizer
        optimizer="SGD",        # SGD is more stable than AdamW for YOLO from-scratch head
        lr0=0.01,               # Standard YOLO lr
        lrf=0.01,               # Final LR fraction (cosine decay)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,

        # Augmentation (aggressive to compensate for small dataset)
        mosaic=1.0,             # Mosaic augmentation - critical for small datasets
        mixup=0.1,              # Light mixup
        fliplr=0.5,
        flipud=0.1,
        scale=0.5,              # Random scale ±50%
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,           # Random rotation ±10°
        translate=0.1,

        # Stability
        amp=False,              # GTX 1650 NaN fix
        patience=15,
        val=True,
        verbose=True,
        close_mosaic=10,        # Disable mosaic in last 10 epochs for fine convergence
    )

    print("\n[Train] Training complete!")

    # Deploy best weights
    best_src = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    if os.path.exists(best_src):
        os.makedirs(os.path.dirname(PRODUCTION_WEIGHTS), exist_ok=True)
        shutil.copy2(best_src, PRODUCTION_WEIGHTS)
        print(f"[Train] Deployed: {PRODUCTION_WEIGHTS}")

        # Print final metrics
        metrics = results.results_dict
        map50 = metrics.get("metrics/mAP50(B)", 0)
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0)
        precision = metrics.get("metrics/precision(B)", 0)
        recall = metrics.get("metrics/recall(B)", 0)
        print(f"\n[Train] === Final Metrics ===")
        print(f"  mAP@50:       {map50:.4f}")
        print(f"  mAP@50-95:    {map50_95:.4f}")
        print(f"  Precision:    {precision:.4f}")
        print(f"  Recall:       {recall:.4f}")
    else:
        print(f"[Train] WARNING: best.pt not found at {best_src}")

if __name__ == "__main__":
    main()
