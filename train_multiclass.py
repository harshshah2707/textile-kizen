"""
Train YOLOv8s on the Multi-Class Fabric Defect Detection Dataset.
8 defect classes + background. ~3067 images total.
Uses COCO-pretrained yolov8s.pt as backbone, full fine-tune.
"""
import os
import shutil
import torch
from ultralytics import YOLO

DATASET_YAML   = "datasets/multiclass_yolo/data.yaml"
PROJECT_DIR    = "runs/textile_detection"
RUN_NAME       = "multiclass_v1"
PRODUCTION_W   = "runs/textile_detection/defect_model_pro_v1/weights/best.pt"


def main():
    print("[Train] Multi-class textile defect detection training")
    print(f"[Train] Dataset: {DATASET_YAML}")

    model = YOLO("yolov8s.pt")

    device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    results = model.train(
        data=DATASET_YAML,
        epochs=80,
        imgsz=640,
        batch=8,
        device=device,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,

        # Optimizer
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,

        # Augmentation
        mosaic=1.0,
        mixup=0.15,
        fliplr=0.5,
        flipud=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        copy_paste=0.1,

        # Stability
        amp=False,          # GTX 1650 NaN fix
        patience=20,        # More patience for larger dataset
        val=True,
        verbose=True,
        close_mosaic=15,    # Disable mosaic in last 15 epochs
    )

    print("\n[Train] Training complete!")

    best_src = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    if os.path.exists(best_src):
        os.makedirs(os.path.dirname(PRODUCTION_W), exist_ok=True)
        shutil.copy2(best_src, PRODUCTION_W)
        print(f"[Train] Deployed: {PRODUCTION_W}")

        m = results.results_dict
        print(f"\n[Train] === Final Metrics ===")
        print(f"  mAP@50:    {m.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@50-95: {m.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision: {m.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall:    {m.get('metrics/recall(B)', 0):.4f}")
    else:
        print("[Train] WARNING: best.pt not saved!")


if __name__ == "__main__":
    main()
