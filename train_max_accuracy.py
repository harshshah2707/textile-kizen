"""
train_max_accuracy.py
======================
Maximum accuracy YOLOv8x training script for commercial textile defect detection.

Hardware target: RTX 3050 8GB VRAM, i7-12700KF, 32GB RAM
Model: YOLOv8m (Medium — optimal balance of speed and high accuracy)
Image size: 640px (standard high-speed resolution, ideal for line scan tiling)
AMP: Enabled (RTX 3050 FP16 → ultra-fast training ~2-3 mins/epoch)
Batch: 16 — Fast parallel batching on 8GB VRAM
Workers: 4 — Fast DataLoader throughput
Optimizer: AdamW (superior convergence on class-imbalanced datasets)

Expected training time: ~8-14 hours on RTX 3050 at 1280px
Expected mAP@50: 0.78-0.90 with merged dataset

Deployment:
  Best weights → runs/textile_detection/defect_model_pro_v1/weights/best.pt
  (Keeps compatibility with existing app.py and web_server.py)
"""

import os
import shutil
import time
import torch
from pathlib import Path
from ultralytics import YOLO

# ─── Paths ──────────────────────────────────────────────────────────────────
DATASET_YAML   = "datasets/full_yolo/data.yaml"
PROJECT_DIR    = "runs/textile_detection"
RUN_NAME       = "max_accuracy_v1"
PRODUCTION_W   = "runs/textile_detection/defect_model_pro_v1/weights/best.pt"

# Fallback to original multiclass dataset if full_yolo hasn't been prepared yet
if not os.path.exists(DATASET_YAML):
    print("[WARN] full_yolo dataset not found — run prepare_full_dataset.py first")
    print("[WARN] Falling back to existing multiclass_yolo dataset")
    DATASET_YAML = "datasets/multiclass_yolo/data.yaml"


def main():
    print("=" * 70)
    print("  TEXTILE DEFECT DETECTION — MAX ACCURACY TRAINING")
    print("=" * 70)
    print(f"  Dataset   : {DATASET_YAML}")
    print("  Model     : yolov8m.pt (medium — fast & accurate)")
    print("  Img size  : 640px")
    print(f"  Optimizer : AdamW")
    print(f"  AMP       : True (FP16)")
    print("=" * 70)

    # ── Device ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        print(f"\n  GPU  : {gpu_name}")
        print(f"  VRAM : {vram_gb} GB")
        device = 0
    else:
        print("\n  [WARN] No CUDA GPU found — training on CPU (very slow!)")
        device = "cpu"

    # ── Load Model ──────────────────────────────────────────────────────────
    # Always start from COCO-pretrained YOLOv8l for best transfer learning
    print(f"\n[Train] Loading YOLOv8m pretrained weights...")
    model = YOLO("yolov8m.pt")

    start_time = time.time()

    # ── Train ────────────────────────────────────────────────────────────────
    results = model.train(
        data=DATASET_YAML,

        # ── Core ──────────────────────────────────────────────────────────
        epochs=100,
        imgsz=640,                  # Fast 640px resolution for high-speed training & line-scan tiling
        batch=16,                   # Safe & fast batch size on RTX 3050 8GB
        device=device,
        workers=4,                  # Windows: keep low to avoid pin_memory OOM
        amp=True,                   # FP16 — safe on RTX 3050

        # ── Project ───────────────────────────────────────────────────────
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,

        # ── Optimizer ─────────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,                  # AdamW initial LR
        lrf=0.0001,                 # Final LR = lr0 * lrf (cosine decay)
        momentum=0.937,             # Beta1 for AdamW
        weight_decay=0.0005,
        warmup_epochs=5.0,          # Longer warmup for AdamW stability
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # ── Augmentation (maximum for small dataset) ───────────────────────
        mosaic=1.0,                 # Full mosaic augmentation
        mixup=0.3,                  # Mixup between images
        copy_paste=0.3,             # Copy-paste augmentation (great for small datasets)
        close_mosaic=30,            # Disable mosaic last 30 epochs for convergence
        fliplr=0.5,                 # Horizontal flip
        flipud=0.2,                 # Vertical flip (fabrics can be upside-down)
        degrees=15.0,               # Rotation (fabric can enter scanner at slight angle)
        translate=0.1,
        scale=0.5,                  # Scale ± 50%
        shear=5.0,                  # Shear for perspective simulation
        perspective=0.0005,         # Mild perspective distortion
        hsv_h=0.015,                # Hue jitter
        hsv_s=0.7,                  # Saturation jitter
        hsv_v=0.4,                  # Value/brightness jitter
        erasing=0.4,                # Random erasing (occlusion simulation)

        # ── Training Quality ───────────────────────────────────────────────
        patience=25,                # Patience for 100-epoch training
        val=True,
        verbose=True,
        seed=42,
        deterministic=True,

        # ── Label Smoothing ───────────────────────────────────────────────
        label_smoothing=0.1,        # Prevents overconfidence on minority classes

        # ── NMS at validation ─────────────────────────────────────────────
        nms=True,

        # ── Checkpointing ─────────────────────────────────────────────────
        save=True,
        save_period=20,             # Save checkpoint every 20 epochs
    )

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes = rem // 60

    print(f"\n[Train] ✓ Training complete in {int(hours)}h {int(minutes)}m")

    # ── Deploy Best Weights ──────────────────────────────────────────────────
    save_dir = Path(model.trainer.save_dir) if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir') else Path(PROJECT_DIR) / RUN_NAME
    best_src = save_dir / "weights" / "best.pt"
    
    # Fallback checks if save_dir moved
    if not best_src.exists():
        fallback = Path("runs/detect") / PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
        if fallback.exists():
            best_src = fallback

    if best_src.exists():
        os.makedirs(os.path.dirname(PRODUCTION_W), exist_ok=True)
        shutil.copy2(best_src, PRODUCTION_W)
        print(f"[Train] ✓ Deployed to production: {PRODUCTION_W}")
    else:
        print(f"[Train] ✗ WARNING: best.pt was not found at {best_src}")
        return

    # ── Final Metrics ─────────────────────────────────────────────────────────
    m = results.results_dict
    map50    = m.get("metrics/mAP50(B)",    0)
    map5095  = m.get("metrics/mAP50-95(B)", 0)
    prec     = m.get("metrics/precision(B)",0)
    recall   = m.get("metrics/recall(B)",   0)

    print("\n" + "=" * 70)
    print("  FINAL TRAINING METRICS")
    print("=" * 70)
    print(f"  mAP@50         : {map50:.4f}   ({map50*100:.1f}%)")
    print(f"  mAP@50-95      : {map5095:.4f}   ({map5095*100:.1f}%)")
    print(f"  Precision      : {prec:.4f}   ({prec*100:.1f}%)")
    print(f"  Recall         : {recall:.4f}   ({recall*100:.1f}%)")
    print("=" * 70)
    print("\n  Next step → Run: python evaluate_model.py")
    print("  This will generate per-class metrics and confusion matrix.\n")


if __name__ == "__main__":
    main()
