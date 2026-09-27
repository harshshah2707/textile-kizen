"""
finetune_800px.py
=================
High-resolution 800px fine-tuning script to boost defect accuracy.

Strategy:
  - Base Model: Existing 85.5% mAP production checkpoint (runs/textile_detection/defect_model_pro_v1/weights/best.pt)
  - Image Size: 800px (56% higher pixel detail for thin vertical lines & micro-holes)
  - Epochs: 25 (Fast fine-tuning, ~30-40 mins total)
  - Batch: 12 (Optimal VRAM usage ~5.5GB on RTX 3050)
  - Learning Rate: 0.0005 (Fine-tuning rate for gentle weight adjustments)
  - Augmentations: Vertical flip (0.5), Copy-Paste (0.4), Rotation (15°)

Deploy:
  Best checkpoint auto-saved to runs/textile_detection/defect_model_pro_v1/weights/best.pt
"""

import os
import shutil
import time
import torch
from pathlib import Path
from ultralytics import YOLO

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.resolve()
PRETRAINED_W   = BASE_DIR / "runs" / "textile_detection" / "defect_model_pro_v1" / "weights" / "best.pt"
DATASET_YAML   = BASE_DIR / "datasets" / "full_yolo" / "data.yaml"
PROJECT_DIR    = BASE_DIR / "runs" / "textile_detection"
RUN_NAME       = "finetune_800px_v1"
PRODUCTION_W   = PRETRAINED_W


def main():
    print("=" * 70)
    print("  TEXTILE DEFECT DETECTION — 800PX ACCURACY BOOST FINE-TUNING")
    print("=" * 70)
    print(f"  Base Model : {PRETRAINED_W}")
    print(f"  Dataset    : {DATASET_YAML}")
    print(f"  Img Size   : 800px")
    print(f"  Epochs     : 25")
    print(f"  Batch Size : 12")
    print(f"  Optimizer  : AdamW (lr0=0.0005)")
    print("=" * 70)

    # ── Check Base Weights ──────────────────────────────────────────────────
    start_weights = str(PRETRAINED_W) if PRETRAINED_W.exists() else "yolov8m.pt"
    if not PRETRAINED_W.exists():
        print(f"[WARN] Base checkpoint {PRETRAINED_W} not found. Starting from yolov8m.pt")

    # ── Device Check ────────────────────────────────────────────────────────
    device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        print(f"\n  GPU  : {gpu_name} ({vram_gb} GB VRAM)")

    # ── Load Model / Resume ─────────────────────────────────────────────────
    start_time = time.time()
    last_ckpt = None
    for cand in [
        PROJECT_DIR / RUN_NAME / "weights" / "last.pt",
        Path("runs/detect") / PROJECT_DIR / RUN_NAME / "weights" / "last.pt",
        Path("runs/detect/runs/textile_detection") / RUN_NAME / "weights" / "last.pt",
    ]:
        if cand.exists():
            last_ckpt = cand
            break

    if last_ckpt:
        print(f"\n[Fine-Tune] Found existing checkpoint: {last_ckpt}")
        print(f"[Fine-Tune] Resuming training from {last_ckpt} with batch=8...")
        model = YOLO(str(last_ckpt))
        results = model.train(resume=True)
    else:
        print(f"\n[Fine-Tune] Loading weights from: {start_weights}")
        model = YOLO(start_weights)

        # ── Fine-Tune ───────────────────────────────────────────────────────────
        results = model.train(
            data=str(DATASET_YAML),

            # ── Core ──────────────────────────────────────────────────────────
            epochs=25,
            imgsz=800,                  # 800px resolution for micro-defect detail
            batch=8,                    # ~4.5GB VRAM usage on RTX 3050 (rock solid)
            device=device,
            workers=4,
            amp=True,                   # FP16

            # ── Project ───────────────────────────────────────────────────────
            project=str(PROJECT_DIR),
            name=RUN_NAME,
            exist_ok=True,

            # ── Fine-Tuning Optimizer ─────────────────────────────────────────
            optimizer="AdamW",
            lr0=0.0005,                 # Lower LR for fine-tuning existing weights
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=2.0,

            # ── Augmentation tuned for line & vertical defects ─────────────────
            mosaic=1.0,
            mixup=0.2,
            copy_paste=0.4,             # Duplicates rare line defects
            close_mosaic=5,
            fliplr=0.5,
            flipud=0.5,                 # Helps horizontal <-> vertical line invariance
            degrees=15.0,
            translate=0.1,
            scale=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,

            # ── Quality ───────────────────────────────────────────────────────
            patience=10,
            val=True,
            verbose=True,
            seed=42,
        )

    elapsed = time.time() - start_time
    minutes = elapsed / 60

    print(f"\n[Fine-Tune] ✓ Completed in {minutes:.1f} minutes")

    # ── Deploy Best Weights ──────────────────────────────────────────────────
    save_dir = Path(model.trainer.save_dir) if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir') else PROJECT_DIR / RUN_NAME
    best_src = save_dir / "weights" / "best.pt"
    if not best_src.exists():
        for cand in [
            Path("runs/detect/runs/textile_detection") / RUN_NAME / "weights" / "best.pt",
            Path("runs/detect") / RUN_NAME / "weights" / "best.pt",
            PROJECT_DIR / RUN_NAME / "weights" / "best.pt",
        ]:
            if cand.exists():
                best_src = cand
                break

    if best_src.exists():
        os.makedirs(os.path.dirname(PRODUCTION_W), exist_ok=True)
        shutil.copy2(best_src, PRODUCTION_W)
        print(f"[Fine-Tune] ✓ Deployed boosted model to: {PRODUCTION_W}")
    else:
        print(f"[Fine-Tune] ✗ WARNING: best.pt not found at {best_src}")

    # ── Final Metrics ─────────────────────────────────────────────────────────
    m = results.results_dict if hasattr(results, 'results_dict') else {}
    map50    = m.get("metrics/mAP50(B)",    0)
    map5095  = m.get("metrics/mAP50-95(B)", 0)
    prec     = m.get("metrics/precision(B)",0)
    recall   = m.get("metrics/recall(B)",   0)

    print("\n" + "=" * 70)
    print("  FINE-TUNED MODEL METRICS")
    print("=" * 70)
    print(f"  mAP@50         : {map50:.4f}   ({map50*100:.1f}%)")
    print(f"  mAP@50-95      : {map5095:.4f}   ({map5095*100:.1f}%)")
    print(f"  Precision      : {prec:.4f}   ({prec*100:.1f}%)")
    print(f"  Recall         : {recall:.4f}   ({recall*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
