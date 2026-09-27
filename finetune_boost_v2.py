"""
finetune_boost_v2.py
====================
Continues fine-tuning from the 72.7% mAP@50 checkpoint
(finetune_800px_v1/weights/best.pt) to push accuracy above 80%.

Strategy:
  - Base     : finetune_800px_v1/weights/best.pt  (72.7% mAP@50 @ epoch 10)
  - Image sz : 800px  (same resolution — proven to work)
  - Epochs   : 60  (gives ample room to converge, early-stopped if plateau)
  - Patience : 20  (was 10 before — prevented good convergence)
  - LR       : 0.0002 (lower than before; model weights are already warm)
  - cos_lr   : True  (cosine annealing — smooth decay, better final accuracy)
  - Batch    : 8    (safe on RTX 3050 8GB at 800px)
  - Freeze   : backbone for first 10 layers via freeze=10
  - Deploy   : Best checkpoint → runs/textile_detection/defect_model_pro_v1/weights/best.pt
"""

import os
import shutil
import time
import json
import torch
from pathlib import Path
from ultralytics import YOLO

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.resolve()
BASE_WEIGHTS  = BASE_DIR / "runs" / "textile_detection" / "finetune_800px_v1" / "weights" / "best.pt"
DATASET_YAML  = BASE_DIR / "datasets" / "full_yolo" / "data.yaml"
PROJECT_DIR   = BASE_DIR / "runs" / "textile_detection"
RUN_NAME      = "finetune_boost_v2"
PRODUCTION_W  = BASE_DIR / "runs" / "textile_detection" / "defect_model_pro_v1" / "weights" / "best.pt"


def main():
    print("=" * 70)
    print("  TEXTILE DEFECT DETECTION — ACCURACY BOOST FINE-TUNE v2")
    print("=" * 70)
    print(f"  Base Model : {BASE_WEIGHTS}")
    print(f"  Dataset    : {DATASET_YAML}")
    print(f"  Img Size   : 800px")
    print(f"  Epochs     : 60  (patience=20, stops early if plateau)")
    print(f"  Batch Size : 8")
    print(f"  LR         : 0.0002 -> cosine decay to 1e-5")
    print(f"  cos_lr     : True")
    print("=" * 70)

    # ── Validation ──────────────────────────────────────────────────────────
    if not BASE_WEIGHTS.exists():
        print(f"[ERROR] Base weights not found: {BASE_WEIGHTS}")
        print("        Run finetune_800px.py first to generate this checkpoint.")
        return

    if not DATASET_YAML.exists():
        print(f"[ERROR] Dataset YAML not found: {DATASET_YAML}")
        print("        Run prepare_full_dataset.py first.")
        return

    # ── Device ──────────────────────────────────────────────────────────────
    device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        print(f"\n  GPU  : {gpu_name} ({vram_gb} GB VRAM)")
        if vram_gb < 6:
            print("  [WARN] Low VRAM detected — reducing batch to 4")
            batch = 4
        elif vram_gb < 8:
            batch = 6
        else:
            batch = 8
    else:
        print("\n  [WARN] No CUDA GPU found — training on CPU (slow!)")
        batch = 2

    print(f"  Batch      : {batch}")

    # ── Check for existing partial run to resume ─────────────────────────────
    last_ckpt = PROJECT_DIR / RUN_NAME / "weights" / "last.pt"
    if last_ckpt.exists():
        print(f"\n[Boost] Found existing checkpoint: {last_ckpt}")
        print(f"[Boost] Resuming from last.pt...")
        model = YOLO(str(last_ckpt))
        start_time = time.time()
        results = model.train(resume=True)
    else:
        print(f"\n[Boost] Loading base weights: {BASE_WEIGHTS}")
        model = YOLO(str(BASE_WEIGHTS))
        start_time = time.time()

        results = model.train(
            data=str(DATASET_YAML),

            # ── Core ──────────────────────────────────────────────────────
            epochs=60,
            imgsz=800,
            batch=batch,
            device=device,
            workers=4,
            amp=True,                   # FP16

            # ── Project ───────────────────────────────────────────────────
            project=str(PROJECT_DIR),
            name=RUN_NAME,
            exist_ok=True,

            # ── Optimizer (fine-tuning regime) ─────────────────────────────
            optimizer="AdamW",
            lr0=0.0002,                 # Lower LR — weights are already warm
            lrf=0.05,                   # Final LR = lr0 * lrf (cosine decay)
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.05,
            cos_lr=True,                # Cosine LR schedule — better convergence

            # ── Freeze backbone ────────────────────────────────────────────
            freeze=10,                  # Freeze first 10 backbone layers

            # ── Augmentation (moderate — model already learned features) ────
            mosaic=0.8,
            mixup=0.15,
            copy_paste=0.3,
            close_mosaic=10,
            fliplr=0.5,
            flipud=0.5,
            degrees=10.0,
            translate=0.1,
            scale=0.4,
            shear=2.0,
            hsv_h=0.015,
            hsv_s=0.6,
            hsv_v=0.4,
            erasing=0.3,

            # ── Loss weights ──────────────────────────────────────────────
            cls=0.6,
            box=7.5,
            dfl=1.5,
            label_smoothing=0.05,

            # ── Quality ───────────────────────────────────────────────────
            patience=20,                # Was 10 before — key improvement
            val=True,
            verbose=True,
            seed=42,
            save=True,
            save_period=10,
            nms=True,
        )

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes = rem // 60
    print(f"\n[Boost] Training complete in {int(hours)}h {int(minutes)}m")

    # ── Find and deploy best weights ─────────────────────────────────────────
    save_dir = (
        Path(model.trainer.save_dir)
        if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir")
        else PROJECT_DIR / RUN_NAME
    )
    best_src = save_dir / "weights" / "best.pt"

    if not best_src.exists():
        for cand in [
            PROJECT_DIR / RUN_NAME / "weights" / "best.pt",
            Path("runs/detect") / str(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt",
        ]:
            if cand.exists():
                best_src = cand
                break

    if best_src.exists():
        PRODUCTION_W.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_src, PRODUCTION_W)
        print(f"[Boost] Deployed boosted model -> {PRODUCTION_W}")
    else:
        print(f"[Boost] WARNING: best.pt not found at {best_src}")

    # ── Print final metrics ──────────────────────────────────────────────────
    m = results.results_dict if hasattr(results, "results_dict") else {}
    map50   = m.get("metrics/mAP50(B)",    0)
    map5095 = m.get("metrics/mAP50-95(B)", 0)
    prec    = m.get("metrics/precision(B)", 0)
    recall  = m.get("metrics/recall(B)",    0)
    f1      = 2 * prec * recall / max(prec + recall, 1e-6)

    print("\n" + "=" * 70)
    print("  BOOST v2 FINAL METRICS")
    print("=" * 70)
    print(f"  mAP@50         : {map50:.4f}   ({map50*100:.1f}%)  [was 72.7%]")
    print(f"  mAP@50-95      : {map5095:.4f}   ({map5095*100:.1f}%)")
    print(f"  Precision      : {prec:.4f}   ({prec*100:.1f}%)")
    print(f"  Recall         : {recall:.4f}   ({recall*100:.1f}%)")
    print(f"  F1 Score       : {f1:.4f}   ({f1*100:.1f}%)")
    print(f"  Training time  : {int(hours)}h {int(minutes)}m")
    print("=" * 70)

    # ── Save summary ────────────────────────────────────────────────────────
    summary = {
        "base_model": str(BASE_WEIGHTS),
        "base_mAP50": 0.727,
        "final_mAP50": round(map50, 4),
        "final_mAP50_95": round(map5095, 4),
        "precision": round(prec, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "epochs": 60,
        "imgsz": 800,
        "batch": batch,
        "training_time_h": round(elapsed / 3600, 2),
        "production_weights": str(PRODUCTION_W),
    }
    sp = save_dir / "boost_summary.json"
    sp.write_text(json.dumps(summary, indent=2))
    print(f"\n[Summary] Written to: {sp}")
    print("\n  Next steps:")
    print("  1. python evaluate_model.py")
    print("  2. python export_linescan_engine.py --format engine --batch 8")
    print("  3. Restart web_server.py (new best.pt is live)\n")


if __name__ == "__main__":
    main()
