"""
evaluate_model.py
=================
Comprehensive accuracy evaluation for the trained textile defect model.

Reports:
  - Overall mAP@50, mAP@50-95, Precision, Recall
  - Per-class breakdown (mAP, P, R, F1)
  - Confusion matrix saved to outputs/confusion_matrix.png
  - Full report saved to outputs/accuracy_report.txt
  - Speed benchmark (ms/image)

Usage:
  python evaluate_model.py
  python evaluate_model.py --model path/to/custom.pt
  python evaluate_model.py --data datasets/full_yolo/data.yaml
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.resolve()
DEFAULT_MODEL  = BASE_DIR / "runs" / "textile_detection" / "defect_model_pro_v1" / "weights" / "best.pt"
DEFAULT_DATA   = BASE_DIR / "datasets" / "full_yolo" / "data.yaml"
OUTPUT_DIR     = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Broken stitch",
    "hole",
    "horizontal",
    "lines",
    "Needle mark",
    "Pinched fabric",
    "stain",
    "Vertical",
]

# ─── CLI Args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate textile defect detection model")
parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
parser.add_argument("--data",  type=str, default=str(DEFAULT_DATA))
parser.add_argument("--imgsz", type=int, default=1280)
parser.add_argument("--conf",  type=float, default=0.25)
parser.add_argument("--iou",   type=float, default=0.45)
args = parser.parse_args()

# ─── Fallback dataset path ─────────────────────────────────────────────────
if not os.path.exists(args.data):
    fallback = str(BASE_DIR / "datasets" / "multiclass_yolo" / "data.yaml")
    print(f"[Eval] full_yolo data.yaml not found, falling back to: {fallback}")
    args.data = fallback


def hr(char="─", width=70):
    return char * width


def main():
    print("\n" + hr("═"))
    print("  TEXTILE DEFECT MODEL — ACCURACY EVALUATION")
    print(hr("═"))
    print(f"  Model : {args.model}")
    print(f"  Data  : {args.data}")
    print(f"  ImgSz : {args.imgsz}px")
    print(f"  Conf  : {args.conf}")
    print(f"  IOU   : {args.iou}")
    print(hr("─"))

    # ── Load model ─────────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"\n[ERROR] Model not found: {args.model}")
        print("  → Train first: python train_max_accuracy.py")
        sys.exit(1)

    device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")

    print("\n[Eval] Loading model...")
    model = YOLO(args.model)

    # ── Run validation ──────────────────────────────────────────────────────
    print("[Eval] Running validation on dataset...\n")
    t0 = time.time()
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        verbose=True,
        save_json=True,
        plots=True,
        project=str(OUTPUT_DIR),
        name="eval_run",
        exist_ok=True,
    )
    elapsed = time.time() - t0

    # ── Extract results ────────────────────────────────────────────────────
    rd = metrics.results_dict

    map50    = rd.get("metrics/mAP50(B)",    0)
    map5095  = rd.get("metrics/mAP50-95(B)", 0)
    prec     = rd.get("metrics/precision(B)", 0)
    recall   = rd.get("metrics/recall(B)",   0)

    # Per-class metrics from metrics object
    # ultralytics stores per-class AP in metrics.ap_class_index and metrics.box.ap
    try:
        class_indices = metrics.ap_class_index         # array of class ids
        ap50_per_cls  = metrics.box.ap50               # per-class AP@50
        ap_per_cls    = metrics.box.ap                 # per-class AP@50-95
        p_per_cls     = metrics.box.p                  # per-class precision
        r_per_cls     = metrics.box.r                  # per-class recall
        has_per_class = True
    except Exception:
        has_per_class = False

    # ── Speed benchmark ─────────────────────────────────────────────────────
    # Quick speed test on a single batch
    speed_info = metrics.speed  # dict with preprocess, inference, postprocess
    inf_ms = speed_info.get("inference", 0)

    # ── Build report ───────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    lines.append("=" * 70)
    lines.append("  TEXTILE DEFECT DETECTION — ACCURACY REPORT")
    lines.append(f"  Generated : {timestamp}")
    lines.append(f"  Model     : {Path(args.model).name}")
    lines.append(f"  Dataset   : {args.data}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("  OVERALL METRICS")
    lines.append("─" * 70)
    lines.append(f"  mAP@50          : {map50:.4f}   ({map50*100:.2f}%)")
    lines.append(f"  mAP@50-95       : {map5095:.4f}   ({map5095*100:.2f}%)")
    lines.append(f"  Precision       : {prec:.4f}   ({prec*100:.2f}%)")
    lines.append(f"  Recall          : {recall:.4f}   ({recall*100:.2f}%)")
    f1 = 2 * (prec * recall) / (prec + recall + 1e-9)
    lines.append(f"  F1 Score        : {f1:.4f}   ({f1*100:.2f}%)")
    lines.append(f"  Inference speed : {inf_ms:.1f} ms/image ({1000/max(inf_ms,0.1):.0f} FPS)")
    lines.append(f"  Eval time       : {elapsed:.1f}s")
    lines.append("")

    if has_per_class:
        lines.append("  PER-CLASS METRICS (mAP@50 | mAP@50-95 | Precision | Recall | F1)")
        lines.append("─" * 70)
        lines.append(f"  {'Class':<18} {'AP50':>8} {'AP50-95':>9} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        lines.append("  " + "─" * 66)
        for i, cls_idx in enumerate(class_indices):
            name    = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
            ap50_c  = float(ap50_per_cls[i])  if i < len(ap50_per_cls) else 0
            ap_c    = float(ap_per_cls[i])    if i < len(ap_per_cls)   else 0
            p_c     = float(p_per_cls[i])     if i < len(p_per_cls)    else 0
            r_c     = float(r_per_cls[i])     if i < len(r_per_cls)    else 0
            f1_c    = 2 * (p_c * r_c) / (p_c + r_c + 1e-9)
            lines.append(
                f"  {name:<18} {ap50_c:>8.3f} {ap_c:>9.3f} {p_c:>8.3f} {r_c:>8.3f} {f1_c:>8.3f}"
            )
        lines.append("")

    lines.append("  CONFUSION MATRIX")
    lines.append("─" * 70)
    lines.append(f"  Saved to : {OUTPUT_DIR / 'eval_run' / 'confusion_matrix.png'}")
    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    # ── Print to console ───────────────────────────────────────────────────
    print("\n" + report_text)

    # ── Save to file ───────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / "accuracy_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[Eval] ✓ Report saved: {report_path}")

    # ── Save JSON metrics ──────────────────────────────────────────────────
    json_metrics = {
        "timestamp": timestamp,
        "model": str(args.model),
        "data": args.data,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "overall": {
            "mAP50": round(map50, 4),
            "mAP50_95": round(map5095, 4),
            "precision": round(prec, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "speed_ms": inf_ms,
        "per_class": {}
    }

    if has_per_class:
        for i, cls_idx in enumerate(class_indices):
            name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
            json_metrics["per_class"][name] = {
                "ap50":      round(float(ap50_per_cls[i]), 4) if i < len(ap50_per_cls) else 0,
                "ap50_95":   round(float(ap_per_cls[i]),   4) if i < len(ap_per_cls)   else 0,
                "precision": round(float(p_per_cls[i]),    4) if i < len(p_per_cls)    else 0,
                "recall":    round(float(r_per_cls[i]),    4) if i < len(r_per_cls)    else 0,
            }

    json_path = OUTPUT_DIR / "accuracy_metrics.json"
    with open(json_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"[Eval] ✓ JSON metrics: {json_path}")
    print(f"[Eval] ✓ Plots & confusion matrix: {OUTPUT_DIR / 'eval_run'}")


if __name__ == "__main__":
    main()
