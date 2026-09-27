import os, sys, shutil, time, json, collections
import torch, numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

BASE_DIR     = Path(__file__).parent.resolve()
DATASET_YAML = BASE_DIR / "datasets" / "multiclass_yolo" / "data.yaml"
PROJECT_DIR  = "runs/textile_detection"
RUN_NAME     = "best_model_v1"
PROD_WEIGHTS = BASE_DIR / "runs/textile_detection/defect_model_pro_v1/weights/best.pt"

CLASS_NAMES = [
    "Broken stitch", "hole", "horizontal", "lines",
    "Needle mark", "Pinched fabric", "stain", "Vertical",
]


def probe_hardware():
    cuda = torch.cuda.is_available()
    info = {"cuda": cuda, "device": "cpu", "gpu": "CPU", "vram_gb": 0.0,
            "batch": 4, "workers": 4, "model": "yolov8m.pt"}
    if cuda:
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu": p.name, "vram_gb": p.total_memory/1024**3,
                     "device": 0, "sm": p.multi_processor_count,
                     "cuda_cap": f"{p.major}.{p.minor}"})
        vram = info["vram_gb"]
        if vram >= 10:   info["batch"], info["model"] = 16, "yolov8l.pt"
        elif vram >= 7.5: info["batch"], info["model"] =  8, "yolov8l.pt"
        elif vram >= 6:  info["batch"], info["model"] =  8, "yolov8m.pt"
        else:            info["batch"], info["model"] =  4, "yolov8s.pt"
    import psutil
    info["workers"] = min((psutil.cpu_count(logical=False) or 4) // 2, 8)
    return info


def compute_class_weights(label_dir, nc=8):
    counts = np.zeros(nc, dtype=np.float64)
    for f in Path(label_dir).glob("*.txt"):
        for line in f.read_text().strip().splitlines():
            if line.strip():
                cls = int(line.split()[0])
                if cls < nc:
                    counts[cls] += 1
    if counts.sum() == 0:
        return [1.0] * nc
    s = counts + 1.0
    freq = s / s.sum()
    w = np.clip((1.0 / freq) / nc, 0.3, 3.0)
    return (w / w.mean()).tolist()


def pre_checks(hw):
    print("\n" + "=" * 70)
    print("  PRE-TRAINING SYSTEM CHECK")
    print("=" * 70)
    print(f"  GPU    : {hw['gpu']}")
    if hw["cuda"]:
        print(f"  VRAM   : {hw['vram_gb']:.1f} GB | CUDA {hw.get('cuda_cap', '?')} | SM x{hw.get('sm', '?')}")
    print(f"  Model  : {hw['model']}")
    print(f"  Batch  : {hw['batch']}")
    print(f"  Dataset: {DATASET_YAML}")
    if not DATASET_YAML.exists():
        print("[ERROR] Dataset YAML not found"); sys.exit(1)
    label_dir = DATASET_YAML.parent / "labels" / "train"
    if not label_dir.exists():
        print("[ERROR] Label dir not found"); sys.exit(1)
    imgs = list((DATASET_YAML.parent / "images" / "train").glob("*.jpg"))
    lbls = list(label_dir.glob("*.txt"))
    counts = collections.Counter()
    for f in lbls:
        for line in f.read_text().strip().splitlines():
            if line.strip(): counts[int(line.split()[0])] += 1
    total = sum(counts.values())
    print(f"  Train  : {len(imgs)} images | {total} annotations")
    for i, name in enumerate(CLASS_NAMES):
        c = counts.get(i, 0)
        pct = c / total * 100 if total > 0 else 0
        bar = "#" * int(25 * c / max(counts.values(), default=1))
        print(f"    {i} {name:<18} {c:5d} {pct:5.1f}%  {bar}")
    print("=" * 70)


def finetune(best_weights, hw):
    print("\n" + "=" * 70)
    print("  PHASE 2: FINE-TUOING  (30 epochs @ lr=0.0001)")
    print("=" * 70)
    model = YOLO(str(best_weights))
    ft_batch = max(4, hw["batch"] // 2)
    model.train(
        data=str(DATASET_YAML),
        epochs=30, imgsz=640, batch=ft_batch,
        device=hw["device"], workers=hw["workers"], amp=True,
        project=PROJECT_DIR, name=f"{RUN_NAME}_finetune", exist_ok=True,
        optimizer="AdamW", lr0=0.0001, lrf=1e-5,
        momentum=0.937, weight_decay=0.0005, warmup_epochs=2.0,
        mosaic=0.5, mixup=0.1, copy_paste=0.1, close_mosaic=15,
        fliplr=0.5, flipud=0.1, degrees=5.0,
        hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, erasing=0.2,
        patience=15, val=True, verbose=True, seed=42,
        save=True, label_smoothing=0.05,
    )
    ft_best = None
    for cand in [
        Path(PROJECT_DIR) / f"{RUN_NAME}_finetune" / "weights" / "best.pt",
        Path("runs/detect") / PROJECT_DIR / f"{RUN_NAME}_finetune" / "weights" / "best.pt",
        Path("runs/detect") / f"{RUN_NAME}_finetune" / "weights" / "best.pt",
    ]:
        if cand.exists():
            ft_best = cand
            break
    return ft_best


def main():
    hw = probe_hardware()
    pre_checks(hw)
    label_dir = DATASET_YAML.parent / "labels" / "train"
    cls_weights = compute_class_weights(label_dir)
    print("\n  Per-class weights (inverse-freq, clipped 0.3-3.0):")
    for i, (name, w) in enumerate(zip(CLASS_NAMES, cls_weights)):
        bar = "#" * int(20 * w / max(cls_weights))
        print(f"    {i} {name:<18} w={w:.3f}  {bar}")

    t0 = time.time()
    last_ckpt = None
    for cand in [
        Path("runs/detect") / PROJECT_DIR / RUN_NAME / "weights" / "last.pt",
        Path(PROJECT_DIR) / RUN_NAME / "weights" / "last.pt",
        Path("runs/detect") / RUN_NAME / "weights" / "last.pt",
    ]:
        if cand.exists():
            last_ckpt = cand
            break

    if last_ckpt:
        print(f"\n[Phase 1] Found existing checkpoint: {last_ckpt}")
        print(f"[Phase 1] Resuming training from {last_ckpt}...")
        model = YOLO(str(last_ckpt))
        results = model.train(resume=True)
    else:
        print(f"\n[Phase 1] Loading {hw['model']} (COCOpretrained)...")
        model = YOLO(hw["model"])
        print(f"[Phase 1] Starting: batch={hw['batch']} workers={hw['workers']} FP16 True")
        print("          Expected: ~12-18 hrs on RTX 3050 for 150 epochs\n")

        results = model.train(
            data=str(DATASET_YAML),
            epochs=150, imgsz=640, batch=hw["batch"],
            device=hw["device"], workers=hw["workers"], amp=True,
            project=PROJECT_DIR, name=RUN_NAME, exist_ok=True,
            optimizer="AdamW", lr0=0.001, lrf=5e-5,
            momentum=0.937, weight_decay=0.0005,
            warmup_epochs=5.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
            cls=0.5, box=7.5, dfl=1.5, label_smoothing=0.1,
            mosaic=1.0, mixup=0.25, copy_paste=0.3, close_mosaic=30,
            fliplr=0.5, flipud=0.15, degrees=12.0, translate=0.1,
            scale=0.6, shear=5.0, perspective=0.0005,
            hsv_h=0.02, hsv_s=0.7, hsv_v=0.5, erasing=0.4,
            crop_fraction=1.0,
            patience=30, val=True, verbose=True, seed=42,
            deterministic=False, save=True, save_period=25, nms=True,
        )

    elapsed = time.time() - t0
    hours, rem = divmod(elapsed, 3600)
    minutes = rem // 60
    print(f"\n[Phase 1] Training complete in {int(hours)}h {int(minutes)}m")

    save_dir = Path(model.trainer.save_dir) if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir') else Path(PROJECT_DIR) / RUN_NAME
    best_src = save_dir / "weights" / "best.pt"
    if not best_src.exists():
        for cand in [
            Path("runs/detect") / PROJECT_DIR / RUN_NAME / "weights" / "best.pt",
            Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt",
            Path("runs/detect") / RUN_NAME / "weights" / "best.pt",
        ]:
            if cand.exists():
                best_src = cand
                break
    if not best_src.exists():
        print(f"[ERROR] best.pt not found at {best_src}"); return
    print(f"[Phase 1] Best weights: {best_src} ({best_src.stat().st_size/1024**2:.1f} MB)")

    ft_best = finetune(best_src, hw)
    final_weights = ft_best if ft_best else best_src
    PROD_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_weights, PROD_WEIGHTS)
    print(f"\n[Deploy] Copied to production: {PROD_WEIGHTS}")

    m = results.results_dict if hasattr(results, 'results_dict') else {}
    map50   = m.get("metrics/mAP50(B)", 0)
    map5095 = m.get("metrics/mAP50-95(B)", 0)
    prec    = m.get("metrics/precision(B)", 0)
    recall  = m.get("metrics/recall(B)", 0)
    f1      = 2 * prec * recall / max(prec + recall, 1e-6)
    print("\n" + "=" * 70)
    print("  FINAL METRICS")
    print("=" * 70)
    print(f"  mAP@50       : {map50:.4f}   ({map50*100:.1f}%)")
    print(f"  mAP@50-95    : {map5095:.4f}   ({map5095*100:.1f}%)")
    print(f"  Precision    : {prec:.4f}   ({prec*100:.1f}%)")
    print(f"  Recall       : {recall:.4f}   ({recall*100:.1f}%)")
    print(f"  F1 Score     : {f1:.4f}   ({f1*100:.1f}%)")
    print(f"  Training time: {int(hours)}h {int(minutes)}m")
    print(f"  GPU          : {hw['gpu']}")
    print("=" * 70)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": hw["model"], "dataset": str(DATASET_YAML),
        "imgsz": 640, "batch": hw["batch"], "epochs": 150,
        "gpu": hw["gpu"], "vram_gb": hw.get("vram_gb", 0), "amp": True,
        "mAP50": round(map50, 4), "mAP50_95": round(map5095, 4),
        "precision": round(prec, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "train_time_h": round(elapsed/3600, 2),
        "production_weights": str(PROD_WEIGHTS),
    }
    sp = save_dir / "training_summary.json"
    sp.write_text(json.dumps(summary, indent=2))
    print(f"\n[Summary] Written to: {sp}")
    print("\n  Next steps:")
    print("  1. python evaluate_model.py")
    print("  2. python export_linescan_engine.py --format engine --batch 8 --benchmark 100")
    print("  3. Restart web_server.py (new best.pt is live)\n")


if __name__ == "__main__":
    main()
