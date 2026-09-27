"""
Textile Defect Detection System - Configuration
================================================
Central configuration file for all system parameters.
Optimized for NVIDIA RTX 3050 (8GB VRAM) — Maximum Accuracy Mode.
"""

import os
from pathlib import Path

# ============================================================
# Project Paths
# ============================================================
BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "datasets" / "full_yolo"         # Merged full dataset
FALLBACK_DATASET = BASE_DIR / "datasets" / "multiclass_yolo"  # Fallback if full not prepared
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
RUNS_DIR = BASE_DIR / "runs" / "textile_detection"
MODEL_DIR = RUNS_DIR / "defect_model_pro_v1"            # Production deployment path
WEIGHTS_DIR = MODEL_DIR / "weights"
MAX_ACC_MODEL_DIR = RUNS_DIR / "max_accuracy_v1"         # New training run
OUTPUT_DIR = BASE_DIR / "outputs"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"

# Ensure directories exist
for d in [IMAGES_DIR, LABELS_DIR, RUNS_DIR, OUTPUT_DIR,
          TEMPLATES_DIR, STATIC_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Defect Classes
# ============================================================
DEFECT_CLASSES = {
    0: "Broken stitch",
    1: "hole",
    2: "horizontal",
    3: "lines",
    4: "Needle mark",
    5: "Pinched fabric",
    6: "stain",
    7: "Vertical",
}
NUM_CLASSES = len(DEFECT_CLASSES)
CLASS_NAMES = list(DEFECT_CLASSES.values())

# ============================================================
# Dataset Generation Config
# ============================================================
DATASET_CONFIG = {
    "num_images": 200,
    "image_size": 640,
    "train_split": 0.8,       # 80% train, 20% val
    "max_defects_per_image": 4,
    "min_defects_per_image": 1,
    "normal_ratio": 0.10,     # 10% images with no defects
    "seed": 42,
}

# ============================================================
# Model Config
# ============================================================
MODEL_CONFIG = {
    "size": "x",                      # YOLOv8 extra-large — max accuracy
    "model_name": "yolov8x.pt",       # COCO-pretrained extra-large backbone
    "pretrained": True,
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
}

# ============================================================
# Training Config (Optimized for RTX 3050 8GB VRAM — Max Accuracy)
# ============================================================
TRAINING_CONFIG = {
    "epochs": 200,
    "batch_size": 4,           # YOLOv8x at 1280px — batch=4 fits in 8GB VRAM (~6GB used)
    "imgsz": 1280,             # High-res for fine textile defect detection
    "device": 0,               # GPU index (0 = RTX 3050)
    "optimizer": "AdamW",      # Better convergence on class-imbalanced data
    "lr0": 0.001,              # AdamW initial LR
    "lrf": 0.0001,             # Final LR (cosine decay)
    "momentum": 0.937,         # Beta1 for AdamW
    "weight_decay": 0.0005,
    "warmup_epochs": 5.0,      # Longer warmup for AdamW
    "warmup_momentum": 0.8,
    "patience": 40,            # Long patience for 8-class training
    "save_period": 20,
    "project": str(RUNS_DIR),
    "name": "max_accuracy_v1",
    "exist_ok": True,
    "verbose": True,
    "seed": 42,
    "amp": True,               # FP16 — RTX 3050 supports Tensor Cores
    "workers": 8,              # i7-12700KF: 16 threads available
    "label_smoothing": 0.1,   # Prevents overconfidence on minority classes
    # Data augmentation
    "augment": True,
    "mosaic": 1.0,
    "mixup": 0.3,
    "copy_paste": 0.3,
    "close_mosaic": 30,
    "flipud": 0.2,
    "fliplr": 0.5,
    "degrees": 15.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 5.0,
    "perspective": 0.0005,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "erasing": 0.4,
}

# ============================================================
# Inference Config
# ============================================================
INFERENCE_CONFIG = {
    "confidence_threshold": 0.30,  # Lower threshold catches borderline defects
    "iou_threshold": 0.45,
    "max_det": 100,                # More detections for dense fabric scans
    "imgsz": 1280,                 # Match training resolution
    "device": 0,
    "half": True,                  # FP16 — RTX 3050 supports it, 2x inference speed
    "agnostic_nms": False,
}

# ============================================================
# Quality Control Thresholds
# ============================================================
QC_THRESHOLDS = {
    "pass": {"max_defects": 0, "label": "PASS", "color": (0, 200, 0)},
    "warning": {"max_defects": 2, "label": "WARNING", "color": (0, 200, 255)},
    "fail": {"max_defects": 999, "label": "FAIL", "color": (0, 0, 255)},
}

# ============================================================
# Visualization Config
# ============================================================
VIS_CONFIG = {
    "box_color": (0, 0, 255),          # Red (BGR)
    "box_thickness": 2,
    "font_scale": 0.6,
    "font_thickness": 2,
    "label_bg_color": (0, 0, 200),
    "label_text_color": (255, 255, 255),
    "status_pass_color": (0, 200, 0),
    "status_fail_color": (0, 0, 255),
}

# ============================================================
# Flask / Web UI Config
# ============================================================
FLASK_CONFIG = {
    "host": "0.0.0.0",
    "port": 5001,
    "debug": True,
    "max_content_length": 16 * 1024 * 1024,  # 16MB max upload
    "allowed_extensions": {"png", "jpg", "jpeg", "bmp", "tiff"},
}

# ============================================================
# Webcam / Video Config
# ============================================================
WEBCAM_CONFIG = {
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "target_fps": 20,
}

# ============================================================
# Export Config (Jetson Deployment)
# ============================================================
EXPORT_CONFIG = {
    "format": "engine",        # TensorRT for Jetson
    "half": True,              # FP16 on Jetson
    "imgsz": 640,
    "simplify": True,
}

# ============================================================
# Logging
# ============================================================
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}

# Best model path shortcut
BEST_MODEL_PATH = WEIGHTS_DIR / "best.pt"
LINE_SCAN_CONFIG = {
    "tile_size": 1280,         # Tile size for line scan tiled inference
    "overlap": 0.15,           # 15% tile overlap — optimised for 1280px tiles (was 25%)
    "conf": 0.30,
    "iou": 0.45,
    "pixel_per_mm": 10.0,      # Camera calibration: update for your setup
    "stitch_mode": "horizontal",  # "horizontal" | "vertical"
}

# ============================================================
# Line-Scan Performance Config (RTX 3050 · i7-12700K · 32GB)
# Auto-tuned at startup — edit only if you change hardware.
# ============================================================
import torch as _torch

def _detect_gpu_vram_gb() -> float:
    try:
        if _torch.cuda.is_available():
            return _torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0

_VRAM_GB = _detect_gpu_vram_gb()

# Batch size: 1 tile ≈ 1280×1280×3 FP16 ≈ ~10 MB on GPU.
# RTX 3050 8 GB → 8 tiles comfortably; leave 2 GB headroom for activations.
_BATCH_FROM_VRAM = max(1, min(int((_VRAM_GB - 2.0) / 1.2), 16)) if _VRAM_GB >= 4 else 1

LINESCAN_PERF_CONFIG = {
    # ── Inference ────────────────────────────────────────────
    "tile_size":          1280,            # px — full resolution tile
    "overlap":            0.15,            # 15% overlap (saves ~30% tiles vs 25%)
    "batch_size":         _BATCH_FROM_VRAM, # tiles per GPU forward pass (8 for RTX 3050)
    "imgsz_live":         640,             # live stream inference size (RTX 3050 handles 640 @ 45+ FPS)
    "use_half":           True,            # FP16 Tensor Cores on RTX 3050 (2× throughput)
    "conf":               0.30,
    "iou":                0.45,
    "nms_backend":        "torch",         # "torch" = torchvision GPU NMS; "cv2" = CPU fallback
    "max_det":            300,

    # ── Threading / CPU ──────────────────────────────────────
    "inference_workers":  8,               # tile-prep threads (8 of 20 logical cores — P-cores)
    "camera_workers":     4,               # camera I/O threads
    "frame_queue_depth":  4,               # producer→consumer queue depth

    # ── Camera / Canvas ──────────────────────────────────────
    "mindvision_slice_height":   128,      # 128 px slices for silky-smooth waterfall scrolling
    "canvas_multiplier":          24,      # 24 × 128 = 3072 px (clean 4:3 square pixel aspect ratio)
    "use_gpu_clahe":              False,   # True if OpenCV-CUDA is installed
    "use_pinned_memory":          True,    # pin ring-buffer for zero-copy DMA to GPU

    # ── MJPEG Stream ─────────────────────────────────────────
    "mjpeg_quality":      88,              # JPEG quality — 88 gives 40% smaller frames vs 95

    # ── Hardware summary (read-only, set at startup) ──────────
    "_gpu_name":          _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else "CPU",
    "_vram_gb":           round(_VRAM_GB, 1),
    "_cuda_available":    _torch.cuda.is_available(),
}

del _torch  # avoid polluting module namespace



def print_config():
    """Print current configuration summary."""
    print("=" * 60)
    print("  TEXTILE DEFECT DETECTION - CONFIGURATION")
    print("=" * 60)
    print(f"  Base Directory    : {BASE_DIR}")
    print(f"  Dataset Directory : {DATASET_DIR}")
    print(f"  Model Directory   : {MODEL_DIR}")
    print(f"  Output Directory  : {OUTPUT_DIR}")
    print(f"  Defect Classes    : {CLASS_NAMES}")
    print(f"  Num Classes       : {NUM_CLASSES}")
    print(f"  Model Size        : YOLOv8{MODEL_CONFIG['size']}")
    print(f"  Training Epochs   : {TRAINING_CONFIG['epochs']}")
    print(f"  Batch Size        : {TRAINING_CONFIG['batch_size']}")
    print(f"  Image Size        : {TRAINING_CONFIG['imgsz']}px")
    print(f"  Optimizer         : {TRAINING_CONFIG['optimizer']}")
    print(f"  AMP (FP16)        : {TRAINING_CONFIG['amp']}")
    print(f"  Device            : GPU {TRAINING_CONFIG['device']} (RTX 3050)")
    print(f"  Confidence Thresh : {INFERENCE_CONFIG['confidence_threshold']}")
    print(f"  Line Scan Tile    : {LINE_SCAN_CONFIG['tile_size']}px  overlap={LINE_SCAN_CONFIG['overlap']*100:.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
