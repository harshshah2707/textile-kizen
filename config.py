"""
Textile Defect Detection System - Configuration
================================================
Central configuration file for all system parameters.
Optimized for NVIDIA GTX 1650 (4GB VRAM).
"""

import os
from pathlib import Path

# ============================================================
# Project Paths
# ============================================================
BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "datasets" / "multiclass_yolo"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
RUNS_DIR = BASE_DIR / "runs" / "textile_detection"
MODEL_DIR = RUNS_DIR / "defect_model_pro_v1"
WEIGHTS_DIR = MODEL_DIR / "weights"
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
    "size": "s",                      # YOLOv8 small
    "model_name": "yolov8s.pt",       # Pretrained weights
    "pretrained": True,
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
}

# ============================================================
# Training Config (Optimized for GTX 1650 4GB VRAM)
# ============================================================
TRAINING_CONFIG = {
    "epochs": 50,
    "batch_size": 16,
    "imgsz": 640,
    "device": 0,               # GPU index (0 = first GPU)
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "patience": 10,            # Early stopping patience
    "save_period": 10,
    "project": str(RUNS_DIR),
    "name": "defect_model_v1",
    "exist_ok": True,
    "verbose": True,
    "seed": 42,
    # Data augmentation
    "augment": True,
    "mosaic": 1.0,
    "mixup": 0.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "degrees": 15.0,
    "translate": 0.1,
    "scale": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
}

# ============================================================
# Inference Config
# ============================================================
INFERENCE_CONFIG = {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "max_det": 50,
    "imgsz": 640,
    "device": 0,
    "half": False,             # FP16 (set True for speed if supported)
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
    print(f"  Image Size        : {TRAINING_CONFIG['imgsz']}")
    print(f"  Device            : GPU {TRAINING_CONFIG['device']}")
    print(f"  Confidence Thresh : {INFERENCE_CONFIG['confidence_threshold']}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
