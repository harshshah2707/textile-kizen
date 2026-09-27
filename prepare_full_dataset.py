"""
prepare_full_dataset.py
=======================
Merges the existing YOLO-formatted dataset (multiclass_yolo) with the raw
Kaggle classification dataset (Multi-Class Fabric Defect Detection Dataset).

The raw Kaggle data is a per-class folder layout → each image IS the defect,
so we convert it by creating a full-image bounding box label (YOLO format).

Outputs:
  datasets/full_yolo/
    images/train/
    images/val/
    labels/train/
    labels/val/
    data.yaml

Class mapping (8 defect classes — "defect free" is background, label-only file):
  0: Broken stitch
  1: hole
  2: horizontal
  3: lines
  4: Needle mark
  5: Pinched fabric
  6: stain
  7: Vertical
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from PIL import Image

# ─── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.resolve()
KAGGLE_DIR   = BASE_DIR / "datasets" / "Multi-Class Fabric Defect Detection Dataset"
YOLO_DIR     = BASE_DIR / "datasets" / "multiclass_yolo"
OUT_DIR      = BASE_DIR / "datasets" / "full_yolo"
VAL_RATIO    = 0.20
SEED         = 42

# Mapping from Kaggle folder name → YOLO class index
CLASS_MAP = {
    "Broken stitch":  0,
    "hole":           1,
    "horizontal":     2,
    "lines":          3,
    "Needle mark":    4,
    "Pinched fabric": 5,
    "stain":          6,
    "Vertical":       7,
    # "defect free" → no label file needed (background)
}

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

random.seed(SEED)

# ─── Output Directory Setup ─────────────────────────────────────────────────
for split in ["train", "val"]:
    (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

counters = {"train": 0, "val": 0}
class_counts = {i: 0 for i in range(8)}
background_count = 0

# ─── Helper ─────────────────────────────────────────────────────────────────
def assign_split(idx, val_ratio=VAL_RATIO):
    return "val" if random.random() < val_ratio else "train"

def copy_image_label(src_img: Path, label_content: str, dest_name: str, split: str):
    dst_img = OUT_DIR / "images" / split / dest_name
    dst_lbl = OUT_DIR / "labels" / split / (Path(dest_name).stem + ".txt")
    shutil.copy2(src_img, dst_img)
    dst_lbl.write_text(label_content)
    counters[split] += 1

# ─── Step 1: Copy existing multiclass_yolo data ─────────────────────────────
print("[Prep] Copying existing YOLO-formatted dataset...")
for split in ["train", "val"]:
    img_dir = YOLO_DIR / "images" / split
    lbl_dir = YOLO_DIR / "labels" / split
    if not img_dir.exists():
        print(f"  [WARN] {img_dir} not found, skipping")
        continue
    imgs = [f for f in img_dir.iterdir() if f.suffix.lower() in VALID_EXT]
    for img_path in imgs:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        dest_name = f"yolo_{split}_{img_path.name}"
        dst_img   = OUT_DIR / "images" / split / dest_name
        dst_lbl   = OUT_DIR / "labels" / split / (Path(dest_name).stem + ".txt")
        shutil.copy2(img_path, dst_img)
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl)
            # count classes
            for line in lbl_path.read_text().strip().splitlines():
                if line.strip():
                    cls_id = int(line.split()[0])
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
        else:
            dst_lbl.write_text("")  # empty = background
        counters[split] += 1

print(f"  [OK] Copied YOLO data: {counters['train']} train, {counters['val']} val")

# ─── Step 2: Convert raw Kaggle classification data ──────────────────────────
print("[Prep] Converting raw Kaggle classification dataset...")
all_kaggle = []
for folder in KAGGLE_DIR.iterdir():
    if not folder.is_dir():
        continue
    class_name = folder.name
    imgs = [f for f in folder.iterdir() if f.suffix.lower() in VALID_EXT]
    for img_path in imgs:
        all_kaggle.append((img_path, class_name))

random.shuffle(all_kaggle)

for idx, (img_path, class_name) in enumerate(all_kaggle):
    split = assign_split(idx)
    dest_name = f"kaggle_{idx:05d}_{img_path.name}"

    if class_name == "defect free":
        # Background image — copy with empty label
        dst_img = OUT_DIR / "images" / split / dest_name
        dst_lbl = OUT_DIR / "labels" / split / (Path(dest_name).stem + ".txt")
        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text("")  # empty = no defect
        counters[split] += 1
        background_count += 1
        continue

    if class_name not in CLASS_MAP:
        print(f"  [WARN] Unknown class: {class_name}, skipping {img_path.name}")
        continue

    cls_id = CLASS_MAP[class_name]

    # Build full-image bbox label (YOLO: cx cy w h all normalized to 1.0)
    # We use 0.95 rather than 1.0 to avoid edge clipping artifacts
    label_line = f"{cls_id} 0.500000 0.500000 0.950000 0.950000"

    dst_img = OUT_DIR / "images" / split / dest_name
    dst_lbl = OUT_DIR / "labels" / split / (Path(dest_name).stem + ".txt")
    shutil.copy2(img_path, dst_img)
    dst_lbl.write_text(label_line)
    counters[split] += 1
    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

print(f"  [OK] Converted Kaggle data")

# ─── Step 3: Write data.yaml ─────────────────────────────────────────────────
data_yaml = {
    "path":  str(OUT_DIR),
    "train": "images/train",
    "val":   "images/val",
    "nc":    8,
    "names": [
        "Broken stitch",
        "hole",
        "horizontal",
        "lines",
        "Needle mark",
        "Pinched fabric",
        "stain",
        "Vertical",
    ],
}
yaml_path = OUT_DIR / "data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

# ─── Step 4: Summary ────────────────────────────────────────────────────────
total = counters["train"] + counters["val"]
print("\n" + "=" * 60)
print("  DATASET PREPARATION COMPLETE")
print("=" * 60)
print(f"  Total images  : {total}")
print(f"  Train         : {counters['train']}")
print(f"  Val           : {counters['val']}")
print(f"  Background    : {background_count} (defect-free images)")
print(f"\n  Per-Class Annotation Counts:")
class_names = data_yaml["names"]
for i, name in enumerate(class_names):
    print(f"    [{i}] {name:18s} : {class_counts.get(i, 0):4d}")
print(f"\n  data.yaml     : {yaml_path}")
print("=" * 60)

if __name__ == "__main__":
    pass
