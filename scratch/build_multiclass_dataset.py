"""
Build a YOLO detection dataset from the Multi-Class Fabric Defect Detection Dataset.
The dataset has images organized by class in subdirectories.

Since these are classification images (full image = one class label), we'll treat
each defect image as containing a defect in the full image (bbox = whole image)
after auto-localizing defects using CLAHE + edge detection.

Classes (detection):
  0 = Broken stitch
  1 = hole
  2 = horizontal
  3 = lines
  4 = Needle mark
  5 = Pinched fabric
  6 = stain
  7 = Vertical

'defect free' images are used as background (empty label files).
"""

import os
import cv2
import numpy as np
import random
import yaml
from pathlib import Path
from tqdm import tqdm

# Source dataset
SRC_DIR = r"datasets\Multi-Class Fabric Defect Detection Dataset"

# Class mapping (folder name -> class id)
CLASS_MAP = {
    "Broken stitch": 0,
    "hole":          1,
    "horizontal":    2,
    "lines":         3,
    "Needle mark":   4,
    "Pinched fabric":5,
    "stain":         6,
    "Vertical":      7,
}
CLASS_NAMES = [k for k in CLASS_MAP]

# Output
OUT_DIR = "datasets/multiclass_yolo"
PATCH_SIZE = 640
TRAIN_RATIO = 0.8

random.seed(42)


def enhance(img):
    """CLAHE on L channel for better defect visibility."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def find_defect_bbox(img):
    """Locate defect region via edge + adaptive threshold."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 20, 80)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    combined = cv2.bitwise_or(edges, thresh)
    kernel = np.ones((7, 7), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    H, W = img.shape[:2]
    valid = [c for c in contours
             if 10 < cv2.boundingRect(c)[2] < W * 0.9
             and 10 < cv2.boundingRect(c)[3] < H * 0.9]
    if not valid:
        return None
    cnt = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    # Add 10% padding for better coverage
    pad_x = int(w * 0.1)
    pad_y = int(h * 0.1)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(W - x, w + 2 * pad_x)
    h = min(H - y, h + 2 * pad_y)
    return x, y, w, h


def make_yolo_label(bbox, img_w, img_h):
    """Convert pixel bbox to YOLO normalized format."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    wn = w / img_w
    hn = h / img_h
    return cx, cy, wn, hn


def resize_and_pad(img, size=PATCH_SIZE):
    """Letterbox resize to square."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, scale, left, top


def main():
    # Prepare output dirs
    for split in ["train", "val"]:
        Path(f"{OUT_DIR}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{OUT_DIR}/labels/{split}").mkdir(parents=True, exist_ok=True)

    # Clear existing
    for split in ["train", "val"]:
        for ext in ["*.jpg", "*.txt"]:
            for f in Path(f"{OUT_DIR}/images/{split}").glob(ext):
                f.unlink()
            for f in Path(f"{OUT_DIR}/labels/{split}").glob(ext):
                f.unlink()

    all_items = []  # (img_path, class_id_or_None)

    # Collect defect images
    for folder_name, class_id in CLASS_MAP.items():
        folder_path = os.path.join(SRC_DIR, folder_name)
        if not os.path.exists(folder_path):
            print(f"  [WARN] Missing folder: {folder_path}")
            continue
        imgs = [f for f in Path(folder_path).glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        for img_path in imgs:
            all_items.append((str(img_path), class_id))
        print(f"  {folder_name}: {len(imgs)} images (class {class_id})")

    # Collect background (defect free)
    bg_path = os.path.join(SRC_DIR, "defect free")
    bg_imgs = [f for f in Path(bg_path).glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
    # Use only 300 background images to keep balance
    random.shuffle(bg_imgs)
    selected_bg = bg_imgs[:300]
    for img_path in selected_bg:
        all_items.append((str(img_path), None))
    print(f"  defect free: {len(selected_bg)}/{len(bg_imgs)} images (background)")

    print(f"\n[Build] Total items: {len(all_items)}")

    random.shuffle(all_items)
    split_idx = int(len(all_items) * TRAIN_RATIO)
    train_items = all_items[:split_idx]
    val_items = all_items[split_idx:]

    def process_and_save(items, img_dir, lbl_dir, split_name):
        ok, fallback, skipped = 0, 0, 0
        for i, (img_path, class_id) in enumerate(tqdm(items, desc=f"[{split_name}]")):
            raw = cv2.imread(img_path)
            if raw is None:
                skipped += 1
                continue

            img = enhance(raw)
            canvas, scale, left, top = resize_and_pad(img)
            H_orig, W_orig = img.shape[:2]

            stem = f"{split_name}_{i:05d}_{Path(img_path).stem}"
            out_img = os.path.join(img_dir, f"{stem}.jpg")
            out_lbl = os.path.join(lbl_dir, f"{stem}.txt")

            cv2.imwrite(out_img, canvas)

            if class_id is None:
                # Background: empty label file
                open(out_lbl, "w").close()
                ok += 1
                continue

            # Find defect bbox
            bbox = find_defect_bbox(img)
            if bbox is not None:
                x, y, w, h = bbox
                # Transform to letterboxed canvas coordinates
                x_c = x * scale + left
                y_c = y * scale + top
                w_c = w * scale
                h_c = h * scale
                cx = (x_c + w_c / 2) / PATCH_SIZE
                cy = (y_c + h_c / 2) / PATCH_SIZE
                wn = w_c / PATCH_SIZE
                hn = h_c / PATCH_SIZE
                # Clamp
                cx = max(0.01, min(0.99, cx))
                cy = max(0.01, min(0.99, cy))
                wn = max(0.01, min(0.99, wn))
                hn = max(0.01, min(0.99, hn))
                with open(out_lbl, "w") as f:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")
                ok += 1
            else:
                # Fallback: full image bbox (defect is somewhere in the whole image)
                with open(out_lbl, "w") as f:
                    f.write(f"{class_id} 0.500000 0.500000 0.900000 0.900000\n")
                fallback += 1

        print(f"  -> {ok} ok, {fallback} fallback-bbox, {skipped} skipped")

    process_and_save(train_items, f"{OUT_DIR}/images/train", f"{OUT_DIR}/labels/train", "train")
    process_and_save(val_items,   f"{OUT_DIR}/images/val",   f"{OUT_DIR}/labels/val",   "val")

    # Write YAML
    yaml_data = {
        "path": os.path.abspath(OUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_MAP),
        "names": list(CLASS_MAP.keys())
    }
    with open(f"{OUT_DIR}/data.yaml", "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"\n[Build] Done! Dataset saved to {OUT_DIR}")
    print(f"  Train: {len(train_items)}, Val: {len(val_items)}")
    print(f"  Classes ({len(CLASS_MAP)}): {list(CLASS_MAP.keys())}")

if __name__ == "__main__":
    main()
