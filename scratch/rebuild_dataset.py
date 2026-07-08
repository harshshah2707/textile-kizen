"""
Rebuild the Lusitano YOLO dataset from scratch.
- Applies CLAHE contrast enhancement (better than min-max for fabric texture)
- Labels ALL defects as class 0 (single-class "defect" detection)
- Creates multiple augmented crops per defect image for more training data
- Writes nc=1 YAML for proper model head alignment
"""
import os
import cv2
import numpy as np
import random
import glob
import yaml
from pathlib import Path

RAW_DIR = "scratch/lusitano_raw"
RAW_DEFECTS = os.path.join(RAW_DIR, "defects")
RAW_NORMALS = os.path.join(RAW_DIR, "non-defects")

OUT_DIR = "datasets/lusitano_yolo"
train_img_dir = os.path.join(OUT_DIR, "images/train")
train_lbl_dir = os.path.join(OUT_DIR, "labels/train")
val_img_dir   = os.path.join(OUT_DIR, "images/val")
val_lbl_dir   = os.path.join(OUT_DIR, "labels/val")

PATCH_SIZE = 640
AUGMENTS_PER_DEFECT = 4  # number of random crops per defect image

def enhance_image(img):
    """Apply CLAHE to each channel for robust contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def find_defect_bbox(img):
    """Find defect bounding box using Canny + adaptive threshold combination."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 80)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    combined = cv2.bitwise_or(edges, thresh)
    kernel = np.ones((7, 7), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    H, W = img.shape[:2]
    valid = [c for c in contours if 5 < cv2.boundingRect(c)[2] < W * 0.85 and
                                     5 < cv2.boundingRect(c)[3] < H * 0.85]
    if not valid:
        return None
    cnt = max(valid, key=cv2.contourArea)
    return cv2.boundingRect(cnt)

def crop_around_point(img, cx, cy, patch_size, shift_x=0, shift_y=0):
    """Crop a patch_size square around (cx, cy) with random shift, clamped to image."""
    H, W = img.shape[:2]
    x1 = int(np.clip(cx - patch_size // 2 + shift_x, 0, max(0, W - patch_size)))
    y1 = int(np.clip(cy - patch_size // 2 + shift_y, 0, max(0, H - patch_size)))
    return img[y1:y1+patch_size, x1:x1+patch_size], x1, y1

def bbox_in_crop(orig_bbox, crop_x1, crop_y1, patch_size, min_px=8):
    """Convert original bbox to patch-relative YOLO coordinates."""
    x, y, w, h = orig_bbox
    nx1 = max(0, x - crop_x1)
    ny1 = max(0, y - crop_y1)
    nx2 = min(patch_size, x + w - crop_x1)
    ny2 = min(patch_size, y + h - crop_y1)
    nw = nx2 - nx1
    nh = ny2 - ny1
    if nw < min_px or nh < min_px:
        return None
    cx_n = (nx1 + nw / 2) / patch_size
    cy_n = (ny1 + nh / 2) / patch_size
    w_n  = nw / patch_size
    h_n  = nh / patch_size
    return cx_n, cy_n, w_n, h_n

def main():
    # Clear and recreate dirs
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
        for f in Path(d).glob("*"):
            if f.is_file():
                f.unlink()

    random.seed(42)

    defect_images = sorted(glob.glob(os.path.join(RAW_DEFECTS, "*.jpg")))
    normal_images = sorted(glob.glob(os.path.join(RAW_NORMALS, "*.jpg")))

    print(f"[Rebuild] Found {len(defect_images)} defects, {len(normal_images)} normals")

    all_patches = []  # list of (crop_img, label_str_or_None, name)

    print("[Rebuild] Processing defect images...")
    for idx, path in enumerate(defect_images):
        raw = cv2.imread(path)
        if raw is None:
            print(f"  Skipped unreadable: {path}")
            continue

        img = enhance_image(raw)
        bbox = find_defect_bbox(img)
        H, W = img.shape[:2]

        if bbox is None:
            # Fallback: center crop, no label
            crop, _, _ = crop_around_point(img, W//2, H//2, PATCH_SIZE)
            all_patches.append((crop, None, f"defect_{idx}_nobox"))
            continue

        x, y, w, h = bbox
        cx = x + w // 2
        cy = y + h // 2

        # Generate AUGMENTS_PER_DEFECT crops with increasing shift variance
        shifts = [(0, 0)] + [(random.randint(-120, 120), random.randint(-60, 60))
                              for _ in range(AUGMENTS_PER_DEFECT - 1)]

        for aug_i, (sx, sy) in enumerate(shifts):
            crop, cx1, cy1 = crop_around_point(img, cx, cy, PATCH_SIZE, sx, sy)
            if crop.shape[0] < PATCH_SIZE or crop.shape[1] < PATCH_SIZE:
                continue

            coords = bbox_in_crop(bbox, cx1, cy1, PATCH_SIZE)
            if coords is not None:
                cx_n, cy_n, w_n, h_n = coords
                label = f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n"
            else:
                label = None

            all_patches.append((crop, label, f"defect_{idx}_aug{aug_i}"))

    print("[Rebuild] Processing normal images...")
    for idx, path in enumerate(normal_images):
        raw = cv2.imread(path)
        if raw is None:
            continue
        img = enhance_image(raw)
        H, W = img.shape[:2]
        px1 = random.randint(0, max(0, W - PATCH_SIZE))
        py1 = random.randint(0, max(0, H - PATCH_SIZE))
        crop = img[py1:py1+PATCH_SIZE, px1:px1+PATCH_SIZE]
        if crop.shape[0] < PATCH_SIZE or crop.shape[1] < PATCH_SIZE:
            crop = cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE))
        all_patches.append((crop, None, f"normal_{idx}"))

    print(f"[Rebuild] Total patches: {len(all_patches)}")

    random.shuffle(all_patches)
    split = int(len(all_patches) * 0.8)
    train_patches = all_patches[:split]
    val_patches   = all_patches[split:]

    defect_train = sum(1 for _, lbl, _ in train_patches if lbl)
    defect_val   = sum(1 for _, lbl, _ in val_patches if lbl)
    print(f"[Rebuild] Train: {len(train_patches)} total ({defect_train} with defects)")
    print(f"[Rebuild] Val:   {len(val_patches)} total ({defect_val} with defects)")

    def save(patches, img_d, lbl_d):
        for crop, lbl, name in patches:
            cv2.imwrite(os.path.join(img_d, f"{name}.jpg"), crop)
            lbl_path = os.path.join(lbl_d, f"{name}.txt")
            with open(lbl_path, "w") as f:
                if lbl:
                    f.write(lbl)

    save(train_patches, train_img_dir, train_lbl_dir)
    save(val_patches, val_img_dir, val_lbl_dir)

    # Write nc=1 YAML
    yaml_data = {
        "path": os.path.abspath(OUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["defect"]
    }
    yaml_path = os.path.join(OUT_DIR, "lusitano_data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"[Rebuild] Done! Dataset written to {OUT_DIR}")
    print(f"[Rebuild] YAML updated: nc=1, class='defect'")

if __name__ == "__main__":
    main()
