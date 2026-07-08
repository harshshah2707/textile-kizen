import os
import glob

dirs = [
    "datasets/pro_yolo/images/train",
    "datasets/pro_yolo/images/val",
    "datasets/pro_yolo/labels/train",
    "datasets/pro_yolo/labels/val"
]

for d in dirs:
    if os.path.exists(d):
        files = os.listdir(d)
        print(f"{d}: {len(files)} files")
    else:
        print(f"{d} does not exist.")
