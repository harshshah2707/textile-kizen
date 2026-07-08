import os
import glob

val_lbls = glob.glob("datasets/lusitano_yolo/labels/val/*.txt")
print("Total val labels:", len(val_lbls))

empty_count = 0
non_empty_count = 0

for p in val_lbls:
    if os.path.getsize(p) == 0:
        empty_count += 1
    else:
        non_empty_count += 1

print(f"Empty labels (background): {empty_count}")
print(f"Non-empty labels (with defects): {non_empty_count}")

# Print first 5 non-empty validation labels
print("\nFirst 5 non-empty val label files:")
printed = 0
for p in val_lbls:
    if os.path.getsize(p) > 0:
        with open(p, 'r') as f:
            print(f"  {os.path.basename(p)}: {f.read().strip()}")
        printed += 1
        if printed >= 5:
            break
