import os
import glob

train_lbls = glob.glob("datasets/lusitano_yolo/labels/train/*.txt")
print("Total train labels:", len(train_lbls))

empty_count = 0
non_empty_count = 0

for p in train_lbls:
    if os.path.getsize(p) == 0:
        empty_count += 1
    else:
        non_empty_count += 1

print(f"Empty labels (background): {empty_count}")
print(f"Non-empty labels (with defects): {non_empty_count}")

# Print first 5 non-empty labels
print("\nFirst 5 non-empty label files:")
printed = 0
for p in train_lbls:
    if os.path.getsize(p) > 0:
        with open(p, 'r') as f:
            print(f"  {os.path.basename(p)}: {f.read().strip()}")
        printed += 1
        if printed >= 5:
            break
