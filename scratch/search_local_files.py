import os
import glob

search_paths = [
    "C:/Users/Harsh/Downloads",
    "C:/Users/Harsh/Desktop",
    "C:/Users/Harsh/Documents",
    "c:/Users/Harsh/Desktop/textile defect detection"
]

patterns = ["*test*", "*nondefects*", "*lusitano*", "*defect*"]

print("Searching for existing files...")
for base in search_paths:
    if os.path.exists(base):
        print(f"\nIn {base}:")
        for pattern in patterns:
            found = glob.glob(os.path.join(base, pattern))
            for f in found:
                # print file name and size in GB
                sz = os.path.getsize(f) / (1024**3)
                print(f"  - {os.path.basename(f)} ({sz:.2f} GB)")
