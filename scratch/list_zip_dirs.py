import zipfile
import os

zip_path = 'textiledefectdetection.zip'
if os.path.exists(zip_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            dirs = set()
            for name in zip_ref.namelist():
                parts = name.split('/')
                if len(parts) > 1:
                    dirs.add(parts[0])
            print("Top-level directories in zip:")
            for d in sorted(dirs):
                print(d)
    except Exception as e:
        print(f"Error reading zip: {e}")
else:
    print("Zip file not found.")
