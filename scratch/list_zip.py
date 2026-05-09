import zipfile
import os

zip_path = 'textiledefectdetection.zip'
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print("Listing first 50 files in zip:")
        for name in zip_ref.namelist()[:50]:
            print(name)
else:
    print("Zip file not found.")
