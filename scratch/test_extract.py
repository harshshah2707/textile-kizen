import zipfile
import os

zip_path = 'textiledefectdetection.zip'
extract_to = 'test_extract'

if os.path.exists(zip_path):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract only 10 files
        for name in zip_ref.namelist()[:10]:
            zip_ref.extract(name, extract_to)
            print(f"Extracted: {name}")
else:
    print("Zip not found")
