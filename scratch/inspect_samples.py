import os
import cv2
import numpy as np
from remotezip import RemoteZip

url = "https://www.it.ubi.pt/DetReIDX_dataset/lusitano/test.zip"
os.makedirs("scratch/samples", exist_ok=True)

with RemoteZip(url) as rz:
    namelist = rz.namelist()
    
    # Let's find some files
    jpg_defects = [n for n in namelist if n.startswith('test/defects/') and n.endswith('.jpg')]
    tif_defects = [n for n in namelist if n.startswith('test/defects/') and n.endswith('.tif')]
    jpg_normals = [n for n in namelist if n.startswith('test/non-defects/') and n.endswith('.jpg')]
    tif_normals = [n for n in namelist if n.startswith('test/non-defects/') and n.endswith('.tif')]
    
    samples = []
    if jpg_defects: samples.append(jpg_defects[0])
    if tif_defects: samples.append(tif_defects[0])
    if jpg_normals: samples.append(jpg_normals[0])
    if tif_normals: samples.append(tif_normals[0])
    
    for s in samples:
        filename = os.path.join("scratch/samples", os.path.basename(s))
        print(f"Downloading {s}...")
        with open(filename, 'wb') as f:
            f.write(rz.read(s))
        
        # Load with opencv and print shape and basic statistics
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if img is not None:
            print(f"  Shape: {img.shape}, Type: {img.dtype}, Min: {img.min()}, Max: {img.max()}, Mean: {img.mean():.2f}")
        else:
            print(f"  Failed to load {filename} as image.")
