import cv2
import numpy as np
import glob
import os

def find_defect_bbox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge Detection
    edges = cv2.Canny(blurred, 30, 100)
    
    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    combined = cv2.bitwise_or(edges, thresh)
    kernel = np.ones((5,5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    H, W = img.shape[:2]
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 8 < w < W * 0.8 and 8 < h < H * 0.8:
            valid_contours.append(cnt)
            
    if not valid_contours:
        return None
        
    cnt = max(valid_contours, key=cv2.contourArea)
    return cv2.boundingRect(cnt)

raw_defects = glob.glob("scratch/lusitano_raw/defects/*.jpg")
total = len(raw_defects)
success = 0

print(f"Analyzing {total} raw defect images...")
for p in raw_defects:
    img = cv2.imread(p)
    if img is not None:
        bbox = find_defect_bbox(img)
        if bbox is not None:
            success += 1
            
print(f"Success count: {success} / {total} ({success/total*100:.1f}%)")
