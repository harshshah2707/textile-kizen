import cv2
import numpy as np
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
        if 5 < w < W*0.8 and 5 < h < H*0.8:
            valid_contours.append(cnt)
            
    if not valid_contours:
        return None
        
    cnt = max(valid_contours, key=cv2.contourArea)
    return cv2.boundingRect(cnt)

img_paths = [
    "scratch/samples/20240219_163908_610.jpg",
    "scratch/samples/20240203_152328_704.tif"
]

for p in img_paths:
    if os.path.exists(p):
        img = cv2.imread(p)
        bbox = find_defect_bbox(img)
        print(f"File: {p}")
        if bbox:
            x, y, w, h = bbox
            print(f"  Detected defect bbox: x={x}, y={y}, w={w}, h={h}")
        else:
            print("  No defect bounding box detected.")
    else:
        print(f"File not found: {p}")
