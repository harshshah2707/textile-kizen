import cv2
import numpy as np

def refine_defect_bbox(img, bbox, class_id=None):
    """
    Refines a YOLO bounding box to tightly fit the actual defect inside it.
    This is extremely useful when the model outputs a fallback full-image bounding box.
    
    Parameters:
      img: numpy array (BGR image)
      bbox: list/tuple of [x1, y1, x2, y2] in pixel coordinates
      class_id: integer representing the class (optional)
      
    Returns:
      refined_bbox: list of [x1_new, y1_new, x2_new, y2_new]
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
    
    # Clamp coordinates to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    
    w_box = x2 - x1
    h_box = y2 - y1
    
    # If the box is small enough already, no need to refine
    # Threshold: if it covers less than 65% of the total image area
    box_area = w_box * h_box
    img_area = W * H
    if box_area < img_area * 0.65 and w_box < W * 0.8 and h_box < H * 0.8:
        return [x1, y1, x2, y2]
        
    # Crop the box region
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return [x1, y1, x2, y2]
        
    # Process crop to find contours
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    
    # Detect edges using Canny
    edges = cv2.Canny(enhanced, 20, 80)
    
    # Adaptive thresholding to segment dark/light spots
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 3
    )
    
    # Combine edges and threshold
    combined = cv2.bitwise_or(edges, thresh)
    
    # Perform morphological closing to merge nearby components
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        
        # Avoid contours that are exactly the border of the crop or too small (noise)
        if cw < 4 or ch < 4:
            continue
        if cw > w_box * 0.95 or ch > h_box * 0.95:
            continue
        # Also avoid contours that touch the borders extensively
        if cx <= 1 and cx + cw >= w_box - 1:
            continue
        if cy <= 1 and cy + ch >= h_box - 1:
            continue
            
        valid_contours.append(cnt)
        
    if not valid_contours:
        # Fallback 1: If no valid sub-contours, let's try a simpler thresholding (Otsu's thresholding)
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Clean edges
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if 5 < cw < w_box * 0.9 and 5 < ch < h_box * 0.9:
                valid_contours.append(cnt)
                
    if valid_contours:
        # Get bounding box of all valid contours combined
        all_pts = np.vstack([cnt for cnt in valid_contours])
        rx, ry, rw, rh = cv2.boundingRect(all_pts)
        
        # Add 15% padding around the refined box
        pad_w = int(rw * 0.15)
        pad_h = int(rh * 0.15)
        
        rx_new = max(0, rx - pad_w)
        ry_new = max(0, ry - pad_h)
        rw_new = min(w_box - rx_new, rw + 2 * pad_w)
        rh_new = min(h_box - ry_new, rh + 2 * pad_h)
        
        # Convert back to absolute image coordinates
        new_x1 = x1 + rx_new
        new_y1 = y1 + ry_new
        new_x2 = x1 + rx_new + rw_new
        new_y2 = y1 + ry_new + rh_new
        
        # Safety check: make sure the refined box is not tiny and stays within original box
        if (new_x2 - new_x1) >= 10 and (new_y2 - new_y1) >= 10:
            return [new_x1, new_y1, new_x2, new_y2]
            
    # Fallback 2: Shrink slightly to center 40% area of the original bounding box 
    # to avoid showing a giant box that covers the entire frame
    cx = int((x1 + x2) / 2.0)
    cy = int((y1 + y2) / 2.0)
    rw = int(w_box * 0.4)
    rh = int(h_box * 0.4)
    # Ensure reasonable size
    rw = max(120, min(rw, w_box))
    rh = max(120, min(rh, h_box))
    
    new_x1 = max(x1, cx - rw // 2)
    new_y1 = max(y1, cy - rh // 2)
    new_x2 = min(x2, cx + rw // 2)
    new_y2 = min(y2, cy + rh // 2)
    
    return [new_x1, new_y1, new_x2, new_y2]
