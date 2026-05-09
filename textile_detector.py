# textile_detector.py
import cv2
import numpy as np

class TextileDetector:
    def __init__(self, config=None):
        # Default color ranges for common fabrics (HSV)
        self.color_ranges = {
            'white':  {'lower': np.array([0, 0, 180]),    'upper': np.array([180, 50, 255])},
            'gray':   {'lower': np.array([0, 0, 50]),     'upper': np.array([180, 40, 180])},
            'cream':  {'lower': np.array([10, 20, 150]),  'upper': np.array([40, 100, 255])},
            'blue':   {'lower': np.array([90, 30, 50]),   'upper': np.array([130, 150, 200])}
        }
        self.min_roi_area = 20000 # Minimum area to consider it a textile piece

    def is_textile(self, frame):
        """
        Detects if the frame contains fabric and returns the ROI.
        Uses color thresholding and morphological operations.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Apply all color filters
        for name, ranges in self.color_ranges.items():
            mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'is_textile': False, 'confidence': 0.0, 'roi_mask': None, 'roi_bbox': (0,0,0,0)}

        # Find largest contour
        largest_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)

        if area < self.min_roi_area:
            return {'is_textile': False, 'confidence': area/self.min_roi_area, 'roi_mask': None, 'roi_bbox': (0,0,0,0)}

        # Extract ROI
        x, y, w, h = cv2.boundingRect(largest_cnt)
        roi_mask = np.zeros_like(combined_mask)
        cv2.drawContours(roi_mask, [largest_cnt], -1, 255, -1)

        # Texture Score (Variance of Laplacian)
        gray_roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        texture_score = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        
        # Normalize confidence based on area and texture (very basic)
        confidence = min(1.0, area / (frame.shape[0]*frame.shape[1] * 0.5))

        return {
            'is_textile': True,
            'confidence': confidence,
            'roi_mask': roi_mask,
            'roi_bbox': (x, y, w, h),
            'texture_score': texture_score
        }
