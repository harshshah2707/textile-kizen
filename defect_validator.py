# defect_validator.py
import cv2
import numpy as np

class DefectValidator:
    def __init__(self):
        self.min_contrast = 5 # Extremely sensitive contrast check
        self.min_size = 5     # pixels
        self.max_size = 640   # pixels

    def validate_defect(self, box, frame, roi_mask):
        """
        Validates if a YOLO detection is a real defect.
        box: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = map(int, box)
        h_f, w_f = frame.shape[:2]
        
        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_f, x2), min(h_f, y2)
        
        if x2 <= x1 or y2 <= y1:
            return {'is_valid': False, 'reason': 'invalid_box'}

        # 1. ROI Check (Must be inside textile mask)
        if roi_mask is not None:
            defect_mask_area = roi_mask[y1:y2, x1:x2]
            if np.mean(defect_mask_area) < 128: # At least 50% must be on fabric
                return {'is_valid': False, 'reason': 'not_on_fabric'}
        else:
            # If no fabric mask was provided/found, we skip this check 
            # or could fail it based on strictness. For now, let's just log it.
            pass

        # 2. Size Check
        w, h = x2 - x1, y2 - y1
        if w < self.min_size or h < self.min_size:
            return {'is_valid': False, 'reason': 'too_small'}
        if w > self.max_size or h > self.max_size:
            return {'is_valid': False, 'reason': 'too_large'}

        # 3. Contrast Check
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Expand box slightly to get background pixels
        pad = 10
        bg_x1, bg_y1 = max(0, x1-pad), max(0, y1-pad)
        bg_x2, bg_y2 = min(w_f, x2+pad), min(h_f, y2+pad)
        bg_roi = cv2.cvtColor(frame[bg_y1:bg_y2, bg_x1:bg_x2], cv2.COLOR_BGR2GRAY)
        
        bg_mean = np.mean(bg_roi)
        defect_mean = np.mean(gray_roi)
        
        contrast = abs(bg_mean - defect_mean)
        if contrast < self.min_contrast:
            return {'is_valid': False, 'reason': 'low_contrast'}

        return {
            'is_valid': True,
            'confidence': min(1.0, contrast / 100.0 + 0.5),
            'reason': 'valid_defect'
        }
