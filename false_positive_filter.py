# false_positive_filter.py
import cv2
import numpy as np

class FalsePositiveFilter:
    def __init__(self):
        self.edge_threshold = 100 # Canny threshold

    def filter_detections(self, box, frame):
        """
        Filters out geometric shapes (like cabinet doors) and edge-case noise.
        """
        x1, y1, x2, y2 = map(int, box)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return {'is_valid': False}

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharp Edge Check (Geometric objects have very sharp, straight edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # If edge density is too high or perfectly straight lines are found
        if edge_density > 0.5:
            return {'is_valid': False, 'reason': 'too_sharp_geometric'}

        # 2. Straight Line Detection (Hough Lines)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
        if lines is not None and len(lines) > 4:
            return {'is_valid': False, 'reason': 'geometric_pattern'}

        return {'is_valid': True}
