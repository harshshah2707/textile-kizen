# defect_tracker.py
import numpy as np
from collections import OrderedDict

class DefectTracker:
    def __init__(self, max_distance=100, max_age=15):
        self.next_id = 1
        self.objects = OrderedDict() # id -> centroid
        self.disappeared = OrderedDict() # id -> frame count
        self.bboxes = OrderedDict() # id -> bbox (x1,y1,x2,y2)
        self.confidences = OrderedDict() # id -> list of confidences
        self.classes = OrderedDict() # id -> class_id
        
        self.max_distance = max_distance
        self.max_age = max_age

    def register(self, centroid, bbox, conf, cls):
        self.objects[self.next_id] = centroid
        self.bboxes[self.next_id] = bbox
        self.confidences[self.next_id] = [conf]
        self.classes[self.next_id] = cls
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bboxes[objectID]
        del self.confidences[objectID]
        del self.classes[objectID]

    def update(self, rects, confs, clss):
        # rects: list of (x1, y1, x2, y2)
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_age:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], rects[i], confs[i], clss[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Compute distances between existing centroids and input centroids
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - inputCentroids, axis=2)
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bboxes[objectID] = rects[col]
                self.confidences[objectID].append(confs[col])
                if len(self.confidences[objectID]) > 10: self.confidences[objectID].pop(0)
                self.classes[objectID] = clss[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.max_age:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col], confs[col], clss[col])

        return self.objects

    def get_active_defects(self, min_hits=1):
        active = []
        for objectID in self.objects.keys():
            # A defect is active if it was seen in the current frame 
            # AND has been seen enough times overall (min_hits)
            if self.disappeared[objectID] == 0 and len(self.confidences[objectID]) >= min_hits:
                active.append({
                    'id': objectID,
                    'bbox': self.bboxes[objectID],
                    'conf': np.mean(self.confidences[objectID]),
                    'class': self.classes[objectID]
                })
        return active

    def clear(self):
        self.objects.clear()
        self.disappeared.clear()
        self.bboxes.clear()
        self.confidences.clear()
        self.classes.clear()
