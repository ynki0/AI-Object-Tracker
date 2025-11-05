import numpy as np

class ObjectTracker:
    def __init__(self):
        self.objects = {}
        self.next_id = 0

    def update(self, detections):
        tracked = []
        for det in detections:
            if len(det) < 4:
                continue
            x1, y1, x2, y2 = det[:4]
            self.objects[self.next_id] = (x1, y1, x2, y2)
            tracked.append((x1, y1, x2, y2, self.next_id))
            self.next_id += 1
        return tracked
