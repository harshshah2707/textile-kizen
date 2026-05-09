# utils/frame_queue.py
import threading
from collections import deque

class FrameQueue:
    """Thread-safe queue for camera frames with a fixed size."""
    def __init__(self, max_size=2):
        self.queue = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def put(self, frame):
        with self.lock:
            self.queue.append(frame)

    def get(self):
        with self.lock:
            if len(self.queue) > 0:
                return self.queue.popleft()
            return None

    def peek(self):
        with self.lock:
            if len(self.queue) > 0:
                return self.queue[-1]
            return None

    def clear(self):
        with self.lock:
            self.queue.clear()
