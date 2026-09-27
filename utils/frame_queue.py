# utils/frame_queue.py
import threading
import time
from collections import deque

class FrameQueue:
    """
    High-performance thread-safe queue for industrial line-scan camera feeds.
    Drops stale intermediate frames if consumer is slow to prevent latency backlog.
    """
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.dropped_frames = 0
        self.total_frames = 0

    def put(self, frame, metadata=None):
        with self.lock:
            self.total_frames += 1
            if len(self.queue) == self.max_size:
                self.dropped_frames += 1
            self.queue.append((frame, metadata, time.time()))

    def get(self):
        with self.lock:
            if len(self.queue) > 0:
                return self.queue.popleft()
            return None, None, None

    def get_latest(self):
        """Retrieves latest frame and flushes stale older items."""
        with self.lock:
            if len(self.queue) > 0:
                item = self.queue[-1]
                self.dropped_frames += len(self.queue) - 1
                self.queue.clear()
                return item
            return None, None, None

    def peek(self):
        with self.lock:
            if len(self.queue) > 0:
                return self.queue[-1]
            return None, None, None

    def stats(self):
        with self.lock:
            return {
                "size": len(self.queue),
                "total": self.total_frames,
                "dropped": self.dropped_frames,
                "drop_rate": round(self.dropped_frames / max(1, self.total_frames) * 100, 2)
            }

    def clear(self):
        with self.lock:
            self.queue.clear()

