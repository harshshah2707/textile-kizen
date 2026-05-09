# utils/camera_handler.py
import cv2
import threading
import time

class CameraHandler:
    def __init__(self, camera_id=0, resolution=(1280, 720)):
        self.camera_id = camera_id
        self.resolution = resolution
        self.cap = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        # Support both integer IDs and URL strings (for IP Cameras)
        try:
            if isinstance(self.camera_id, str):
                # Use FFMPEG backend for URLs
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_FFMPEG)
            else:
                # Try DirectShow first, fallback to MSMF if it fails
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_MSMF)
        except Exception as e:
            print(f"Driver Error: {e}")
            self.cap = cv2.VideoCapture(self.camera_id) # Last resort: auto-detect
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return True

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame is not None, self.frame

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
