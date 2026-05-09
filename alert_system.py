# alert_system.py
import cv2
import winsound
import threading
import time

class AlertSystem:
    def __init__(self, enable_sound=True, frequency=1000, duration=200):
        self.enable_sound = enable_sound
        self.frequency = frequency
        self.duration = duration
        self.alert_active = False
        self.last_alert_time = 0
        self.alert_cooldown = 1.0 # seconds

    def trigger(self):
        now = time.time()
        if now - self.last_alert_time > self.alert_cooldown:
            self.last_alert_time = now
            if self.enable_sound:
                threading.Thread(target=self._play_sound, daemon=True).start()
            self.alert_active = True
            return True
        return False

    def _play_sound(self):
        try:
            winsound.Beep(self.frequency, self.duration)
        except:
            pass

    def draw_alert(self, frame, defect_count, persistence_counter):
        if persistence_counter > 0:
            # Flash red border
            h, w = frame.shape[:2]
            thickness = 10
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), thickness)
            
            # Warning text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"!!! DEFECT DETECTED ({defect_count}) !!!"
            (tw, th), _ = cv2.getTextSize(text, font, 1.5, 3)
            tx = (w - tw) // 2
            ty = 100
            
            # Semi-transparent background for text
            overlay = frame.copy()
            cv2.rectangle(overlay, (tx - 10, ty - th - 20), (tx + tw + 10, ty + 20), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            cv2.putText(frame, text, (tx, ty), font, 1.5, (255, 255, 255), 3)
            return True
        return False
