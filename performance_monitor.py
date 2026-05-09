# performance_monitor.py
import time
import psutil
try:
    import torch
except ImportError:
    torch = None

class PerformanceMonitor:
    def __init__(self, smoothing=30):
        self.smoothing = smoothing
        self.frame_times = []
        self.inference_latencies = []
        self.start_time = None
        self.last_frame_time = time.time()
        
    def start_inference(self):
        self.start_time = time.time()
        
    def end_inference(self):
        if self.start_time:
            latency = (time.time() - self.start_time) * 1000
            self.inference_latencies.append(latency)
            if len(self.inference_latencies) > self.smoothing:
                self.inference_latencies.pop(0)
            return latency
        return 0

    def update_fps(self):
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        if dt > 0:
            fps = 1.0 / dt
            self.frame_times.append(fps)
            if len(self.frame_times) > self.smoothing:
                self.frame_times.pop(0)
                
    def get_stats(self):
        avg_fps = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        avg_latency = sum(self.inference_latencies) / len(self.inference_latencies) if self.inference_latencies else 0
        
        gpu_mem = 0
        if torch and torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_reserved(0) / (1024**3) # GB
            
        cpu_usage = psutil.cpu_percent()
        
        return {
            'fps': avg_fps,
            'latency': avg_latency,
            'gpu_mem': gpu_mem,
            'cpu': cpu_usage
        }
