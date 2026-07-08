# utils/config_live.py
import os

LIVE_CONFIG = {
    'camera_id': 1,# Use 1 or 2 for USB Virtual Camera (DroidCam)
    'resolution': (1280, 720),
    'fps_target': 30,
    'confidence_threshold': 0.3, 
    'confidence_min': 0.1,
    'confidence_max': 0.95,
    
    # MindVision Line-Scan Camera Settings
    'use_mindvision': True,               # Enable line-scan camera support if present
    'mindvision_slice_height': 200,       # Height of each camera acquisition block
    'mindvision_canvas_multiplier': 10,   # Canvas scrolling height buffer (10x slice height)
    'mindvision_enhance': True,           # Enable CLAHE + Sharpening filter on line-scan feed
    
    
    'detection': {
        'model_path': 'runs/textile_detection/defect_model_pro_v1/weights/best.pt',
        'classifier_path': 'runs/textile_classification/fabric_classifier_v1/weights/best.pt',
        'use_classifier': True,
        'classifier_conf': 0.4,
        'device': 0,
        'imgsz': 640,
        'strict_mode': False,
    },
    
    'tracking': {
        'max_distance': 100,
        'max_age': 15,
        'min_hits': 3,
    },
    
    'alerts': {
        'enable_sound': True,
        'sound_frequency': 1000,
        'sound_duration': 200,
        'enable_visual': True,
        'alert_persistence': 10,
    },
    
    'logging': {
        'enable': True,
        'log_dir': 'logs/',
        'save_frames': True,
        'frame_dir': 'saved_frames/',
        'interval': 1,
    },
    
    'performance': {
        'monitor': True,
        'gpu_monitoring': True,
        'show_latency': True,
        'fps_smoothing': 30,
    }
}
