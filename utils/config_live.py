# utils/config_live.py
# Live inference runtime configuration.
# Performance values are auto-tuned from LINESCAN_PERF_CONFIG (config.py).
import os
try:
    from config import LINESCAN_PERF_CONFIG as _P
except ImportError:
    _P = {}

LIVE_CONFIG = {
    'camera_id': 1,                    # 0 = webcam; 'mindvision' = line-scan
    'resolution': (1280, 720),
    'fps_target': 60,                  # Target FPS (RTX 3050 handles 640px FP16 @ 60 FPS)
    'confidence_threshold': 0.30,
    'confidence_min': 0.10,
    'confidence_max': 0.95,

    # ── MindVision Line-Scan Camera ──────────────────────────
    'use_mindvision': True,
    'mindvision_slice_height':   _P.get('mindvision_slice_height',   128),  # 128px slices
    'mindvision_canvas_multiplier': _P.get('canvas_multiplier',       24),   # 24x = 3072px (exact 4:3 aspect ratio)
    'mindvision_enhance': True,        # CLAHE + sharpening (GPU if available)
    'use_gpu_clahe': _P.get('use_gpu_clahe', False),
    'use_pinned_memory': _P.get('use_pinned_memory', True),

    # ── Detection / Inference ────────────────────────────────
    'detection': {
        'model_path': 'runs/textile_detection/defect_model_pro_v1/weights/best.pt',
        'classifier_path': 'runs/textile_classification/fabric_classifier_v1/weights/best.pt',
        'use_classifier': True,
        'classifier_conf': 0.4,
        'device': 0,
        'imgsz': _P.get('imgsz_live', 640),          # 640 px (was 320)
        'half':  _P.get('use_half', True),            # FP16 Tensor Cores
        'batch': _P.get('batch_size', 8),             # tiles per forward pass
        'nms_backend': _P.get('nms_backend', 'torch'),
        'strict_mode': False,
    },

    # ── Tracking ─────────────────────────────────────────────
    'tracking': {
        'max_distance': 100,
        'max_age': 15,
        'min_hits': 3,
    },

    # ── Alerts ───────────────────────────────────────────────
    'alerts': {
        'enable_sound': True,
        'sound_frequency': 1000,
        'sound_duration': 200,
        'enable_visual': True,
        'alert_persistence': 10,
    },

    # ── Logging ──────────────────────────────────────────────
    'logging': {
        'enable': True,
        'log_dir': 'logs/',
        'save_frames': True,
        'frame_dir': 'saved_frames/',
        'interval': 1,
    },

    # ── Performance Monitoring ───────────────────────────────
    'performance': {
        'monitor': True,
        'gpu_monitoring': True,
        'show_latency': True,
        'fps_smoothing': 30,
        'adaptive_conf': True,         # raise conf if GPU > 85% load
        'gpu_load_threshold': 85,      # %
        'adaptive_conf_delta': 0.05,   # bump conf by this when overloaded
    },

    # ── MJPEG Stream & Camera Pipeline ───────────────────────
    'mjpeg_quality': _P.get('mjpeg_quality', 88),
    'max_mjpeg_fps': int(os.getenv('MAX_MJPEG_FPS', 30)),
    'debug_overlay': os.getenv('DEBUG_OVERLAY', '0') == '1',
    'clahe_clip_limit': float(os.getenv('CLAHE_CLIP_LIMIT', 2.5)),
    'camera_watchdog_interval': int(os.getenv('CAMERA_WATCHDOG_INTERVAL', 5)),

    # ── Threading ────────────────────────────────────────────
    'frame_queue_depth': _P.get('frame_queue_depth', 4),
    'inference_workers': _P.get('inference_workers', 8),
}

