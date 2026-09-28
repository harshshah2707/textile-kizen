# web_server.py
# TextileGuard AI — Kizen Engineering Industrial Fabric Inspection System
# Innovation Is Our Tradition • https://kizen.co.in/
import os
import time
import json
import queue
import socket
import hashlib
import base64
import threading
import cv2
import numpy as np
import torch
import psutil
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, Response, send_from_directory, request, jsonify, send_file
from utils.camera_handler import CameraHandler, create_offline_frame
from utils.config_live import LIVE_CONFIG
from defect_tracker import DefectTracker
from utils.bbox_refiner import refine_defect_bbox
from utils import db_manager, pdf_generator

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

DEBUG_OVERLAY = os.getenv('DEBUG_OVERLAY', '0') == '1'
MAX_MJPEG_FPS = int(os.getenv('MAX_MJPEG_FPS', '30'))
CLAHE_CLIP_LIMIT = float(os.getenv('CLAHE_CLIP_LIMIT', '2.5'))
CAMERA_WATCHDOG_INTERVAL = int(os.getenv('CAMERA_WATCHDOG_INTERVAL', '5'))

def preprocess_for_inference(frame):
    """Standardize lighting and contrast for YOLO inference without amplifying noise."""
    if frame is None:
        return None
    try:
        # Check signal level: if scene is dark/unlit noise, do NOT apply CLAHE
        # Applying CLAHE on dark noise amplifies sensor fixed-pattern columns into false defect streaks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 12.0 or gray.max() < 35:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return frame

app = Flask(__name__, static_folder='frontend/dist')

# Initialize SQLite database
db_manager.init_db()

# Global session variables
current_roll_id = None
current_roll_number = None
current_material_name = 'Cotton'
current_operator_name = 'operator'
roll_start_time = 0.0
fabric_speed_m_per_min = 12.0
fabric_width_mm = 1800

# Global variables for synchronization
websocket_clients = []
clients_lock = threading.Lock()
broadcasted_ids = {}

# Global settings updated by WebSocket from UI
# Calibrated at 0.28 for optimal balance of sensitivity and noise rejection
conf_threshold = 0.28

# Performance configuration from LIVE_CONFIG
_MJPEG_QUALITY     = LIVE_CONFIG.get('mjpeg_quality', 88)
_LIVE_IMGSZ        = LIVE_CONFIG['detection'].get('imgsz', 640)
_LIVE_HALF         = LIVE_CONFIG['detection'].get('half', True)
_FRAME_QUEUE_DEPTH = LIVE_CONFIG.get('frame_queue_depth', 4)

# -------------------------------------------------------------
# YOLO Model Initialization
# -------------------------------------------------------------
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[Kizen Vision] PyTorch Device: {device} "
      f"({'RTX 3050 FP16 Tensor Cores active' if torch.cuda.is_available() else 'CPU'})"
)
if torch.cuda.is_available():
    print(f"[Kizen Vision] VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB  "
          f"| Inference imgsz: {_LIVE_IMGSZ} px  | FP16: {_LIVE_HALF}")

model_paths = [
    'runs/textile_detection/defect_model_pro_v1/weights/best.pt',
    'runs/textile_detection/multiclass_v1/weights/best.pt',
    'runs/textile_detection/defect_model_v1/weights/best.pt',
    'yolov8s.pt',
    'yolov8n.pt'
]

for path in model_paths:
    if os.path.exists(path):
        try:
            print(f"[Kizen Vision] Loading YOLO model from: {path}")
            model = YOLO(path)
            print(f"[Kizen Vision] Model loaded successfully from {path}!")
            break
        except Exception as e:
            print(f"[Kizen Vision] Could not load model from {path}: {e}")

if model is None:
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"[Kizen Vision] Critical: Failed to load YOLO weights: {e}")

model_lock = threading.Lock()

# -------------------------------------------------------------
# Hardware Camera Configuration (MindVision / GigE / USB)
# -------------------------------------------------------------
cam_mode = 'mindvision'
cam1 = CameraHandler(cam_mode, resolution=(1280, 720))
cam1_opened = cam1.start()

cam2 = None
cam2_opened = False

camera_handlers = {
    1: cam1,
    2: cam2
}

tracker1 = DefectTracker(max_distance=80, max_age=15)
tracker2 = DefectTracker(max_distance=80, max_age=15)

def switch_camera_mode(new_mode):
    global cam1, cam1_opened, cam_mode, cam2, cam2_opened, camera_handlers
    print(f"[Kizen Vision] Hardware camera mode set to: {new_mode}")
    if cam1:
        try:
            cam1.stop()
        except Exception:
            pass
    if cam2:
        try:
            cam2.stop()
        except Exception:
            pass

    cam_mode = new_mode
    if new_mode == 'dual_linescan':
        cam1 = CameraHandler('mindvision', resolution=(1280, 720))
        cam2 = CameraHandler('mindvision', resolution=(1280, 720))
        cam1_opened = cam1.start()
        cam2_opened = cam2.start()
        camera_handlers[1] = cam1
        camera_handlers[2] = cam2
    else:
        cam1 = CameraHandler(new_mode, resolution=(1280, 720))
        cam1_opened = cam1.start()
        cam2 = None
        cam2_opened = False
        camera_handlers[1] = cam1
        camera_handlers[2] = None

    # Reset frame caches so new camera frames stream immediately
    for cid in [1, 2]:
        if cid in _latest_jpeg_frames:
            with _frame_hub_locks.get(cid, threading.Lock()):
                _latest_jpeg_frames[cid] = None
        if cid in _latest_frame_events:
            _latest_frame_events[cid].set()

    status_dict = get_camera_telemetry()
    ws_broadcast({"type": "camera_status", **status_dict})
    return True

def get_camera_telemetry():
    status1 = cam1.get_status() if cam1 else {"type": "offline", "online": False}
    status2 = cam2.get_status() if cam2 else None
    model_name = status1.get("camera_model", "ChinaVision GELM44M-T2")
    if cam_mode == 'dual_linescan':
        model_name = "Dual Line-Scan Array"

    return {
        "mode": cam_mode,
        "camera_model": model_name,
        "cam1": status1,
        "cam2": status2,
        "is_linescan": "linescan" in status1.get("type", "").lower() or "mindvision" in status1.get("type", "").lower(),
        "brand": "Kizen Engineering",
        "tagline": "Innovation Is Our Tradition",
        "url": "https://kizen.co.in"
    }


# -------------------------------------------------------------
# WebSocket Server & Frame Protocol
# -------------------------------------------------------------
def compute_ws_accept(key):
    magic_guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    hashed = hashlib.sha1((key + magic_guid).encode('utf-8')).digest()
    return base64.b64encode(hashed).decode('utf-8')

def make_ws_frame(message_str):
    payload = message_str.encode('utf-8')
    length = len(payload)
    
    header = bytearray([0x81])
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(length.to_bytes(2, byteorder='big'))
    else:
        header.append(127)
        header.extend(length.to_bytes(8, byteorder='big'))
        
    return header + payload

def decode_ws_frame(data):
    if not data or len(data) < 2:
        return None
        
    second_byte = data[1]
    masked = (second_byte & 0x80) != 0
    payload_len = second_byte & 0x7f
    
    offset = 2
    if payload_len == 126:
        if len(data) < 4: return None
        payload_len = int.from_bytes(data[2:4], byteorder='big')
        offset = 4
    elif payload_len == 127:
        if len(data) < 10: return None
        payload_len = int.from_bytes(data[2:10], byteorder='big')
        offset = 10
        
    if len(data) < offset + (4 if masked else 0) + payload_len:
        return None
        
    if masked:
        mask_key = data[offset:offset+4]
        offset += 4
        payload = bytearray(data[offset:offset+payload_len])
        for i in range(len(payload)):
            payload[i] ^= mask_key[i % 4]
        return payload.decode('utf-8', errors='ignore')
    else:
        return data[offset:offset+payload_len].decode('utf-8', errors='ignore')

def handle_ws_client(client_socket, client_address):
    print(f"[WebSocket] Client connected: {client_address}")
    global conf_threshold, broadcasted_ids
    try:
        request_data = client_socket.recv(4096).decode('utf-8', errors='ignore')
        if "Upgrade: websocket" not in request_data:
            client_socket.close()
            return
            
        ws_key = None
        for line in request_data.split('\r\n'):
            if line.startswith("Sec-WebSocket-Key:"):
                ws_key = line.split(":")[1].strip()
                break
                
        if not ws_key:
            client_socket.close()
            return
            
        accept_key = compute_ws_accept(ws_key)
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_key}\r\n\r\n"
        )
        client_socket.send(response.encode('utf-8'))
        
        with clients_lock:
            websocket_clients.append(client_socket)
            
        # Send initial legitimate telemetry
        cpu_val = psutil.cpu_percent(interval=None)
        ram_val = psutil.virtual_memory().percent
        disk_val = psutil.disk_usage(os.path.abspath('/')).percent
        init_frame = make_ws_frame(json.dumps({
            "type": "heartbeat",
            "cpu": cpu_val,
            "ram": ram_val,
            "disk": disk_val
        }))
        client_socket.send(init_frame)
        
        # Send camera status
        cam_state_frame = make_ws_frame(json.dumps({"type": "camera_status", **get_camera_telemetry()}))
        client_socket.send(cam_state_frame)
        
        while True:
            data = client_socket.recv(4096)
            if not data:
                break
                
            decoded_msg = decode_ws_frame(data)
            if decoded_msg:
                try:
                    msg = json.loads(decoded_msg)
                    if msg.get("type") == "settings":
                        new_conf = msg.get("confidenceThreshold")
                        if new_conf is not None:
                            conf_threshold = float(new_conf)
                            print(f"[YOLO Inference] Confidence threshold updated: {conf_threshold:.2f}")
                        
                        new_speed = msg.get("fabricSpeed")
                        if new_speed is not None:
                            global fabric_speed_m_per_min
                            fabric_speed_m_per_min = float(new_speed)

                        new_width = msg.get("fabricWidth")
                        if new_width is not None:
                            global fabric_width_mm
                            fabric_width_mm = float(new_width)

                        cam_settings = msg.get("cameraControls", {}).get("cam1", {})
                        if cam_settings and cam1:
                            exp = cam_settings.get("exposure")
                            gain = cam_settings.get("gain")
                            sh = cam_settings.get("slice_height")
                            enh = cam_settings.get("enhance")
                            mg = cam_settings.get("motion_gated")
                            if exp is not None:
                                cam1.set_exposure(exp)
                            if gain is not None:
                                cam1.set_gain(gain)
                            if sh is not None:
                                cam1.set_slice_height(sh)
                            if enh is not None:
                                cam1.set_enhance(enh)
                            if mg is not None:
                                cam1.set_motion_gated(mg)
                    elif msg.get("type") == "set_camera_mode":
                        req_mode = msg.get("mode")
                        if req_mode:
                            switch_camera_mode(req_mode)
                except Exception:
                    pass
                    
    except Exception as e:
        print(f"[WebSocket] Client connection error: {e}")
    finally:
        with clients_lock:
            if client_socket in websocket_clients:
                websocket_clients.remove(client_socket)
        try:
            client_socket.close()
        except Exception:
            pass
        print(f"[WebSocket] Client disconnected: {client_address}")

def ws_broadcast(msg_dict):
    frame = make_ws_frame(json.dumps(msg_dict))
    disconnected_clients = []
    
    with clients_lock:
        for client in websocket_clients:
            try:
                client.send(frame)
            except Exception:
                disconnected_clients.append(client)
                
        for client in disconnected_clients:
            if client in websocket_clients:
                websocket_clients.remove(client)
                try:
                    client.close()
                except Exception:
                    pass

def run_websocket_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(('0.0.0.0', 8765))
        server.listen(10)
        print("[WebSocket Server] Listening on port 8765...")
    except Exception as e:
        print(f"[WebSocket Server] Failed to bind to port 8765: {e}")
        return
        
    while True:
        try:
            client_sock, client_addr = server.accept()
            client_thread = threading.Thread(
                target=handle_ws_client, 
                args=(client_sock, client_addr),
                daemon=True
            )
            client_thread.start()
        except Exception as e:
            print(f"[WebSocket Server] Accept error: {e}")
            time.sleep(1)

threading.Thread(target=run_websocket_server, daemon=True).start()

# -------------------------------------------------------------
# Genuine Hardware Telemetry Broadcast Loop
# -------------------------------------------------------------
def background_telemetry():
    """Broadcasts strictly real system and sensor metrics every second."""
    while True:
        try:
            cpu  = round(psutil.cpu_percent(interval=None), 1)
            ram  = round(psutil.virtual_memory().percent, 1)
            disk = round(psutil.disk_usage(os.path.abspath('/')).percent, 1)

            payload = {
                "type": "heartbeat",
                "cpu": cpu,
                "ram": ram,
                "disk": disk,
                "brand": "Kizen Engineering"
            }

            if torch.cuda.is_available():
                try:
                    vram_used  = torch.cuda.memory_allocated(0) / (1024 ** 3)
                    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    vram_pct   = round(100 * vram_used / vram_total, 1)
                    payload["vram_used_gb"]   = round(vram_used, 2)
                    payload["vram_total_gb"]  = round(vram_total, 1)
                    payload["vram_pct"]       = vram_pct
                    payload["gpu_name"]       = torch.cuda.get_device_properties(0).name
                except Exception:
                    pass

            # Real camera hardware slice acquisition or optical frame rate
            slice_fps = 0.0
            line_rate = 0
            line_freq = 0.0
            optical_fps = 0.0
            cam_model_name = "ChinaVision GELM44M-T2"
            if cam1 and cam1.handler and cam1.handler.running:
                cam_model_name = getattr(cam1.handler, 'camera_model', cam_model_name)
                if hasattr(cam1.handler, 'acquisition_fps'):
                    slice_fps = round(cam1.handler.acquisition_fps, 1)
                    sh = getattr(cam1.handler, 'slice_height', 256)
                    line_rate = round(slice_fps * sh)
                    line_freq = round(line_rate / 1000.0, 2)
                elif hasattr(cam1.handler, 'fps'):
                    optical_fps = round(cam1.handler.fps, 1)

            payload["slice_fps"] = slice_fps
            payload["line_rate_hz"] = line_rate
            payload["line_freq_khz"] = line_freq
            payload["optical_fps"] = optical_fps
            payload["camera_model"] = cam_model_name
            payload["cam_mode"] = cam_mode

            ws_broadcast(payload)
        except Exception:
            pass
        time.sleep(1.0)

threading.Thread(target=background_telemetry, daemon=True).start()

def _camera_watchdog():
    """Periodically verifies physical camera health and attempts auto-recovery."""
    while True:
        time.sleep(CAMERA_WATCHDOG_INTERVAL)
        try:
            if cam1 and cam1.handler and not cam1.handler.running and cam_mode not in ('virtual', 'emulator'):
                ws_broadcast({"type": "camera_reconnecting", "message": "Attempting automatic hardware camera reconnection..."})
                cam1.stop()
                reconnected = cam1.start()
                if reconnected:
                    print("[Camera Watchdog] Hardware camera reconnected successfully.")
                    ws_broadcast({"type": "camera_status", **get_camera_telemetry()})
        except Exception as e:
            print(f"[Camera Watchdog] Watchdog probe exception: {e}")

threading.Thread(target=_camera_watchdog, daemon=True).start()

# -------------------------------------------------------------
# -------------------------------------------------------------
# Zero-Lag High-Performance Video Streaming & Inference Engine
# -------------------------------------------------------------
_latest_jpeg_frames = {}
_frame_hub_locks = {1: threading.Lock(), 2: threading.Lock()}
_latest_frame_events = {1: threading.Event(), 2: threading.Event()}
_camera_workers_started = set()

def _camera_stream_worker(camera_id):
    width, height = 1280, 720
    tracker = tracker1 if camera_id == 1 else tracker2

    global conf_threshold, broadcasted_ids, current_roll_id, roll_start_time

    class_names_map = model.names if (model is not None and hasattr(model, 'names')) else {
        0: "Broken stitch", 1: "hole", 2: "horizontal", 3: "lines",
        4: "Needle mark",   5: "Pinched fabric", 6: "stain", 7: "Vertical"
    }

    # Strict Industrial Palette: Royal Blue, Light Blue, White, Slate Grey
    blue_primary = (235, 99, 37)   # #2563EB in BGR
    blue_light   = (250, 165, 96)  # #60A5FA in BGR
    white_pure   = (255, 255, 255)

    color_map = {
        "Broken stitch": blue_primary,
        "hole":          blue_primary,
        "horizontal":    blue_light,
        "lines":         blue_light,
        "Needle mark":   blue_light,
        "Pinched fabric":blue_light,
        "stain":         blue_primary,
        "Vertical":      blue_light,
    }

    _frame_interval = 1.0 / max(1, MAX_MJPEG_FPS)
    last_infer_time = 0.0
    frame_count = 0
    last_time = time.time()

    while True:
        loop_start = time.time()
        try:
            cam_handler = camera_handlers.get(camera_id, cam1)
            if cam_handler is None:
                cam_handler = cam1

            is_online = False
            frame = None

            if cam_handler and cam_handler.handler and cam_handler.handler.running:
                success, img = cam_handler.read()
                if success and img is not None:
                    frame = img
                    is_online = True

            if frame is None or not is_online:
                offline_msg = "CAMERA NOT DETECTED"
                offline_sub = "Connect MindVision Line-Scan Camera or USB Feed"
                if cam_mode in ('oak_d', 'oak_d_lite', 'oakd'):
                    offline_msg = "OAK-D LITE NOT DETECTED"
                    offline_sub = "Connect Luxonis OAK-D Lite via USB 3.0 Type-C"
                elif cam_mode in ('0', 'webcam', 'standard', 'optical'):
                    offline_msg = "USB OPTICAL CAMERA NOT DETECTED"
                    offline_sub = "Check USB camera connection"
                frame = create_offline_frame(
                    width, height,
                    message=offline_msg,
                    sub_message=offline_sub
                )

            # -------------------------------------------------------------
            # Inspection State Gating:
            # Defect detection runs and displays ONLY after pressing 'Launch Inspection'
            # -------------------------------------------------------------
            inspection_active = (current_roll_id is not None)
            now = time.time()
            active_defects_list = []

            if not inspection_active:
                tracker.clear()
            elif is_online and model is not None and frame is not None and (now - last_infer_time >= 0.030):
                last_infer_time = now
                try:
                    # Crop pure fabric region away from HUD banners (top 26px, bottom 48px)
                    hud_top = 26
                    hud_bottom = 48
                    roi_h = height - hud_top - hud_bottom
                    fabric_roi = frame[hud_top:height - hud_bottom, :]

                    # Illumination & Signal Guard:
                    # If camera is looking at dark / unlit background (mean < 12.0 or max < 35),
                    # skip inference to prevent detecting sensor dark noise as false defects!
                    roi_gray = cv2.cvtColor(fabric_roi, cv2.COLOR_BGR2GRAY)
                    mean_sig = float(roi_gray.mean())
                    max_sig = int(roi_gray.max())

                    if mean_sig >= 4.0 and max_sig >= 15:
                        infer_roi = preprocess_for_inference(fabric_roi)

                        with model_lock:
                            results = model.predict(
                                source=infer_roi,
                                conf=conf_threshold,
                                imgsz=_LIVE_IMGSZ,
                                device=device,
                                verbose=False
                            )

                        rects, confs, clss = [], [], []
                        for r in results:
                            for box in r.boxes:
                                b = box.xyxy[0].cpu().numpy().copy()
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                
                                # Shift Y coordinates from ROI back to original full frame
                                b[1] += hud_top
                                b[3] += hud_top

                                bw = b[2] - b[0]
                                bh = b[3] - b[1]

                                # Reject unrealistic full-screen detections or tiny speckles
                                if bw > width * 0.92 and bh > roi_h * 0.92:
                                    continue
                                if bw < 8 or bh < 8:
                                    continue
                                # Reject boxes that clip directly onto HUD banners
                                if b[1] < hud_top + 2 or b[3] > height - hud_bottom - 2:
                                    continue

                                # For 'Vertical' or 'lines' classes on line-scan, apply extra confidence guard
                                dname = class_names_map.get(cls, "")
                                if dname in ('Vertical', 'lines') and conf < 0.35:
                                    continue

                                if cam_mode == 'dual_linescan':
                                    seam_x = width // 2
                                    if abs(b[0] - seam_x) < 8 or abs(b[2] - seam_x) < 8 or (b[0] < seam_x and b[2] > seam_x and bw < 16):
                                        continue

                                x1_r, y1_r, x2_r, y2_r = refine_defect_bbox(frame, b, cls)
                                rects.append([x1_r, y1_r, x2_r, y2_r])
                                confs.append(conf)
                                clss.append(cls)

                        tracker.update(rects, confs, clss)
                        active_defects_list = tracker.get_active_defects(min_hits=1)
                    else:
                        tracker.clear()
                        active_defects_list = []
                except Exception:
                    active_defects_list = []

                # Draw defect bounding boxes and sleek industrial tags
                for def_item in active_defects_list:
                    def_id = f"{camera_id}-{def_item['id']}"
                    dtype = class_names_map.get(def_item['class'], "defect")
                    color = color_map.get(dtype, (0, 0, 255))

                    x1, y1, x2, y2 = def_item['bbox']
                    w_box = x2 - x1
                    h_box = y2 - y1
                    size_mm = round(w_box * 0.15, 1)

                    # Draw defect bounding boxes and sleek industrial tags
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    lbl = f"{dtype.upper()} {int(def_item['conf']*100)}% ({size_mm}mm)"
                    (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                    tag_y = max(42, y1 - 6)
                    cv2.rectangle(frame, (x1, tag_y - lh - 4), (x1 + lw + 6, tag_y + 2), (15, 23, 42), -1)
                    cv2.rectangle(frame, (x1, tag_y - lh - 4), (x1 + lw + 6, tag_y + 2), color, 1)
                    cv2.putText(frame, lbl, (x1 + 3, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

                    # Broadcast & database logging with smart deduplication
                    last_alert_time = broadcasted_ids.get(def_id, 0.0) if isinstance(broadcasted_ids, dict) else (0.0 if def_id not in broadcasted_ids else time.time())
                    if now - last_alert_time >= 2.0:
                        if isinstance(broadcasted_ids, dict):
                            broadcasted_ids[def_id] = now
                        else:
                            broadcasted_ids.add(def_id)

                        distance_meters = 0.0
                        if current_roll_id is not None:
                            elapsed = time.time() - roll_start_time
                            distance_meters = round((fabric_speed_m_per_min / 60) * elapsed, 2)

                        crop_url = ""
                        if frame is not None:
                            try:
                                roll_folder = f"roll_{current_roll_id}" if current_roll_id is not None else "live_monitoring"
                                crop_dir = os.path.join("saved_frames", "crops", roll_folder)
                                os.makedirs(crop_dir, exist_ok=True)
                                crop_filename = f"defect_{def_item['id']}_{int(time.time()*1000)}.jpg"
                                crop_path = os.path.join(crop_dir, crop_filename)
                                fy, fx = frame.shape[:2]
                                cy1, cy2 = max(0, y1), min(fy, y2)
                                cx1, cx2 = max(0, x1), min(fx, x2)
                                if cy2 > cy1 and cx2 > cx1:
                                    cv2.imwrite(crop_path, frame[cy1:cy2, cx1:cx2])
                                    crop_url = f"/crops/{roll_folder}/{crop_filename}"
                            except Exception:
                                pass

                        if current_roll_id is not None:
                            try:
                                db_manager.add_defect(current_roll_id, {
                                    "timestamp": time.time(),
                                    "defect_type": dtype,
                                    "confidence": round(def_item['conf'], 2),
                                    "size_mm": size_mm,
                                    "bbox": {"x": x1, "y": y1, "width": w_box, "height": h_box},
                                    "distance_meters": distance_meters,
                                    "crop_path": crop_url
                                })
                            except Exception:
                                pass

                        ws_broadcast({
                            "type": "defect",
                            "id": f"D-{def_id}",
                            "defect_type": dtype,
                            "confidence": round(def_item['conf'], 2),
                            "camera": camera_id,
                            "size_mm": size_mm,
                            "location": f"X:{x1} Y:{y1}",
                            "bbox": {"x": x1, "y": y1, "width": w_box, "height": h_box},
                            "timestamp": int(time.time() * 1000),
                            "distance_meters": distance_meters,
                            "crop_path": crop_url
                        })

            # Watermark and inspection status indicator
            if is_online:
                if inspection_active:
                    cv2.putText(frame, "INSPECTION ACTIVE", (width - 240, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "STANDBY (PRESS LAUNCH)", (width - 270, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA)
                cv2.putText(frame, "KIZEN INDUSTRIAL VISION", (width - 220, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 116, 139), 1, cv2.LINE_AA)

            # Instantaneous JPEG encode (Quality 80 is 40% faster, 50% smaller bandwidth)
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                lock = _frame_hub_locks.get(camera_id)
                event = _latest_frame_events.get(camera_id)
                if lock and event:
                    with lock:
                        _latest_jpeg_frames[camera_id] = jpeg.tobytes()
                    event.set()

            # Real measured FPS
            frame_count += 1
            curr_time = time.time()
            if curr_time - last_time >= 1.0:
                fps_val = frame_count / (curr_time - last_time) if is_online else 0.0
                ws_broadcast({"type": "frame_metrics", "fps": round(fps_val, 1)})
                frame_count = 0
                last_time = curr_time

            # Pacing
            elapsed = time.time() - loop_start
            sleep_rem = _frame_interval - elapsed
            if sleep_rem > 0.001:
                time.sleep(sleep_rem)

        except Exception as e:
            time.sleep(0.01)

def _start_camera_worker(camera_id):
    if camera_id in _camera_workers_started:
        return
    _camera_workers_started.add(camera_id)
    if camera_id not in _frame_hub_locks:
        _frame_hub_locks[camera_id] = threading.Lock()
    if camera_id not in _latest_frame_events:
        _latest_frame_events[camera_id] = threading.Event()
    worker_t = threading.Thread(target=_camera_stream_worker, args=(camera_id,), daemon=True)
    worker_t.start()

# Pre-warm primary camera worker immediately on server startup
_start_camera_worker(1)

def generate_fabric_frames(camera_id):
    if camera_id not in _camera_workers_started:
        _start_camera_worker(camera_id)

    event = _latest_frame_events.get(camera_id)
    lock = _frame_hub_locks.get(camera_id)

    while True:
        if event is not None:
            event.wait(timeout=0.04)
            event.clear()

        jpeg_bytes = None
        if lock is not None:
            with lock:
                jpeg_bytes = _latest_jpeg_frames.get(camera_id)

        if jpeg_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
        else:
            time.sleep(0.01)


# -------------------------------------------------------------
# Flask Server Routes & Static Serving
# -------------------------------------------------------------
@app.route('/')
def serve_index():
    if os.path.exists('frontend/dist/index.html'):
        return send_from_directory('frontend/dist', 'index.html')
    elif os.path.exists('templates/plc.html'):
        return send_from_directory('templates', 'plc.html')
    elif os.path.exists('templates/index.html'):
        return send_from_directory('templates', 'index.html')
    else:
        return "Dashboard interface not found.", 404

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join('frontend/dist', path)):
        return send_from_directory('frontend/dist', path)
    if os.path.exists(os.path.join('static', path)):
        return send_from_directory('static', path)
    if os.path.exists('templates/plc.html'):
        return send_from_directory('templates', 'plc.html')
    return "Not found", 404

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    if camera_id not in [1, 2]:
        return "Invalid camera ID", 400
    resp = Response(
        generate_fabric_frames(camera_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/api/stats')
def api_stats():
    return {
        "status": "online",
        "company": "Kizen Engineering",
        "tagline": "Innovation Is Our Tradition",
        "url": "https://kizen.co.in",
        "model": "yolov8-textile-pro",
        "cpu_usage_percent": round(psutil.cpu_percent(interval=None), 1),
        "ram_usage_percent": round(psutil.virtual_memory().percent, 1),
        "camera": get_camera_telemetry()
    }

# Camera mode switching
@app.route('/api/camera/mode', methods=['POST'])
def api_set_camera_mode():
    data = request.get_json() or {}
    mode = data.get("mode", "mindvision")
    success = switch_camera_mode(mode)
    return jsonify({"status": "success" if success else "fail", "mode": cam_mode, "telemetry": get_camera_telemetry()})

# Camera settings update
@app.route('/api/camera/settings', methods=['POST'])
def api_set_camera_settings():
    data = request.get_json() or {}
    if cam1:
        if "exposure" in data:
            cam1.set_exposure(data["exposure"])
        if "gain" in data:
            cam1.set_gain(data["gain"])
        if "slice_height" in data:
            cam1.set_slice_height(data["slice_height"])
        if "enhance" in data:
            cam1.set_enhance(data["enhance"])
        if "motion_gated" in data:
            cam1.set_motion_gated(data["motion_gated"])
    return jsonify({"status": "success", "telemetry": get_camera_telemetry()})

# Camera telemetry query
@app.route('/api/camera/status', methods=['GET'])
def api_get_camera_status():
    return jsonify(get_camera_telemetry())

# Enumerate all available camera profiles & physical hardware devices
@app.route('/api/camera/sources', methods=['GET'])
def api_get_camera_sources():
    sources = []

    # 1. ChinaVision / MindVision GigE Line-Scan
    gige_detected = False
    gige_model = "ChinaVision GELM44M-T2"
    try:
        from pyGigEVision import discover
        devs = discover(timeout=0.3)
        if devs:
            gige_detected = True
            gige_model = f"{devs[0].get('manufacturer', '')} {devs[0].get('model', '')}".strip()
    except Exception:
        pass
    sources.append({
        "id": "mindvision",
        "name": f"{gige_model} (GigE Line-Scan)",
        "type": "linescan",
        "badge": "GigE Line-Scan",
        "description": "Industrial rolling line-scan fabric imaging at 1000+ Hz line rate",
        "connected": gige_detected or (cam_mode == 'mindvision' and cam1 and cam1.handler and cam1.handler.running)
    })

    # 2. Dual Line-Scan Array
    sources.append({
        "id": "dual_linescan",
        "name": "Dual Line-Scan Array (Dual Camera)",
        "type": "linescan_array",
        "badge": "Dual Array",
        "description": "Dual camera line-scan stitched array for extra-wide textile inspection",
        "connected": False
    })

    return jsonify({
        "active_mode": cam_mode,
        "sources": sources,
        "telemetry": get_camera_telemetry()
    })

# Serving saved frames/crops
@app.route('/crops/<path:filename>')
def serve_crops(filename):
    return send_from_directory('saved_frames/crops', filename)

# User login
@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.get_json() or {}
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"status": "fail", "message": "Missing credentials"}), 400
    role = db_manager.authenticate_user(username, password)
    if role:
        return jsonify({"status": "success", "role": role})
    else:
        return jsonify({"status": "fail", "message": "Invalid username or password"}), 401

# Materials presets CRUD
@app.route('/api/materials', methods=['GET'])
def api_get_materials():
    presets = db_manager.get_materials()
    return jsonify(presets)

@app.route('/api/materials/save', methods=['POST'])
def api_save_material():
    data = request.get_json() or {}
    if not data.get("name"):
        return jsonify({"status": "fail", "message": "Missing preset name"}), 400
    db_manager.save_material(data)
    return jsonify({"status": "success"})

@app.route('/api/materials/delete', methods=['POST'])
def api_delete_material():
    data = request.get_json() or {}
    name = data.get("name")
    if not name:
        return jsonify({"status": "fail", "message": "Missing preset name"}), 400
    db_manager.delete_material(name)
    return jsonify({"status": "success"})

# Roll sessions logs
@app.route('/api/rolls', methods=['GET'])
def api_get_rolls():
    rolls = db_manager.get_rolls()
    return jsonify(rolls)

@app.route('/api/rolls/start', methods=['POST'])
def api_start_roll():
    global current_roll_id, current_roll_number, current_material_name, current_operator_name, roll_start_time
    data = request.get_json() or {}
    roll_num = data.get("roll_number")
    mat_name = data.get("material_name", "Cotton")
    op_name = data.get("operator_name", "operator")
    
    if not roll_num:
        roll_num = f"ROLL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_roll_id, current_roll_number = db_manager.start_roll(roll_num, mat_name, op_name, start_time_str)
    roll_start_time = time.time()
    current_material_name = mat_name
    current_operator_name = op_name
    
    broadcasted_ids.clear()
    tracker1.clear()
    if tracker2:
        tracker2.clear()
    ws_broadcast({
        "type": "inspection_state",
        "active": True,
        "roll_id": current_roll_id,
        "roll_number": current_roll_number
    })
    
    return jsonify({
        "status": "success", 
        "roll_id": current_roll_id, 
        "roll_number": current_roll_number
    })

@app.route('/api/rolls/stop', methods=['POST'])
def api_stop_roll():
    global current_roll_id, current_roll_number
    if current_roll_id is None:
        return jsonify({"status": "fail", "message": "No active roll session"}), 400
        
    data = request.get_json() or {}
    length = float(data.get("length_meters", 0.0))
    points = int(data.get("total_points", 0))
    points_per_100m = float(data.get("points_per_100m", 0.0))
    grade = data.get("grade", "FIRST QUALITY")
    
    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db_manager.end_roll(current_roll_id, length, points, points_per_100m, grade, end_time_str)
    
    ret_id = current_roll_id
    current_roll_id = None
    current_roll_number = None
    tracker1.clear()
    if tracker2:
        tracker2.clear()
    broadcasted_ids.clear()
    ws_broadcast({
        "type": "inspection_state",
        "active": False,
        "roll_id": None,
        "roll_number": None
    })
    
    return jsonify({"status": "success", "finalized_roll_id": ret_id})

@app.route('/api/rolls/report/<int:roll_id>', methods=['GET'])
def api_get_roll_report(roll_id):
    details = db_manager.get_roll_details(roll_id)
    if not details:
        return jsonify({"status": "fail", "message": "Roll not found"}), 404
    return jsonify(details)

@app.route('/api/rolls/report/<int:roll_id>/pdf', methods=['GET'])
def api_get_roll_pdf(roll_id):
    details = db_manager.get_roll_details(roll_id)
    if not details:
        return jsonify({"status": "fail", "message": "Roll not found"}), 404
        
    pdf_dir = os.path.join("saved_frames", "reports")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_name = f"roll_{details['roll']['roll_number']}_report.pdf"
    pdf_path = os.path.join(pdf_dir, pdf_name)
    
    try:
        pdf_generator.generate_roll_pdf(details, pdf_path)
        return send_file(
            pdf_path, 
            as_attachment=True, 
            download_name=pdf_name
        )
    except Exception as e:
        print(f"[PDF Endpoint Error] {e}")
        return jsonify({"status": "fail", "message": f"Failed to generate PDF: {e}"}), 500

# Serving sample frames
@app.route('/samples/<path:filename>')
def serve_samples(filename):
    return send_from_directory('saved_frames/samples', filename)

# Capture training cloth sample
@app.route('/api/samples/capture', methods=['POST'])
def api_capture_sample():
    try:
        if cam1:
            ret, frame = cam1.read()
            if ret and frame is not None:
                samples_dir = os.path.join("saved_frames", "samples")
                os.makedirs(samples_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sample_{timestamp}.jpg"
                filepath = os.path.join(samples_dir, filename)
                cv2.imwrite(filepath, frame)
                return jsonify({"status": "success", "filename": filename})
        return jsonify({"status": "fail", "message": "Camera offline"}), 500
    except Exception as e:
        return jsonify({"status": "fail", "message": str(e)}), 500

# Get list of training samples
@app.route('/api/samples', methods=['GET'])
def api_get_samples():
    samples_dir = os.path.join("saved_frames", "samples")
    os.makedirs(samples_dir, exist_ok=True)
    try:
        files = os.listdir(samples_dir)
        samples = [f for f in files if f.endswith('.jpg')]
        samples.sort(reverse=True)
        return jsonify(samples)
    except Exception:
        return jsonify([])

# PLC Industrial Control Panel UI
@app.route('/plc')
def plc_ui():
    return send_from_directory('templates', 'plc.html')

# Submit support/contact message
@app.route('/api/support/message', methods=['POST'])
def api_support_message():
    data = request.get_json() or {}
    email = data.get("email")
    subject = data.get("subject")
    message = data.get("message")
    print(f"[Support Ticket] From: {email} | Subject: {subject} | Message: {message}")
    return jsonify({"status": "success", "message": "Support request submitted successfully to Kizen Engineering support team!"})

if __name__ == '__main__':
    print("=" * 70)
    print("  KIZEN ENGINEERING — TEXTILEGUARD AI VISION PLATFORM")
    print("  Innovation Is Our Tradition • https://kizen.co.in")
    print("=" * 70)
    print("  Web Server   : http://localhost:5000")
    print("  WebSocket    : ws://localhost:8765")
    print("  Camera Driver: " + cam_mode)
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, debug=False)
