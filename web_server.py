import os
import time
import json
import random
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
from utils.camera_handler import CameraHandler
from utils.config_live import LIVE_CONFIG
from defect_tracker import DefectTracker
from utils.bbox_refiner import refine_defect_bbox
from utils import db_manager, pdf_generator

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
broadcasted_ids = set()

# Global settings updated by WebSocket from React UI
conf_threshold = 0.25  # Lowered default confidence threshold for maximum sensitivity
simulation_mode = False  # Global Loom Simulation Mode toggle


# -------------------------------------------------------------
# YOLOv8 Model Initialization & Fallback
# -------------------------------------------------------------
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[System] PyTorch Device selected: {device}")

# Attempt to load custom trained weights or fallback
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
            print(f"[System] Attempting to load YOLO model from: {path}")
            model = YOLO(path)
            print(f"[System] Model loaded successfully from {path}!")
            break
        except Exception as e:
            print(f"[System] Failed to load model {path}: {e}")

if model is None:
    try:
        print("[System] No local weights found. Downloading and loading pre-trained yolov8n.pt...")
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"[System] Critical: Failed to load any YOLO model: {e}")

model_lock = threading.Lock()

# -------------------------------------------------------------
# Camera & Tracker Configuration
# -------------------------------------------------------------
# Auto-detect MindVision Line-Scan Camera
has_mindvision = False
try:
    import mvsdk
    mv_devices = mvsdk.CameraEnumerateDevice()
    has_mindvision = len(mv_devices) > 0
except Exception as e:
    print(f"[System] Failed to import mvsdk or query devices: {e}")
    has_mindvision = False

if has_mindvision:
    print("[System] MindVision Line-Scan camera detected physically! Assigning to Camera 1.")
    cam1_id = 'mindvision'
else:
    print("[System] No MindVision camera found physically. Assigning Camera 1 to laptop webcam.")
    cam1_id = 0

print(f"[System] Initializing camera stream (Primary: {cam1_id})")
cam1 = CameraHandler(cam1_id, resolution=(640, 480))
cam1_opened = cam1.start()

print(f"[System] Camera 1 Status: {'ONLINE (MindVision)' if cam1_id == 'mindvision' and cam1_opened else ('ONLINE' if cam1_opened else 'OFFLINE')}")

# Initialize tracker for the camera
tracker1 = DefectTracker(max_distance=80, max_age=15)

# -------------------------------------------------------------
# Raw WebSocket Server & Frame Decoder
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
    print(f"[WebSocket] Connected client from {client_address}")
    global conf_threshold, simulation_mode, broadcasted_ids
    try:
        request = client_socket.recv(4096).decode('utf-8', errors='ignore')
        if "Upgrade: websocket" not in request:
            client_socket.close()
            return
            
        ws_key = None
        for line in request.split('\r\n'):
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
            
        init_frame = make_ws_frame(json.dumps({"type": "heartbeat", "cpu": 20, "gpu": 30, "temp": 48, "bandwidth": 8.0}))
        client_socket.send(init_frame)
        
        # Send current simulation state to client on connection
        sim_state_frame = make_ws_frame(json.dumps({"type": "simulation_state", "enabled": simulation_mode}))
        client_socket.send(sim_state_frame)
        
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
                            print(f"[YOLO Inference] Dynamic Confidence Threshold updated to: {conf_threshold:.2f}")
                        
                        new_speed = msg.get("fabricSpeed")
                        if new_speed is not None:
                            global fabric_speed_m_per_min
                            fabric_speed_m_per_min = float(new_speed)

                        new_width = msg.get("fabricWidth")
                        if new_width is not None:
                            global fabric_width_mm
                            fabric_width_mm = float(new_width)

                        cam_settings = msg.get("cameraControls", {}).get("cam1", {})
                        if cam_settings:
                            exp = cam_settings.get("exposure")
                            gain = cam_settings.get("gain")
                            sh = cam_settings.get("slice_height")
                            enh = cam_settings.get("enhance")
                            if exp is not None:
                                cam1.set_exposure(exp)
                            if gain is not None:
                                cam1.set_gain(gain)
                            if sh is not None:
                                cam1.set_slice_height(sh)
                            if enh is not None:
                                cam1.set_enhance(enh)
                    elif msg.get("type") == "toggle_simulation":
                        simulation_mode = bool(msg.get("enabled", False))
                        print(f"[WebSocket] Simulation Mode toggled: {simulation_mode}")
                        # Reset broadcasted defect IDs on state transition to allow re-detecting during simulation
                        broadcasted_ids.clear()
                        ws_broadcast({"type": "simulation_state", "enabled": simulation_mode})
                except Exception as e:
                    pass
                    
    except Exception as e:
        print(f"[WebSocket] Client connection error: {e}")
    finally:
        with clients_lock:
            if client_socket in websocket_clients:
                websocket_clients.remove(client_socket)
        try:
            client_socket.close()
        except:
            pass
        print(f"[WebSocket] Disconnected client from {client_address}")

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
                except:
                    pass

def run_websocket_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(('0.0.0.0', 8765))
        server.listen(10)
        print("[WebSocket Server] Running raw WS on port 8765...")
    except Exception as e:
        print(f"[WebSocket Server] Failed to bind to 8765: {e}")
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
# Background Telemetry Broadcast Loop
# -------------------------------------------------------------
def background_telemetry():
    while True:
        try:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            disk = psutil.disk_usage(os.path.abspath('/')).percent
            
            ws_broadcast({
                "type": "heartbeat",
                "cpu": cpu,
                "ram": ram,
                "disk": disk
            })
        except Exception as e:
            pass
        time.sleep(1.0)

threading.Thread(target=background_telemetry, daemon=True).start()

# -------------------------------------------------------------
# Live Model Inference and Video Streaming Loop
# -------------------------------------------------------------
def generate_fabric_frames(camera_id):
    width, height = 640, 480
    # Always route to camera 1 (MindVision line-scan) as Camera 2 is removed
    cam_handler = cam1
    is_opened = cam1_opened
    tracker = tracker1
    
    global conf_threshold, simulation_mode
    
    last_time = time.time()
    frame_count = 0
    
    # Class IDs to label names mapping (from config.py) dynamically loaded or default fallback
    class_names_map = model.names if (model is not None and hasattr(model, 'names')) else {
        0: "Broken stitch",
        1: "hole",
        2: "horizontal",
        3: "lines",
        4: "Needle mark",
        5: "Pinched fabric",
        6: "stain",
        7: "Vertical"
    }

    # BGR Color codes for overlays
    color_map = {
        "Broken stitch": (82, 63, 244),   # Rose/Red
        "hole": (11, 158, 245),          # Orange/Amber
        "horizontal": (8, 179, 234),      # Yellow
        "lines": (247, 85, 168),          # Purple
        "Needle mark": (34, 197, 94),     # Green
        "Pinched fabric": (59, 130, 246),  # Blue
        "stain": (236, 72, 153),          # Pink
        "Vertical": (234, 179, 8),        # Cyan/Yellowish
    }

    # Pre-build a list of validation images that contain defects
    val_img_dir = os.path.join("datasets", "multiclass_yolo", "images", "val")
    val_lbl_dir = os.path.join("datasets", "multiclass_yolo", "labels", "val")
    defect_images = []
    if os.path.exists(val_img_dir) and os.path.exists(val_lbl_dir):
        try:
            for f in sorted(os.listdir(val_img_dir)):
                if f.endswith('.jpg'):
                    lbl_name = f.replace('.jpg', '.txt')
                    lbl_path = os.path.join(val_lbl_dir, lbl_name)
                    # Check if the label file exists and is not empty (contains defect coordinates)
                    if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
                        defect_images.append(os.path.join(val_img_dir, f))
        except Exception as e:
            print(f"[Simulation Setup] Error listing validation images: {e}")
            
    # Add some background (defect free) images to make the simulation feel like a real loom roll
    bg_images = []
    if os.path.exists(val_img_dir) and os.path.exists(val_lbl_dir):
        try:
            for f in sorted(os.listdir(val_img_dir)):
                if f.endswith('.jpg'):
                    lbl_name = f.replace('.jpg', '.txt')
                    lbl_path = os.path.join(val_lbl_dir, lbl_name)
                    if os.path.exists(lbl_path) and os.path.getsize(lbl_path) == 0:
                        bg_images.append(os.path.join(val_img_dir, f))
        except Exception as e:
            pass

    # Create a realistic interleaved sequence: [BG, Defect, BG, BG, Defect, BG, ...]
    sim_sequence = []
    for idx, def_img in enumerate(defect_images):
        # Interleave 1-2 background images between each defect image
        if bg_images:
            sim_sequence.append(bg_images[(idx * 2) % len(bg_images)])
            if len(bg_images) > 1:
                sim_sequence.append(bg_images[(idx * 2 + 1) % len(bg_images)])
        sim_sequence.append(def_img)
        
    if not sim_sequence:
        sim_sequence = defect_images if defect_images else bg_images

    sim_idx = 0
    last_sim_time = 0.0

    while True:
        time.sleep(0.04)  # ~25 FPS loop rate
        
        frame = None
        
        # 1. Image source logic
        if simulation_mode:
            now = time.time()
            # Cycle to the next image in the sequence every 4 seconds
            if now - last_sim_time >= 4.0:
                if sim_sequence:
                    sim_idx = (sim_idx + 1) % len(sim_sequence)
                last_sim_time = now
                
            if sim_sequence:
                img_path = sim_sequence[sim_idx]
                img = cv2.imread(img_path)
                if img is not None:
                    frame = cv2.resize(img, (width, height))
            
            if frame is None:
                # Fallback if image failed to load
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:] = (235, 230, 225) if camera_id == 1 else (238, 233, 228)
        else:
            # Normal webcam feed mode
            if is_opened:
                success, img = cam_handler.read()
                if success and img is not None:
                    frame = cv2.resize(img, (width, height))
                    
        # Apply visual fallback if no frame available
        if frame is None:
            # Display scroll texture fallback if camera offline (light blue-grey theme)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (235, 230, 225) if camera_id == 1 else (238, 233, 228)
            for x in range(0, width, 6):
                cv2.line(frame, (x, 0), (x, height), (242, 238, 235), 1)
            scroll_y = int(time.time() * 100) % 8
            for y in range(scroll_y, height, 8):
                cv2.line(frame, (0, y), (width, y), (228, 222, 218), 1)

        # Run real-time YOLOv8 model inference on frames
        # In simulation mode, we run on all frames (to simulate continuous feed)
        if model is not None and frame is not None and (simulation_mode or is_opened):
            try:
                # Predict at imgsz=320 for maximum CPU FPS
                with model_lock:
                    results = model.predict(
                        source=frame,
                        conf=conf_threshold,
                        imgsz=320,
                        device=device,
                        verbose=False
                    )
                
                rects, confs, clss = [], [], []
                for r in results:
                    for box in r.boxes:
                        b = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        # Apply Bounding Box Refinement to shrink giant fallback boxes
                        x1_r, y1_r, x2_r, y2_r = refine_defect_bbox(frame, b, cls)
                        
                        rects.append([x1_r, y1_r, x2_r, y2_r])
                        confs.append(conf)
                        clss.append(cls)
                
                # Update Tracker
                tracker.update(rects, confs, clss)
                active_defects_list = tracker.get_active_defects(min_hits=1)
                
                # Broadcast and Annotate Frame
                for def_item in active_defects_list:
                    def_id = f"{camera_id}-{def_item['id']}"
                    dtype = class_names_map.get(def_item['class'], "defect")
                    color = color_map.get(dtype, (0, 0, 255))
                    
                    x1, y1, x2, y2 = def_item['bbox']
                    w_box = x2 - x1
                    h_box = y2 - y1
                    
                    # Convert bounding box pixels to approximate mm
                    size_mm = round(w_box * 0.15, 1)
                    
                    # Draw box and class label on MJPEG frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, 
                        f"ID: {def_item['id']} {dtype.upper()} {def_item['conf']:.2f}",
                        (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        cv2.LINE_AA
                    )
                    
                    # Broadcast defect coordinate payload to websocket clients on first registration
                    if def_id not in broadcasted_ids:
                        broadcasted_ids.add(def_id)
                        
                        # Calculate distance along the roll
                        distance_meters = 0.0
                        if current_roll_id is not None:
                            elapsed = time.time() - roll_start_time
                            distance_meters = round((fabric_speed_m_per_min / 60) * elapsed, 2)
                        
                        # Save defect crop
                        crop_url = ""
                        if current_roll_id is not None and frame is not None:
                            try:
                                crop_dir = os.path.join("saved_frames", "crops", f"roll_{current_roll_id}")
                                os.makedirs(crop_dir, exist_ok=True)
                                crop_filename = f"defect_{def_item['id']}_{int(time.time())}.jpg"
                                crop_path = os.path.join(crop_dir, crop_filename)
                                
                                fy, fx = frame.shape[:2]
                                cy1, cy2 = max(0, y1), min(fy, y2)
                                cx1, cx2 = max(0, x1), min(fx, x2)
                                if cy2 > cy1 and cx2 > cx1:
                                    crop_img = frame[cy1:cy2, cx1:cx2]
                                    cv2.imwrite(crop_path, crop_img)
                                    crop_url = f"/crops/roll_{current_roll_id}/{crop_filename}"
                            except Exception as e:
                                print(f"[Crop Error] {e}")
                                
                        # Log to database
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
                            except Exception as e:
                                print(f"[DB Log Error] {e}")

                        defect_payload = {
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
                        }
                        ws_broadcast(defect_payload)
                        
            except Exception as e:
                print(f"[Inference Error] {e}")

        # Calculate actual FPS and broadcast metrics
        frame_count += 1
        curr_time = time.time()
        if curr_time - last_time >= 1.0:
            fps_val = frame_count / (curr_time - last_time)
            ws_broadcast({"type": "frame_metrics", "fps": round(fps_val, 1)})
            frame_count = 0
            last_time = curr_time

        # Encode to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# -------------------------------------------------------------
# Flask Server Routes & Static Serving
# -------------------------------------------------------------
@app.route('/')
def serve_index():
    if os.path.exists('frontend/dist/index.html'):
        return send_from_directory('frontend/dist', 'index.html')
    else:
        return "Frontend compiled bundle not found. Please compile frontend folder with npm run build first.", 404

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join('frontend/dist', path)):
        return send_from_directory('frontend/dist', path)
    return send_from_directory('frontend/dist', 'index.html')

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    if camera_id != 1:
        return "Invalid camera ID", 400
    return Response(
        generate_fabric_frames(camera_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/stats')
def api_stats():
    return {
        "status": "online",
        "model": "yolov8s-textile",
        "cpu_usage_percent": psutil.cpu_percent(),
        "ram_usage_percent": psutil.virtual_memory().percent
    }

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
        # Get frame from standard webcam/camera 1
        cam_handler = camera_handlers.get(1)
        if cam_handler:
            frame = cam_handler.read()
            if frame is not None:
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
    except Exception as e:
        return jsonify([])

# Submit support/contact message
@app.route('/api/support/message', methods=['POST'])
def api_support_message():
    data = request.get_json() or {}
    email = data.get("email")
    subject = data.get("subject")
    message = data.get("message")
    print(f"[Support Ticket] From: {email} | Subject: {subject} | Message: {message}")
    return jsonify({"status": "success", "message": "Support request submitted successfully!"})

if __name__ == '__main__':
    print("=" * 60)
    print("  TEXTILEGUARD AI - WEB SERVER DASHBOARD")
    print("=" * 60)
    print("  Flask Host   : http://localhost:5000")
    print("  WebSocket    : ws://localhost:8765")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
