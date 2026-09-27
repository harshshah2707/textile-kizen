"""
Textile Defect Detection - Flask Web UI
=======================================
Web interface for uploading images, selecting validation presets,
and viewing dynamic, high-accuracy detection results.
"""

import os
import cv2
import numpy as np
import shutil
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from inference import TextileDefectDetector
from config import FLASK_CONFIG, UPLOADS_DIR, BEST_MODEL_PATH
from utils.bbox_refiner import refine_defect_bbox

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
detector = None

# BGR Color codes for overlays matching config and React UI
COLOR_MAP = {
    "Broken stitch": (82, 63, 244),   # Rose/Red
    "hole": (11, 158, 245),          # Orange/Amber
    "horizontal": (8, 179, 234),      # Yellow
    "lines": (247, 85, 168),          # Purple
    "Needle mark": (34, 197, 94),     # Green
    "Pinched fabric": (59, 130, 246),  # Blue
    "stain": (236, 72, 153),          # Pink
    "Vertical": (234, 179, 8),        # Cyan/Yellowish
}

def get_detector():
    global detector
    if detector is None:
        if os.path.exists(BEST_MODEL_PATH):
            detector = TextileDefectDetector(str(BEST_MODEL_PATH))
        else:
            # Fallback to pretrained if custom model doesn't exist yet
            detector = TextileDefectDetector("yolov8s.pt")
    return detector

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plc')
def plc_control():
    """Industrial PLC-type machine control panel UI."""
    return render_template('plc.html')

@app.route('/api/quick_test_images')
def quick_test_images():
    """Returns a list of validation images representing various fabric defect types."""
    presets = [
        {"name": "Hole defect (Low Contrast)", "file": "val_00000_hole_2018-10-11 13_47_44.290326.jpg", "class": "hole"},
        {"name": "Needle Mark Line", "file": "val_00002_A_08_008.jpg", "class": "Needle mark"},
        {"name": "Stain Spot", "file": "val_00003_294.jpg", "class": "stain"},
        {"name": "Needle Mark Line 2", "file": "val_00004_A_08_036.jpg", "class": "Needle mark"},
        {"name": "Broken Stitch Spot", "file": "val_00005_A_02_091.jpg", "class": "Broken stitch"},
        {"name": "Hole (High Contrast)", "file": "val_00006_42_processed (3).jpg", "class": "hole"},
        {"name": "Horizontal line defect", "file": "val_00007_line_2018-10-10 12_04_47.643010.jpg", "class": "lines"},
        {"name": "Defect-Free Fabric", "file": "val_00001_5bdc2db614dc354d0917204416.jpg", "class": "clean"}
    ]
    return jsonify(presets)

@app.route('/api/detect', methods=['POST'])
def detect_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Check for dynamic confidence parameter
    conf_val = request.form.get('conf') or request.args.get('conf')
    conf_threshold = float(conf_val) if conf_val else 0.25
    
    det = get_detector()
    results = det.model.predict(
        source=filepath,
        conf=conf_threshold,
        iou=0.45,
        device=det.config["device"]
    )
    
    # Process image for refined visualization
    img = cv2.imread(filepath)
    detections = []
    defect_count = 0
    class_names_map = det.model.names
    
    for r in results:
        for box in r.boxes:
            b = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Apply Bounding Box Refinement
            x1_r, y1_r, x2_r, y2_r = refine_defect_bbox(img, b, cls)
            
            dtype = class_names_map.get(cls, "defect")
            color = COLOR_MAP.get(dtype, (0, 0, 255))
            
            # Draw Refined Bounding Box
            cv2.rectangle(img, (x1_r, y1_r), (x2_r, y2_r), color, 2)
            
            w_box = x2_r - x1_r
            size_mm = round(w_box * 0.15, 1)
            
            label = f"{dtype} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1_r, y1_r - th - 5), (x1_r + tw, y1_r), color, -1)
            cv2.putText(img, label, (x1_r, y1_r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            defect_count += 1
            detections.append({
                "class": cls,
                "class_name": dtype,
                "conf": round(conf, 2),
                "bbox": [x1_r, y1_r, x2_r, y2_r],
                "size_mm": size_mm,
                "location": f"X:{x1_r} Y:{y1_r}"
            })
            
    # Status overlay
    status = "PASS" if defect_count == 0 else "FAIL"
    color_status = (0, 200, 0) if defect_count == 0 else (0, 0, 255)
    cv2.putText(img, f"STATUS: {status} ({defect_count} defects)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_status, 2)
    
    result_filename = "res_" + filename
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, img)
    
    return jsonify({
        "defect_count": defect_count,
        "status": status,
        "result_image": result_filename,
        "detections": detections
    })

@app.route('/api/quick_test/<filename>', methods=['POST'])
def quick_test(filename):
    val_dir = os.path.join("datasets", "multiclass_yolo", "images", "val")
    src_path = os.path.join(val_dir, filename)
    if not os.path.exists(src_path):
        return jsonify({"error": f"Sample image {filename} not found"}), 404
        
    dest_filename = secure_filename(filename)
    dest_path = os.path.join(app.config['UPLOAD_FOLDER'], dest_filename)
    shutil.copy(src_path, dest_path)
    
    # Get dynamic confidence parameter
    conf_val = request.form.get('conf') or request.args.get('conf')
    conf_threshold = float(conf_val) if conf_val else 0.25
    
    det = get_detector()
    results = det.model.predict(
        source=dest_path,
        conf=conf_threshold,
        iou=0.45,
        device=det.config["device"]
    )
    
    img = cv2.imread(dest_path)
    detections = []
    defect_count = 0
    class_names_map = det.model.names
    
    for r in results:
        for box in r.boxes:
            b = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Apply Bounding Box Refinement
            x1_r, y1_r, x2_r, y2_r = refine_defect_bbox(img, b, cls)
            
            dtype = class_names_map.get(cls, "defect")
            color = COLOR_MAP.get(dtype, (0, 0, 255))
            
            # Draw Refined Bounding Box
            cv2.rectangle(img, (x1_r, y1_r), (x2_r, y2_r), color, 2)
            
            w_box = x2_r - x1_r
            size_mm = round(w_box * 0.15, 1)
            
            label = f"{dtype} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1_r, y1_r - th - 5), (x1_r + tw, y1_r), color, -1)
            cv2.putText(img, label, (x1_r, y1_r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            defect_count += 1
            detections.append({
                "class": cls,
                "class_name": dtype,
                "conf": round(conf, 2),
                "bbox": [x1_r, y1_r, x2_r, y2_r],
                "size_mm": size_mm,
                "location": f"X:{x1_r} Y:{y1_r}"
            })
            
    # Status overlay
    status = "PASS" if defect_count == 0 else "FAIL"
    color_status = (0, 200, 0) if defect_count == 0 else (0, 0, 255)
    cv2.putText(img, f"STATUS: {status} ({defect_count} defects)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_status, 2)
    
    result_filename = "res_" + dest_filename
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, img)
    
    return jsonify({
        "defect_count": defect_count,
        "status": status,
        "result_image": result_filename,
        "detections": detections
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host=FLASK_CONFIG["host"], port=FLASK_CONFIG["port"], debug=FLASK_CONFIG["debug"])
