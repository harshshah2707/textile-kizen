"""
Textile Defect Detection - Flask Web UI
=======================================
Web interface for uploading images and viewing detection results.
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from inference import TextileDefectDetector
from config import FLASK_CONFIG, UPLOADS_DIR, BEST_MODEL_PATH

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
detector = None

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
    
    det = get_detector()
    results = det.detect(filepath)
    
    # Process image for visualization
    img = cv2.imread(filepath)
    img, count = det.visualize(img, results)
    
    result_filename = "res_" + filename
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, img)
    
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": int(box.cls[0]),
                "conf": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })
            
    return jsonify({
        "defect_count": count,
        "status": "PASS" if count == 0 else "FAIL",
        "result_image": result_filename,
        "detections": detections
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host=FLASK_CONFIG["host"], port=FLASK_CONFIG["port"], debug=FLASK_CONFIG["debug"])
