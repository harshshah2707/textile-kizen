# debug_detection.py
import cv2
import numpy as np
from ultralytics import YOLO
from textile_detector import TextileDetector
from defect_validator import DefectValidator
from false_positive_filter import FalsePositiveFilter
from utils.config_live import LIVE_CONFIG
from config import BEST_MODEL_PATH, DATASET_DIR

def run_debug():
    print("="*60)
    print("  TEXTILE DETECTION DIAGNOSTIC TOOL")
    print("="*60)
    
    # Check model
    if not BEST_MODEL_PATH.exists():
        print(f"ERROR: Model not found at {BEST_MODEL_PATH}")
        return

    model = YOLO(str(BEST_MODEL_PATH))
    textile_det = TextileDetector()
    validator = DefectValidator()
    fp_filter = FalsePositiveFilter()

    # Find a test image (validation set)
    val_images = list((DATASET_DIR / "images" / "val").glob("*.jpg"))
    if not val_images:
        print("ERROR: No test images found in datasets/synthetic/images/val")
        return
    
    test_img_path = str(val_images[0])
    print(f"Loading test image: {test_img_path}")
    frame = cv2.imread(test_img_path)
    if frame is None:
        print("ERROR: Failed to load image.")
        return

    # TEST 1: Raw Model
    print("\n--- TEST 1: Raw YOLOv8 Output (Conf=0.25) ---")
    raw_results = model.predict(source=frame, conf=0.25, verbose=False)
    raw_count = len(raw_results[0].boxes)
    print(f"Raw detections found: {raw_count}")
    for i, box in enumerate(raw_results[0].boxes):
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        print(f"  [{i}] Class: {cls}, Conf: {conf:.2f}")

    # TEST 2: Textile Detector
    print("\n--- TEST 2: Textile Detection Filter ---")
    textile_res = textile_det.is_textile(frame)
    print(f"Is Textile: {textile_res['is_textile']}")
    print(f"Confidence: {textile_res['confidence']:.2f}")
    print(f"Texture Score: {textile_res['texture_score']:.2f}")

    # TEST 3: Validation & Filtering
    print("\n--- TEST 3: Validation Pipeline (Conf=0.5) ---")
    processed_count = 0
    if raw_count > 0:
        roi_mask = textile_res['roi_mask']
        for i, box in enumerate(raw_results[0].boxes):
            b = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            # Validation
            val = validator.validate_defect(b, frame, roi_mask if roi_mask is not None else np.ones(frame.shape[:2], dtype=np.uint8)*255)
            # FP Filter
            fp = fp_filter.filter_detections(b, frame)
            
            print(f"  [{i}] Conf: {conf:.2f}")
            print(f"      Validation: {val['is_valid']} (Reason: {val.get('reason', 'N/A')})")
            print(f"      FP Filter:  {fp['is_valid']} (Reason: {fp.get('reason', 'N/A')})")
            
            if conf >= 0.5 and val['is_valid'] and fp['is_valid']:
                processed_count += 1
    
    print(f"\nFinal valid detections after all filters: {processed_count}")
    
    print("\n" + "="*60)
    if processed_count == 0 and raw_count > 0:
        print("SUMMARY: Filters are BLOCKING detections. Use live_detection_fixed.py")
    elif raw_count == 0:
        print("SUMMARY: Model is not detecting anything. Retrain requested.")
    else:
        print("SUMMARY: Detection system is healthy.")
    print("="*60)

if __name__ == "__main__":
    run_debug()
