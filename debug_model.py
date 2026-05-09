import cv2
from ultralytics import YOLO
from utils.config_live import LIVE_CONFIG

def debug_live():
    # Load the NEW PRO model
    model_path = 'runs/textile_detection/defect_model_pro_v1/weights/best.pt'
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(LIVE_CONFIG['camera_id'])
    print("DEBUG MODE: Showing RAW detections (No filters, No tracking)")
    print("Press 'Q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Run RAW inference with a very low threshold (0.1)
        results = model.predict(frame, conf=0.1, verbose=False)[0]
        
        # Draw results
        annotated_frame = results.plot()
        
        cv2.imshow("RAW YOLO DEBUG", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    debug_live()
