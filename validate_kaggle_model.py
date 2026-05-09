# validate_kaggle_model.py
from ultralytics import YOLO
import os

def validate():
    model_path = 'runs/textile_detection/defect_model_kaggle/weights/best.pt'
    if not os.path.exists(model_path):
        print("Fine-tuned model not found.")
        return

    model = YOLO(model_path)
    
    # Run validation on test split
    print("Running validation on TEST split...")
    results = model.val(data='kaggle_data.yaml', split='test')
    
    print("\n" + "="*40)
    print("  VALIDATION RESULTS (REAL DATA)")
    print("="*40)
    print(f"  Precision: {results.results_dict['metrics/precision(B)']:.3f}")
    print(f"  Recall:    {results.results_dict['metrics/recall(B)']:.3f}")
    print(f"  mAP 50:    {results.results_dict['metrics/mAP50(B)']:.3f}")
    print("="*40)

if __name__ == "__main__":
    validate()
