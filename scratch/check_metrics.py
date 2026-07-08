import os
import pandas as pd

csv_path = "runs/textile_detection/lusitano_finetune/results.csv"
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        # Clean column names (strip spaces)
        df.columns = [c.strip() for c in df.columns]
        print(f"Training history ({len(df)} epochs completed):")
        # Print epoch, box loss, cls loss, precision, recall, mAP50, mAP50-95
        cols = ['epoch', 'train/box_loss', 'train/cls_loss', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
        cols = [c for c in cols if c in df.columns]
        print(df[cols].tail(10).to_string(index=False))
    except Exception as e:
        print("Error reading CSV:", e)
else:
    print("results.csv not found yet.")
