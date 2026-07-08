import os
import pandas as pd

csv_path = "runs/textile_detection/lusitano_finetune/results.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=['train/box_loss'])
    print(f"Clean training history ({len(df)} valid epochs):")
    cols = ['epoch', 'train/box_loss', 'train/cls_loss', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))
else:
    print("CSV not found.")
