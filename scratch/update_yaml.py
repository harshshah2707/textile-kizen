import os
import yaml

yaml_path = "datasets/lusitano_yolo/lusitano_data.yaml"
if os.path.exists(yaml_path):
    with open(yaml_path, 'r') as f:
        content = yaml.safe_load(f)
        
    content["nc"] = 5
    content["names"] = ["hole", "stain", "lines", "Needle mark", "Pinched fabric"]
    
    with open(yaml_path, 'w') as f:
        yaml.dump(content, f, default_flow_style=False)
    print("lusitano_data.yaml successfully updated to 5 classes!")
else:
    print("Error: lusitano_data.yaml not found.")
