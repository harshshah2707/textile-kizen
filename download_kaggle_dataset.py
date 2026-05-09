import os
import subprocess
from pathlib import Path

# Dataset owner and name
DATASET_ID = "belkhirnacim/textiledefectdetection"
DEST_PATH = "datasets/kaggle_textile"

def download_and_extract():
    # Check for credentials
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    
    if not username or not key:
        print("ERROR: Kaggle credentials not found!")
        print("Please set them in your terminal before running:")
        print('  $env:KAGGLE_USERNAME="your_username"')
        print('  $env:KAGGLE_KEY="your_api_key"')
        return

    print(f"Credentials found for user: {username}")
    
    # Ensure destination exists
    Path(DEST_PATH).mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset {DATASET_ID}...")
    try:
        # Use subprocess to run kaggle command
        subprocess.run([
            "venv/Scripts/kaggle.exe", 
            "datasets", 
            "download", 
            "-d", DATASET_ID, 
            "-p", DEST_PATH, 
            "--unzip"
        ], check=True)
        
        print(f"\nSuccess! Dataset downloaded and extracted to {DEST_PATH}")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: Failed to download dataset. Check your credentials or internet connection.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    download_and_extract()
