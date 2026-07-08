import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from remotezip import RemoteZip

# Configuration
ZIP_URL = "https://www.it.ubi.pt/DetReIDX_dataset/lusitano/test.zip"
RAW_OUT_DIR = "scratch/lusitano_raw"
DEFECTS_DIR = os.path.join(RAW_OUT_DIR, "defects")
NORMALS_DIR = os.path.join(RAW_OUT_DIR, "non-defects")

NUM_DEFECT_IMAGES = 100
NUM_NORMAL_IMAGES = 50
MAX_WORKERS = 8

def main():
    print(f"[Ingest] Initializing Lusitano dataset downloader...")
    os.makedirs(DEFECTS_DIR, exist_ok=True)
    os.makedirs(NORMALS_DIR, exist_ok=True)
    
    t_start = time.time()
    
    print("[Ingest] Listing files in remote zip...")
    with RemoteZip(ZIP_URL) as rz:
        namelist = rz.namelist()
        
        # Filter only JPG files to keep file format uniform
        defect_candidates = [n for n in namelist if n.startswith("test/defects/") and n.endswith(".jpg")]
        normal_candidates = [n for n in namelist if n.startswith("test/non-defects/") and n.endswith(".jpg")]
        
        print(f"[Ingest] Found {len(defect_candidates)} defect JPGs and {len(normal_candidates)} normal JPGs in remote zip.")
        
        # Systematically sample to get a diverse set across the entire timeline
        random.seed(42)  # For reproducibility
        defect_candidates.sort()
        normal_candidates.sort()
        
        # Select systematically spaced indices
        def select_subset(candidates, target_count):
            if len(candidates) <= target_count:
                return candidates
            step = len(candidates) / target_count
            return [candidates[int(i * step)] for i in range(target_count)]
            
        selected_defects = select_subset(defect_candidates, NUM_DEFECT_IMAGES)
        selected_normals = select_subset(normal_candidates, NUM_NORMAL_IMAGES)
        
        all_downloads = []
        for path in selected_defects:
            all_downloads.append((path, DEFECTS_DIR))
        for path in selected_normals:
            all_downloads.append((path, NORMALS_DIR))
            
        print(f"[Ingest] Selected {len(selected_defects)} defects and {len(selected_normals)} normal images for download.")
        
    # Helper download function for executor
    def download_image(args):
        remote_path, target_dir = args
        basename = os.path.basename(remote_path)
        dest_path = os.path.join(target_dir, basename)
        
        # Skip if already downloaded
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000000:
            return remote_path, True, 0
            
        max_retries = 3
        for attempt in range(max_retries):
            try:
                t0 = time.time()
                with RemoteZip(ZIP_URL) as rz_thread:
                    data = rz_thread.read(remote_path)
                with open(dest_path, "wb") as f:
                    f.write(data)
                duration = time.time() - t0
                size_mb = len(data) / (1024 * 1024)
                return remote_path, True, len(data)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[Error] Failed to download {basename} after {max_retries} attempts: {e}")
                    return remote_path, False, 0
                time.sleep(1)
                
    print(f"[Ingest] Starting parallel download using {MAX_WORKERS} workers...")
    total_bytes = 0
    successful_downloads = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_image, item): item for item in all_downloads}
        
        for future in as_completed(futures):
            remote_path, success, bytes_written = future.result()
            if success:
                successful_downloads += 1
                total_bytes += bytes_written
                if successful_downloads % 10 == 0 or bytes_written > 0:
                    mb_downloaded = total_bytes / (1024 * 1024)
                    print(f"Progress: {successful_downloads}/{len(all_downloads)} images completed. Total downloaded: {mb_downloaded:.1f} MB")
            
    t_end = time.time()
    duration = t_end - t_start
    total_mb = total_bytes / (1024 * 1024)
    speed = total_mb / duration if duration > 0 else 0
    
    print(f"\n[Ingest] Download session complete!")
    print(f"  Successfully downloaded: {successful_downloads} / {len(all_downloads)} files.")
    print(f"  Total Data: {total_mb:.2f} MB")
    print(f"  Total Duration: {duration:.1f} seconds")
    print(f"  Average Throughput Speed: {speed:.2f} MB/s")
    print(f"  Raw images stored in: {RAW_OUT_DIR}")

if __name__ == "__main__":
    main()
