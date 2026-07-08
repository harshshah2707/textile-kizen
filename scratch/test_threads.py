import time
import os
from concurrent.futures import ThreadPoolExecutor
from remotezip import RemoteZip

url = "https://www.it.ubi.pt/DetReIDX_dataset/lusitano/test.zip"
os.makedirs("scratch/thread_test", exist_ok=True)

def download_file(rz_info, name):
    t0 = time.time()
    data = rz_info.read(name)
    filename = os.path.join("scratch/thread_test", os.path.basename(name))
    with open(filename, 'wb') as f:
        f.write(data)
    t1 = time.time()
    size_mb = len(data) / (1024 * 1024)
    speed = size_mb / (t1 - t0)
    print(f"Downloaded {os.path.basename(name)} ({size_mb:.2f} MB) in {t1-t0:.2f}s ({speed:.2f} MB/s)")
    return len(data)

with RemoteZip(url) as rz:
    namelist = rz.namelist()
    jpg_defects = [n for n in namelist if n.startswith('test/defects/') and n.endswith('.jpg')][:10]
    
    print(f"Starting parallel download of {len(jpg_defects)} files...")
    t_start = time.time()
    
    # We open a new RemoteZip inside each thread, or use a thread-safe connection pool.
    # Note: RemoteZip instances are not thread-safe if they share the same socket,
    # so we should open RemoteZip inside the thread or just fetch the bytes directly using range requests!
    # Let's fetch using requests range requests directly in threads, which is 100% thread-safe.
    # We can get the file offsets from RemoteZip members.
    members = {m.filename: m for m in rz.infolist()}
    
    def download_by_range(name):
        member = members[name]
        # RemoteZip uses zipfile info which has header_offset.
        # But wait, extracting from zip using range requests directly requires parsing zip headers.
        # It's simpler to just create a RemoteZip instance in each thread!
        t0 = time.time()
        with RemoteZip(url) as thread_rz:
            data = thread_rz.read(name)
        filename = os.path.join("scratch/thread_test", os.path.basename(name))
        with open(filename, 'wb') as f:
            f.write(data)
        t1 = time.time()
        size_mb = len(data) / (1024 * 1024)
        print(f"Thread downloaded {os.path.basename(name)} ({size_mb:.2f} MB) in {t1-t0:.2f}s")
        return len(data)

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(download_by_range, jpg_defects))
        
    t_end = time.time()
    total_bytes = sum(results)
    total_mb = total_bytes / (1024 * 1024)
    avg_speed = total_mb / (t_end - t_start)
    print(f"\nTotal: {total_mb:.2f} MB downloaded in {t_end - t_start:.2f} seconds.")
    print(f"Average Speed: {avg_speed:.2f} MB/s")
