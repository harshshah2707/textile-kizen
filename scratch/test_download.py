import time
import requests

url = "https://www.it.ubi.pt/DetReIDX_dataset/lusitano/test.zip"
print("Testing download speed from", url)

start_time = time.time()
try:
    # Fetch headers first to verify connection and file size
    response = requests.head(url, timeout=10)
    print("Headers status code:", response.status_code)
    print("Content-Length:", response.headers.get("Content-Length"))
    
    # Download 10MB chunk
    headers = {"Range": "bytes=0-10485760"} # 10 MB
    t0 = time.time()
    res = requests.get(url, headers=headers, timeout=15)
    t1 = time.time()
    
    if res.status_code in [200, 206]:
        data_len = len(res.content)
        duration = t1 - t0
        speed_mb = (data_len / (1024 * 1024)) / duration
        print(f"Downloaded {data_len} bytes in {duration:.2f} seconds.")
        print(f"Speed: {speed_mb:.2f} MB/s")
    else:
        print("Status code:", res.status_code)
except Exception as e:
    print("Error:", e)
