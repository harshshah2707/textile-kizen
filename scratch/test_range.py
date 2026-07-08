import requests

url = "https://www.it.ubi.pt/DetReIDX_dataset/lusitano/test.zip"
headers = {"Range": "bytes=0-100"}
response = requests.get(url, headers=headers, timeout=10)
print("Status Code:", response.status_code)
print("Headers:")
for k, v in response.headers.items():
    print(f"  {k}: {v}")
