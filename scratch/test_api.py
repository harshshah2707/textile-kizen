import requests
import json

url = 'http://127.0.0.1:5001/api/quick_test/val_00001_5bdc2db614dc354d0917204416.jpg?conf=0.25'

try:
    r = requests.post(url)
    print(f"Status Code: {r.status_code}")
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print(f"Error calling API: {e}")
