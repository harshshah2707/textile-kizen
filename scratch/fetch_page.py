import requests

url = "https://kailashhambarde.github.io/Lusitano/"
response = requests.get(url)
print("Status Code:", response.status_code)
with open("scratch/page.html", "w", encoding="utf-8") as f:
    f.write(response.text)
print("Page length:", len(response.text))
