import os

path = r"C:\Users\Harsh\.gemini\antigravity\brain\f540abdb-0b87-4878-8cd0-7c56d09c6ed9\.system_generated\steps\1602\content.md"
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    print("Length of content:", len(content))
    # Print the lines containing defect descriptions or categories
    lines = content.split('\n')
    for i in range(100, min(len(lines), 350)):
        print(f"{i}: {lines[i]}")
else:
    print("File not found")
