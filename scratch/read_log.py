import os

log_dir = "C:/Users/Harsh/.gemini/antigravity/brain/f540abdb-0b87-4878-8cd0-7c56d09c6ed9/.system_generated/tasks"
# Find the actual log file for task-671
log_files = [f for f in os.listdir(log_dir) if "task-671" in f]
print("Log files found:", log_files)

if log_files:
    log_path = os.path.join(log_dir, log_files[0])
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        print(f"\n--- First 40 lines of {log_files[0]} ---")
        for line in lines[:40]:
            print(line, end="")
            
        print(f"\n--- Last 30 lines of {log_files[0]} ---")
        for line in lines[-30:]:
            print(line, end="")
else:
    print("No log file found.")
