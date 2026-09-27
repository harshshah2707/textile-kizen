"""
check_status.py
================
Quick CLI utility to display the current pipeline execution status,
GPU memory usage, active stage, and the latest training logs.
"""

import os
import sys
import json
import time
from pathlib import Path

# Ensure UTF-8 output encoding for Windows PowerShell / CMD
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).parent.resolve()
STATUS_FILE = BASE_DIR / "outputs" / "pipeline_live_status.json"
LOG_FILE = BASE_DIR / "outputs" / "pipeline.log"

def main():
    if not STATUS_FILE.exists():
        print("[Pipeline Status] No active pipeline status found (pipeline_live_status.json does not exist yet).")
        return

    try:
        data = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Pipeline Status] Error reading status file: {e}")
        return

    print("=" * 72)
    print("           TEXTILE AI PIPELINE: LIVE MONITOR")
    print("=" * 72)
    print(f"  Overall Status : {data.get('status', 'UNKNOWN')}")
    print(f"  Current Stage  : [{data.get('current_stage_idx', 0)}/{data.get('total_stages', 4)}] {data.get('current_stage_name', 'N/A')}")
    print(f"  Total Elapsed  : {data.get('elapsed_total_min', 0.0):.1f} minutes")
    print(f"  Last Heartbeat : {data.get('last_updated', 'N/A')}")

    completed = data.get("completed_stages", [])
    if completed:
        print("\n  --- Completed Stages ---")
        for s in completed:
            print(f"   ✓ Stage {s.get('stage_idx')}: {s.get('name')} ({s.get('runtime_mins', 0)}m)")

    recent = data.get("recent_logs", [])
    if recent:
        print("\n  --- Recent Live Logs ---")
        for line in recent[-8:]:
            print(f"    {line}")
    print("=" * 72)

if __name__ == "__main__":
    main()
