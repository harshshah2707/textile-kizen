"""
run_complete_pipeline.py
========================
Automated End-to-End Execution Pipeline with Live Status Tracking:
  Stage A: Base Training & Fine-Tuning (train_best_model.py)
  Stage B: 800px High-Resolution Fine-Tuning (finetune_800px.py)
  Stage C: Model Evaluation & Performance Reporting (evaluate_model.py)
  Stage D: Line-Scan Hardware Optimization & Export (export_linescan_engine.py)

Tracks live progress in:
  - Real-time console output (unbuffered)
  - outputs/pipeline_live_status.json (for live dashboard/UI or polling)
  - outputs/pipeline.log (complete historical run log)
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Ensure UTF-8 output encoding for Windows PowerShell / CMD
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PYTHON_EXEC = sys.executable
BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATUS_FILE = OUTPUT_DIR / "pipeline_live_status.json"
LOG_FILE = OUTPUT_DIR / "pipeline.log"

# Global pipeline state
pipeline_state = {
    "status": "INITIALIZING",
    "start_time": datetime.now().isoformat(),
    "current_stage_idx": 0,
    "total_stages": 4,
    "current_stage_name": "Pending",
    "elapsed_total_min": 0.0,
    "completed_stages": [],
    "recent_logs": [],
    "last_updated": datetime.now().isoformat()
}

def update_status_file():
    try:
        pipeline_state["last_updated"] = datetime.now().isoformat()
        STATUS_FILE.write_text(json.dumps(pipeline_state, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        pass

def log(msg, also_stdout=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    if also_stdout:
        print(formatted, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass
    pipeline_state["recent_logs"].append(formatted)
    if len(pipeline_state["recent_logs"]) > 25:
        pipeline_state["recent_logs"] = pipeline_state["recent_logs"][-25:]
    update_status_file()

def run_step(stage_idx, stage_name, cmd_args, total_start):
    pipeline_state["status"] = "RUNNING"
    pipeline_state["current_stage_idx"] = stage_idx
    pipeline_state["current_stage_name"] = stage_name
    update_status_file()

    print("\n" + "=" * 76, flush=True)
    log(f">>> [STAGE {stage_idx}/4 STARTING]: {stage_name}")
    log(f"    Command: {' '.join(cmd_args)}")
    print("=" * 76 + "\n", flush=True)
    t0 = time.time()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Stream stdout and stderr live with real-time flushing
    proc = subprocess.Popen(
        cmd_args,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        encoding="utf-8",
        errors="replace"
    )

    last_status_write = time.time()
    for line in iter(proc.stdout.readline, ''):
        stripped = line.rstrip("\r\n")
        if stripped:
            print(f"  {stripped}", flush=True)
            try:
                with open(LOG_FILE, "a", encoding="utf-8") as lf:
                    lf.write(f"  {stripped}\n")
            except Exception:
                pass
            pipeline_state["recent_logs"].append(stripped)
            if len(pipeline_state["recent_logs"]) > 25:
                pipeline_state["recent_logs"] = pipeline_state["recent_logs"][-25:]
        
        now = time.time()
        if now - last_status_write >= 3.0:
            pipeline_state["elapsed_total_min"] = round((now - total_start) / 60, 2)
            update_status_file()
            last_status_write = now

    proc.wait()
    ret = proc.returncode

    elapsed = time.time() - t0
    mins = elapsed / 60
    pipeline_state["elapsed_total_min"] = round((time.time() - total_start) / 60, 2)

    if ret != 0:
        log(f"❌ [FAILED] {stage_name} exited with code {ret} after {mins:.1f} mins!")
        pipeline_state["status"] = "FAILED"
        pipeline_state["error"] = f"Stage {stage_idx} failed with returncode {ret}"
        update_status_file()
        sys.exit(ret)
    else:
        log(f"✅ [SUCCESS] {stage_name} completed in {mins:.1f} mins.")
        pipeline_state["completed_stages"].append({
            "stage_idx": stage_idx,
            "name": stage_name,
            "runtime_mins": round(mins, 2),
            "status": "COMPLETED"
        })
        update_status_file()

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== TEXTILE DEFECT DETECTION PIPELINE RUN LOG ({datetime.now().isoformat()}) ===\n")

    print("\n" + "#" * 76, flush=True)
    log("🚀 TEXTILE DEFECT DETECTION: STARTING END-TO-END PIPELINE")
    log("   Stages: A (Base Train) -> B (800px Boost) -> C (Eval) -> D (ONNX Export)")
    log(f"   Status file: {STATUS_FILE}")
    log(f"   Log file   : {LOG_FILE}")
    print("#" * 76 + "\n", flush=True)

    total_start = time.time()

    # Stage A: Base Training (150 epochs) + Phase 2 Fine-Tuning (30 epochs)
    stage_a_prod_w = BASE_DIR / "runs" / "textile_detection" / "defect_model_pro_v1" / "weights" / "best.pt"
    stage_a_summary = BASE_DIR / "runs" / "detect" / "runs" / "textile_detection" / "best_model_v1" / "training_summary.json"
    if not stage_a_summary.exists():
        stage_a_summary = BASE_DIR / "runs" / "textile_detection" / "best_model_v1" / "training_summary.json"

    if stage_a_prod_w.exists() and stage_a_summary.exists():
        log("✅ [STAGE 1/4 ALREADY COMPLETED] Stage A (Base Training & Phase 2) already finished.")
        log(f"   Production checkpoint verified: {stage_a_prod_w}")
        pipeline_state["completed_stages"].append({
            "stage_idx": 1,
            "name": "STAGE A: Base Training & Fine-Tuning (train_best_model.py)",
            "runtime_mins": 174.54,
            "status": "COMPLETED"
        })
        update_status_file()
    else:
        run_step(
            1,
            "STAGE A: Base Training & Fine-Tuning (train_best_model.py)",
            [PYTHON_EXEC, "-u", "train_best_model.py"],
            total_start
        )

    # Stage B: 800px High-Resolution Fine-Tuning (25 epochs)
    run_step(
        2,
        "STAGE B: 800px High-Resolution Fine-Tuning (finetune_800px.py)",
        [PYTHON_EXEC, "-u", "finetune_800px.py"],
        total_start
    )

    # Stage C: Model Accuracy Evaluation & Metrics Generation
    run_step(
        3,
        "STAGE C: Model Accuracy Evaluation & Metrics Generation (evaluate_model.py)",
        [PYTHON_EXEC, "-u", "evaluate_model.py", "--imgsz", "800"],
        total_start
    )

    # Stage D: Hardware Optimization & ONNX FP16 Export
    run_step(
        4,
        "STAGE D: Hardware Optimization & ONNX FP16 Export (export_linescan_engine.py)",
        [PYTHON_EXEC, "-u", "export_linescan_engine.py", "--format", "onnx", "--half", "--benchmark", "50"],
        total_start
    )

    total_time_h = (time.time() - total_start) / 3600
    pipeline_state["status"] = "COMPLETED"
    pipeline_state["elapsed_total_min"] = round(total_time_h * 60, 2)
    pipeline_state["current_stage_name"] = "All Stages Finished"
    update_status_file()

    print("\n" + "=" * 76, flush=True)
    log("🎉 ALL STAGES (A -> B -> C -> D) COMPLETED SUCCESSFULLY!")
    log(f"   Total Pipeline Runtime: {total_time_h:.2f} hours ({total_time_h * 60:.1f} minutes)")
    log(f"   Final Production Model: runs/textile_detection/defect_model_pro_v1/weights/best.pt")
    log(f"   Exported ONNX Engine  : runs/textile_detection/defect_model_pro_v1/weights/best.onnx")
    log(f"   Evaluation Reports    : outputs/accuracy_report.txt, outputs/accuracy_metrics.json")
    print("=" * 76 + "\n", flush=True)

if __name__ == "__main__":
    main()
