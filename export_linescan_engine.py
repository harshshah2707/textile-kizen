"""
export_linescan_engine.py
=========================
Export utility to convert trained YOLOv8 PyTorch model (.pt) to high-speed
TensorRT (.engine) and ONNX (.onnx) formats with FP16 half-precision, optimized
for industrial line-scan camera inference.

Usage:
    python export_linescan_engine.py --format engine --imgsz 640
    python export_linescan_engine.py --format onnx --half
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

BASE_DIR       = Path(__file__).parent.resolve()
DEFAULT_MODEL = BASE_DIR / "runs" / "textile_detection" / "defect_model_pro_v1" / "weights" / "best.pt"


def export_model(
    model_path:  Path  = DEFAULT_MODEL,
    format_type: str   = "engine",
    imgsz:       int   = 640,
    half:        bool  = True,
    dynamic:     bool  = False,     # static batch is faster for TensorRT
    device:      int   = 0,
    batch:       int   = 8,         # batch=8 tiles for RTX 3050 (8 GB VRAM)
    workspace:   int   = 2048,      # TensorRT workspace memory in MB
    benchmark_runs: int = 0,        # >0: run timed benchmark after export
):
    print("=" * 70)
    print("  TEXTILE DEFECT DETECTION \u2014 LINE-SCAN HARDWARE EXPORTER")
    print("=" * 70)
    print(f"  Input Model    : {model_path}")
    print(f"  Export Format  : {format_type.upper()}")
    print(f"  Tile Resolution: {imgsz} x {imgsz}")
    print(f"  FP16 Half-Prec : {half}")
    print(f"  Static Batch   : {batch}")
    print(f"  TRT Workspace  : {workspace} MB")
    print(f"  Dynamic Shape  : {dynamic}")
    print("=" * 70)

    if not model_path.exists():
        print(f"[Error] Weight file not found: {model_path}")
        return False

    model = YOLO(str(model_path))

    export_kwargs = dict(
        format=format_type,
        imgsz=imgsz,
        half=half,
        dynamic=dynamic,
        device=device if torch.cuda.is_available() else "cpu",
        simplify=True,
        verbose=True,
        batch=batch,
    )
    if format_type.lower() == "engine":
        export_kwargs["workspace"] = workspace

    try:
        exported_path = model.export(**export_kwargs)
        print("\n" + "=" * 70)
        print(f"  \u2705 SUCCESS: Exported line-scan model to:\n  --> {exported_path}")
        print("=" * 70)

        # Optional post-export benchmark
        if benchmark_runs > 0:
            _benchmark_exported(exported_path, imgsz, half, device, benchmark_runs, batch=batch)

        return True
    except Exception as e:
        print(f"\n[Error] Failed to export model to {format_type}: {e}")
        return False


def _benchmark_exported(exported_path, imgsz: int, half: bool, device: int, runs: int = 100, batch: int = 1):
    """Run timed inference benchmark on exported model to validate FPS."""
    import numpy as np, time
    print(f"\n[Benchmark] Loading exported model: {exported_path} (batch={batch})")
    exported_model = YOLO(str(exported_path), task="detect")
    dummy = [np.zeros((imgsz, imgsz, 3), dtype=np.uint8)] * batch

    # Warmup
    for _ in range(5):
        exported_model.predict(source=dummy, verbose=False, device=device, half=half)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        exported_model.predict(source=dummy, verbose=False, device=device, half=half)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times = sorted(times)
    avg_ms = sum(times) / len(times)
    per_tile_ms = avg_ms / batch
    fps = 1000.0 / per_tile_ms if per_tile_ms > 0 else 0
    print("\n" + "=" * 70)
    print("  POST-EXPORT BENCHMARK")
    print("=" * 70)
    print(f"  Format         : {Path(str(exported_path)).suffix}")
    print(f"  imgsz          : {imgsz}")
    print(f"  Batch size     : {batch}")
    print(f"  FP16           : {half}")
    print(f"  Runs           : {runs}")
    print(f"  Batch latency  : {avg_ms:.1f} ms")
    print(f"  Per-tile lat   : {per_tile_ms:.1f} ms")
    print(f"  Min latency    : {min(times)/batch:.1f} ms/tile")
    print(f"  P95 latency    : {times[int(len(times)*0.95)]/batch:.1f} ms/tile")
    print(f"  Throughput FPS : {fps:.1f} tiles/sec")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model for line-scan hardware acceleration (RTX 3050 optimised)"
    )
    parser.add_argument("--weights",    type=str,   default=str(DEFAULT_MODEL),
                        help="Path to PyTorch .pt model")
    parser.add_argument("--format",     type=str,   default="onnx",
                        choices=["engine", "onnx", "torchscript"],
                        help="Export target format")
    parser.add_argument("--imgsz",      type=int,   default=640,
                        help="Tile resolution")
    parser.add_argument("--half",       action="store_true", default=True,
                        help="Enable FP16 half precision (default: on)")
    parser.add_argument("--batch",      type=int,   default=8,
                        help="Static batch size for TensorRT export (default: 8 for RTX 3050)")
    parser.add_argument("--workspace",  type=int,   default=2048,
                        help="TensorRT workspace memory in MB (default: 2048)")
    parser.add_argument("--benchmark",  type=int,   default=0,
                        help="Run post-export benchmark with N inference passes (0 = skip)")
    args = parser.parse_args()

    export_model(
        model_path=Path(args.weights),
        format_type=args.format,
        imgsz=args.imgsz,
        half=args.half,
        batch=args.batch,
        workspace=args.workspace,
        benchmark_runs=args.benchmark,
    )
