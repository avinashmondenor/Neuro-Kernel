"""
NeuroKernel live Linux process monitor.

Place this file at: Neuro-Kernel/src/live_monitor.py

Run from the repository root:
    python src/live_monitor.py
    python src/live_monitor.py --interval 2 --top 25
    python src/live_monitor.py --csv logs/live_predictions.csv

This module loads the existing models/neurokernel_random_forest.pkl model.
It does NOT retrain the model or change the training data.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import psutil


# Repository root is the parent of this script's src/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "neurokernel_random_forest.pkl"

FEATURES = [
    "cpu_percent",
    "memory_mb",
    "memory_percent",
    "threads",
    "context_switches",
    "cpu_user_time",
    "cpu_system_time",
    "cpu_delta",
    "read_bytes_per_sec",
    "write_bytes_per_sec",
]


def is_kernel_thread(pid):
    """Return True only when Linux explicitly marks a process as a kernel thread."""
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Kthread:"):
                    return line.split(":", 1)[1].strip() == "1"
    except (OSError, PermissionError):
        pass
    return False


class LiveProcessMonitor:
    """Collect per-process telemetry and predict Normal (0) / Anomaly (1)."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Expected models/neurokernel_random_forest.pkl in the project."
            )

        self.model = joblib.load(self.model_path)

        # Validate feature compatibility before making any predictions.
        model_features = getattr(self.model, "feature_names_in_", None)
        if model_features is not None and list(model_features) != FEATURES:
            raise ValueError(
                "The saved model's feature names/order do not match this monitor.\n"
                f"Model:   {list(model_features)}\n"
                f"Monitor: {FEATURES}"
            )

        self.previous: dict[int, dict[str, float]] = {}
        self.prediction_class_index = None
        classes = list(getattr(self.model, "classes_", []))
        if 1 in classes:
            self.prediction_class_index = classes.index(1)

    @staticmethod
    def _safe_io_counters(proc: psutil.Process):
        try:
            return proc.io_counters()
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, AttributeError):
            return None

    def scan(self) -> list[dict[str, Any]]:
        """
        Collect one telemetry snapshot.

        A process needs two samples before it can be predicted because CPU and
        disk rates are calculated from changes over elapsed time. On the first
        sample, the process is returned with prediction=None.
        """
        now = time.monotonic()
        records: list[dict[str, Any]] = []
        seen: set[int] = set()

        for proc in psutil.process_iter():
            try:
                with proc.oneshot():
                    pid = proc.pid
                    name = proc.name()
                    if sys.platform.startswith("linux") and is_kernel_thread(proc.pid):
                        continue
                    create_time = proc.create_time()
                    mem_info = proc.memory_info()
                    mem_percent = proc.memory_percent()
                    thread_count = proc.num_threads()
                    cpu_times = proc.cpu_times()
                    ctx = proc.num_ctx_switches()
                    io = self._safe_io_counters(proc)

                    cpu_user = float(cpu_times.user)
                    cpu_system = float(cpu_times.system)
                    cpu_total = cpu_user + cpu_system
                    read_bytes = float(io.read_bytes) if io is not None else 0.0
                    write_bytes = float(io.write_bytes) if io is not None else 0.0
                    context_switches = int(ctx.voluntary + ctx.involuntary)

                seen.add(pid)
                old = self.previous.get(pid)

                # Guard against PID reuse: a new process may receive an old PID.
                if old is not None and old["create_time"] != create_time:
                    old = None

                cpu_delta = 0.0
                cpu_percent = 0.0
                read_rate = 0.0
                write_rate = 0.0
                ready_to_predict = old is not None

                if old is not None:
                    elapsed = now - old["time"]
                    if elapsed > 0:
                        cpu_delta = max(0.0, cpu_total - old["cpu_total"])
                        # Match the calculation used by the existing collector.
                        cpu_percent = (cpu_delta / elapsed) * 100.0
                        read_rate = max(0.0, read_bytes - old["read_bytes"]) / elapsed
                        write_rate = max(0.0, write_bytes - old["write_bytes"]) / elapsed
                    else:
                        ready_to_predict = False

                self.previous[pid] = {
                    "time": now,
                    "create_time": create_time,
                    "cpu_total": cpu_total,
                    "read_bytes": read_bytes,
                    "write_bytes": write_bytes,
                }

                features = {
                    "cpu_percent": round(max(0.0, cpu_percent), 2),
                    "memory_mb": round(float(mem_info.rss) / (1024 * 1024), 2),
                    "memory_percent": round(float(mem_percent), 3),
                    "threads": int(thread_count),
                    "context_switches": context_switches,
                    "cpu_user_time": cpu_user,
                    "cpu_system_time": cpu_system,
                    "cpu_delta": round(max(0.0, cpu_delta), 6),
                    "read_bytes_per_sec": round(max(0.0, read_rate), 2),
                    "write_bytes_per_sec": round(max(0.0, write_rate), 2),
                }

                record: dict[str, Any] = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "pid": pid,
                    "process_name": name,
                    **features,
                    "prediction": None,
                    "status": "Warming up" if not ready_to_predict else "Pending",
                    "anomaly_probability": None,
                    "telemetry_note": (
                        "First sample; waiting for rate calculation"
                        if not ready_to_predict
                        else ""
                    ),
                }

                if ready_to_predict:
                    # Explicit ordered DataFrame: same names and order as training.
                    input_df = pd.DataFrame(
                        [[features[column] for column in FEATURES]],
                        columns=FEATURES,
                    )
                    prediction = int(self.model.predict(input_df)[0])
                    record["prediction"] = prediction
                    record["status"] = "Anomaly" if prediction == 1 else "Normal"

                    if (
                        hasattr(self.model, "predict_proba")
                        and self.prediction_class_index is not None
                    ):
                        probabilities = self.model.predict_proba(input_df)[0]
                        record["anomaly_probability"] = round(
                            float(probabilities[self.prediction_class_index]), 4
                        )

                    if io is None:
                        record["telemetry_note"] = (
                            "Process I/O counters unavailable; disk rates treated as 0"
                        )

                records.append(record)

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess,
                ProcessLookupError,
            ):
                # Processes may exit or be restricted while a scan is running.
                continue
            except (OSError, ValueError) as exc:
                # Skip an individual process with malformed/unavailable telemetry.
                print(
                    f"Warning: skipped PID {getattr(proc, 'pid', '?')}: {exc}",
                    file=sys.stderr,
                )
                continue

        # Remove stale PID baselines so the cache does not grow indefinitely.
        for pid in list(self.previous):
            if pid not in seen:
                self.previous.pop(pid, None)

        return records


def _format_probability(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:5.1f}%"


def print_snapshot(records: list[dict[str, Any]], top: int = 20) -> None:
    ready = [r for r in records if r["prediction"] is not None]
    # Show anomalies first, then processes with the highest CPU usage.
    ready.sort(
        key=lambda r: (
            r["prediction"] != 1,
            -(r["anomaly_probability"] or 0.0),
            -r["cpu_percent"],
        )
    )
    warmup = [r for r in records if r["prediction"] is None]

    print("\n" + "=" * 112)
    print(f"NeuroKernel live monitor | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processes scanned: {len(records)} | Predictions ready: {len(ready)} | Warming up: {len(warmup)}")
    print("-" * 112)
    print(
        f"{'PID':>7}  {'PROCESS':<24} {'CPU %':>8} {'MEM MB':>9} "
        f"{'THREADS':>8} {'STATUS':<12} {'ANOMALY %':>10}"
    )
    print("-" * 112)

    for r in ready[: max(1, top)]:
        name = (r["process_name"] or "unknown")[:24]
        print(
            f"{r['pid']:>7}  {name:<24} {r['cpu_percent']:>8.2f} "
            f"{r['memory_mb']:>9.2f} {r['threads']:>8} "
            f"{r['status']:<12} {_format_probability(r['anomaly_probability']):>10}"
        )

    if warmup:
        print(f"\n{len(warmup)} process(es) are warming up; their second sample is needed for predictions.")
    if not ready:
        print("No predictions are ready yet. Wait for the next sampling cycle.")
    print("Stop with Ctrl+C.")


def append_csv(csv_path: str | Path, records: list[dict[str, Any]]) -> None:
    """Append snapshots to a CSV file, creating parent directories as needed."""
    path = Path(csv_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        return

    fieldnames = list(records[0].keys())
    file_exists_and_nonempty = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists_and_nonempty:
            writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monitor live Linux processes with the existing NeuroKernel Random Forest model."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between scans (default: 2). Must be > 0.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Maximum predicted processes displayed per refresh (default: 20).",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the existing .pkl model.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path, e.g. logs/live_predictions.csv.",
    )
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be greater than 0.")
    if args.top <= 0:
        parser.error("--top must be greater than 0.")

    try:
        monitor = LiveProcessMonitor(args.model)
    except Exception as exc:
        print(f"Could not load NeuroKernel model: {exc}", file=sys.stderr)
        return 1

    print("NeuroKernel — Live Linux Process Anomaly Detection")
    print(f"Model: {monitor.model_path}")
    print(f"Refresh interval: {args.interval:g} seconds")
    print("Starting telemetry collection. The first sample establishes CPU/disk baselines.")

    try:
        while True:
            records = monitor.scan()
            print_snapshot(records, top=args.top)
            if args.csv:
                append_csv(args.csv, records)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopping NeuroKernel live monitor.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
