"""Synthetic unit tests for Member 4 anomaly investigation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.anomaly_investigation import FEATURES, FeatureBaseline, investigate_process


def make_baseline():
    # Synthetic stable normal reference values. Replace with actual normal
    # X_train rows when integrating into the project.
    rows = []
    for i in range(100):
        rows.append({
            "cpu_percent": 5 + (i % 5) * 0.2,
            "memory_mb": 100 + (i % 5),
            "memory_percent": 2 + (i % 5) * 0.02,
            "threads": 4 + (i % 2),
            "context_switches": 1000 + (i % 5) * 10,
            "cpu_user_time": 20 + (i % 5) * 0.1,
            "cpu_system_time": 5 + (i % 5) * 0.05,
            "cpu_delta": 1 + (i % 5) * 0.05,
            "read_bytes_per_sec": 10000 + (i % 5) * 100,
            "write_bytes_per_sec": 8000 + (i % 5) * 100,
        })
    return FeatureBaseline.from_normal_rows(pd.DataFrame(rows, columns=FEATURES))


def test_cpu():
    b = make_baseline()
    record = {f: 1 for f in FEATURES}
    record.update({
        "pid": 11, "name": "cpu-worker", "prediction": 1,
        "cpu_percent": 99, "memory_mb": 102, "memory_percent": 2.04,
        "threads": 4, "context_switches": 1020,
        "cpu_user_time": 20.2, "cpu_system_time": 5.1,
        "cpu_delta": 1.1, "read_bytes_per_sec": 10100,
        "write_bytes_per_sec": 8100,
    })
    result = investigate_process(record, b)
    assert result["status"] == "Anomalous"
    assert "CPU" in result["anomaly_type"]


def test_memory():
    b = make_baseline()
    record = {
        "pid": 12, "name": "memory-worker", "prediction": 1,
        "cpu_percent": 5.2, "memory_mb": 900, "memory_percent": 40,
        "threads": 4, "context_switches": 1020, "cpu_user_time": 20.2,
        "cpu_system_time": 5.1, "cpu_delta": 1.1,
        "read_bytes_per_sec": 10100, "write_bytes_per_sec": 8100,
    }
    result = investigate_process(record, b)
    assert result["status"] == "Anomalous"
    assert "Memory" in result["anomaly_type"]


def test_disk():
    b = make_baseline()
    record = {
        "pid": 13, "name": "disk-worker", "prediction": 1,
        "cpu_percent": 5.2, "memory_mb": 102, "memory_percent": 2.04,
        "threads": 4, "context_switches": 1020, "cpu_user_time": 20.2,
        "cpu_system_time": 5.1, "cpu_delta": 1.1,
        "read_bytes_per_sec": 9000000, "write_bytes_per_sec": 8100,
    }
    result = investigate_process(record, b)
    assert result["status"] == "Anomalous"
    assert "Disk" in result["anomaly_type"]


def test_normal():
    b = make_baseline()
    record = {
        "pid": 14, "name": "bash", "prediction": 0,
        "cpu_percent": 5.2, "memory_mb": 102, "memory_percent": 2.04,
        "threads": 4, "context_switches": 1020, "cpu_user_time": 20.2,
        "cpu_system_time": 5.1, "cpu_delta": 1.1,
        "read_bytes_per_sec": 10100, "write_bytes_per_sec": 8100,
    }
    result = investigate_process(record, b)
    assert result["status"] == "Normal"
    assert result["anomaly_type"] == "None"


def test_anomaly_unclassified():
    b = make_baseline()
    record = {
        "pid": 15, "name": "unknown-worker", "prediction": 1,
        "cpu_percent": 5.2, "memory_mb": 102, "memory_percent": 2.04,
        "threads": 4, "context_switches": 1020, "cpu_user_time": 20.2,
        "cpu_system_time": 5.1, "cpu_delta": 1.1,
        "read_bytes_per_sec": 10100, "write_bytes_per_sec": 8100,
    }
    result = investigate_process(record, b)
    assert result["status"] == "Anomalous"
    assert result["anomaly_type"] == "Unclassified"


if __name__ == "__main__":
    test_cpu()
    test_memory()
    test_disk()
    test_normal()
    test_anomaly_unclassified()
    print("All 5 Member 4 investigation tests passed.")
