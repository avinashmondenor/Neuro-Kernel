"""Member 5 regression tests: evaluation integrity and integration edge cases."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import pytest

from member5_validation import evaluate_saved_model
from anomaly_investigation import FEATURES, FeatureBaseline, investigate_process


def test_held_out_evaluation_has_consistent_confusion_counts():
    result = evaluate_saved_model()
    tn, fp, fn, tp = sum(result["confusion_matrix_labels_0_1"], [])
    assert result["test_samples"] == tn + fp + fn + tp
    assert result["false_positive"] == fp
    assert result["false_negative"] == fn
    assert 0 <= result["accuracy"] <= 1


def _baseline():
    rows = []
    for i in range(80):
        rows.append({
            "cpu_percent": 3 + i % 4,
            "memory_mb": 80 + i % 4,
            "memory_percent": 1 + (i % 4) * .01,
            "threads": 3 + i % 2,
            "context_switches": 500 + i % 4,
            "cpu_user_time": 10 + (i % 4) * .1,
            "cpu_system_time": 2 + (i % 4) * .1,
            "cpu_delta": .1 + (i % 4) * .01,
            "read_bytes_per_sec": 100 + i % 4,
            "write_bytes_per_sec": 80 + i % 4,
        })
    return FeatureBaseline.from_normal_rows(pd.DataFrame(rows, columns=FEATURES))


def test_warming_up_process_is_not_misclassified():
    record = {"pid": 1, "process_name": "new-process", "prediction": None}
    result = investigate_process(record, _baseline())
    assert result["status"] == "Unknown"
    assert result["anomaly_type"] == "Insufficient prediction"


def test_missing_telemetry_is_reported_as_unavailable_not_crash():
    record = {"pid": 2, "process_name": "restricted-process", "prediction": 1}
    result = investigate_process(record, _baseline())
    assert result["status"] == "Anomalous"
    assert result["anomaly_type"] == "Unclassified"
    assert all(not item["available"] for item in result["features"].values())


def test_normal_prediction_stays_normal_even_with_high_resource_feature():
    record = {
        "pid": 3, "process_name": "normal-labelled", "prediction": 0,
        "cpu_percent": 99, "memory_mb": 80, "memory_percent": 1,
        "threads": 3, "context_switches": 500, "cpu_user_time": 10,
        "cpu_system_time": 2, "cpu_delta": .1,
        "read_bytes_per_sec": 100, "write_bytes_per_sec": 80,
    }
    result = investigate_process(record, _baseline())
    assert result["status"] == "Normal"
    assert result["anomaly_type"] == "None"
