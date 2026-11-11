"""Member 5 validation utilities for NeuroKernel.

Runs the existing saved model on the held-out test split and performs a live
monitor + Member 4 adapter smoke test. It does not retrain or modify the model.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_monitor import FEATURES, LiveProcessMonitor
from member4_integration import Member4Investigator


def evaluate_saved_model() -> dict[str, Any]:
    """Evaluate the repository model against its untouched held-out test CSVs."""
    x_test = pd.read_csv(ROOT / "data" / "X_test.csv")
    y_test_frame = pd.read_csv(ROOT / "data" / "y_test.csv")
    if "label" not in y_test_frame:
        raise ValueError("data/y_test.csv must contain a 'label' column")
    y_true = pd.to_numeric(y_test_frame["label"], errors="raise").astype(int)

    model = joblib.load(ROOT / "models" / "neurokernel_random_forest.pkl")
    expected = list(getattr(model, "feature_names_in_", FEATURES))
    if list(x_test.columns) != expected:
        raise ValueError(
            "Held-out feature order does not match saved model: "
            f"expected {expected}, got {list(x_test.columns)}"
        )
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "normal_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "normal_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "anomaly_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "anomaly_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "anomaly_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]],
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["Normal", "Anomaly"], output_dict=True, zero_division=0
        ),
    }


def smoke_test_live_integration() -> dict[str, Any]:
    """Check monitor -> ML -> Member 4 adapter using two live scans."""
    monitor = LiveProcessMonitor(ROOT / "models" / "neurokernel_random_forest.pkl")
    investigator = Member4Investigator(ROOT)
    first = monitor.scan()
    time.sleep(1.1)  # second sample enables rate-based features
    second = monitor.scan()
    enriched = investigator.investigate_many(second)

    required = {
        "pid", "process_name", "prediction", "status", "anomaly_probability",
        "anomaly_type", "investigation_explanation", "resource_details",
    }
    missing = sorted(required.difference(enriched[0].keys())) if enriched else sorted(required)
    predicted = [r for r in enriched if r.get("prediction") is not None]
    return {
        "first_scan_processes": len(first),
        "second_scan_processes": len(second),
        "second_scan_predictions_ready": len(predicted),
        "second_scan_warming_up": sum(r.get("prediction") is None for r in enriched),
        "member4_fields_present": not missing,
        "missing_fields": missing,
        "sample_predictions": [
            {
                "pid": r.get("pid"),
                "process_name": r.get("process_name"),
                "prediction": int(r["prediction"]) if r.get("prediction") is not None else None,
                "status": r.get("status"),
                "anomaly_type": r.get("anomaly_type"),
            }
            for r in predicted[:10]
        ],
        "note": (
            "Live predictions are a smoke test only. Live processes have no "
            "independent ground-truth labels, so this cannot establish live FP/FN."
        ),
    }


def main() -> int:
    report: dict[str, Any] = {
        "project": "NeuroKernel",
        "member": 5,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": "models/neurokernel_random_forest.pkl",
        "model_retrained": False,
    }
    errors = []
    try:
        report["held_out_model_evaluation"] = evaluate_saved_model()
    except Exception as exc:
        errors.append(f"held-out evaluation: {type(exc).__name__}: {exc}")
    try:
        report["live_integration_smoke_test"] = smoke_test_live_integration()
    except Exception as exc:
        errors.append(f"live integration: {type(exc).__name__}: {exc}")
    report["errors"] = errors
    report["overall_status"] = "PASS" if not errors else "PARTIAL"
    output = ROOT / "logs" / "member5_validation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nSaved report: {output.relative_to(ROOT)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
