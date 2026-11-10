"""
NeuroKernel Member 4: process anomaly investigation and source identification.

The repository README defines these ten model features:
cpu_percent, memory_mb, memory_percent, threads, context_switches,
cpu_user_time, cpu_system_time, cpu_delta, read_bytes_per_sec,
write_bytes_per_sec.

This module does NOT retrain or alter the Random Forest. It accepts the live
process record and model prediction, compares each feature with a normal-only
reference profile, groups unusual features by CPU / Memory / Disk, and returns
a dashboard-friendly explanation.

Important:
- Fit the reference profile from NORMAL training rows only (label == 0).
- Use the same units and sampling interval as the training pipeline.
- `read_bytes_per_sec` and `write_bytes_per_sec` must be rates, not cumulative
  counters.
- A binary anomaly prediction does not itself identify the resource cause.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import math

FEATURES: List[str] = [
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

RESOURCE_FEATURES = {
    "CPU": [
        "cpu_percent", "cpu_user_time", "cpu_system_time", "cpu_delta",
        "context_switches",
    ],
    "Memory": ["memory_mb", "memory_percent"],
    "Disk": ["read_bytes_per_sec", "write_bytes_per_sec"],
    "Process activity": ["threads"],
}


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _get(record: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def normalize_live_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize common psutil / project field aliases into repository schema."""
    aliases = {
        "cpu_percent": ("cpu_percent", "cpu_usage", "cpu"),
        "memory_mb": ("memory_mb", "rss_mb", "memory_usage_mb"),
        "memory_percent": ("memory_percent", "mem_percent"),
        "threads": ("threads", "num_threads", "thread_count"),
        "context_switches": ("context_switches", "ctx_switches"),
        "cpu_user_time": ("cpu_user_time", "user_time"),
        "cpu_system_time": ("cpu_system_time", "system_time"),
        "cpu_delta": ("cpu_delta", "cpu_usage_delta"),
        "read_bytes_per_sec": ("read_bytes_per_sec", "disk_read_bytes_per_sec"),
        "write_bytes_per_sec": ("write_bytes_per_sec", "disk_write_bytes_per_sec"),
    }
    result: Dict[str, Any] = {
        "pid": _get(record, "pid", "process_id"),
        "process_name": str(_get(record, "process_name", "name", default="Unknown")),
        "prediction": _get(record, "prediction", "ml_prediction", "status", default=None),
        "anomaly_score": _to_float(_get(record, "anomaly_score", "score")),
    }
    for feature, names in aliases.items():
        result[feature] = _to_float(_get(record, *names))
    return result


@dataclass
class FeatureBaseline:
    """Robust normal-behaviour profile for the ten model features."""
    median: Dict[str, float]
    mad: Dict[str, float]
    q01: Dict[str, float]
    q99: Dict[str, float]

    @classmethod
    def from_normal_rows(cls, rows: Any) -> "FeatureBaseline":
        """
        Build a profile from a pandas DataFrame containing NORMAL rows only.
        Pass X_train.loc[y_train == 0] (or equivalent), not the full dataset.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required to build a baseline profile") from exc

        missing = [f for f in FEATURES if f not in rows.columns]
        if missing:
            raise ValueError(f"Baseline data is missing model features: {missing}")

        frame = rows[FEATURES].apply(pd.to_numeric, errors="coerce").dropna()
        if frame.empty:
            raise ValueError("No complete normal rows available to build baseline")

        med, mad, q01, q99 = {}, {}, {}, {}
        for feature in FEATURES:
            values = frame[feature]
            med[feature] = float(values.median())
            mad[feature] = float((values - values.median()).abs().median())
            q01[feature] = float(values.quantile(0.01))
            q99[feature] = float(values.quantile(0.99))
        return cls(med, mad, q01, q99)

    def assess(self, feature_values: Mapping[str, Any], robust_z_limit: float = 3.5) -> Dict[str, Dict[str, Any]]:
        """
        Assess high-side deviation from the normal baseline.
        For non-negative resource telemetry, high-side deviations are the
        primary investigation signal. q99 is also reported for transparency.
        """
        result: Dict[str, Dict[str, Any]] = {}
        for feature in FEATURES:
            value = _to_float(feature_values.get(feature))
            if value is None:
                result[feature] = {
                    "value": None, "normal_median": self.median[feature],
                    "normal_q99": self.q99[feature], "robust_z": None,
                    "unusual": False, "available": False,
                }
                continue

            mad = self.mad[feature]
            # 1.4826 * MAD estimates standard deviation for normal-like data.
            scale = 1.4826 * mad
            if scale > 1e-12:
                rz = (value - self.median[feature]) / scale
            else:
                # A constant/near-constant normal feature: only values above
                # the 99th percentile are considered unusual.
                rz = float("inf") if value > self.q99[feature] else 0.0

            unusual = (value > self.q99[feature]) and (rz >= robust_z_limit or math.isinf(rz))
            result[feature] = {
                "value": value,
                "normal_median": self.median[feature],
                "normal_q99": self.q99[feature],
                "robust_z": None if math.isinf(rz) else round(rz, 3),
                "unusual": bool(unusual),
                "available": True,
            }
        return result


def _is_anomaly(prediction: Any) -> Optional[bool]:
    if prediction is None:
        return None
    if isinstance(prediction, bool):
        return prediction
    value = str(prediction).strip().lower()
    if value in {"1", "anomaly", "anomalous", "true", "outlier"}:
        return True
    if value in {"0", "normal", "false", "inlier"}:
        return False
    return None


def investigate_process(
    live_record: Mapping[str, Any],
    baseline: FeatureBaseline,
    robust_z_limit: float = 3.5,
) -> Dict[str, Any]:
    """
    Investigate one process record.

    Expected prediction convention in this repository: 0 = Normal, 1 = Anomaly.
    `prediction` can be 0/1 or Normal/Anomalous.
    """
    record = normalize_live_record(live_record)
    is_anomaly = _is_anomaly(record["prediction"])
    assessment = baseline.assess(record, robust_z_limit=robust_z_limit)

    resource_details: Dict[str, List[str]] = {}
    for resource, features in RESOURCE_FEATURES.items():
        unusual = [f for f in features if assessment[f]["unusual"]]
        if unusual:
            resource_details[resource] = unusual

    # Do not invent a resource cause if the ML model flags an anomaly but
    # the baseline analysis cannot isolate an unusual resource.
    if is_anomaly is False:
        anomaly_type = "None"
        status = "Normal"
    elif is_anomaly is True and resource_details:
        anomaly_type = " + ".join(resource_details.keys())
        status = "Anomalous"
    elif is_anomaly is True:
        anomaly_type = "Unclassified"
        status = "Anomalous"
    else:
        anomaly_type = "Insufficient prediction"
        status = "Unknown"

    reasons: List[str] = []
    for resource, features in resource_details.items():
        for feature in features:
            item = assessment[feature]
            reasons.append(
                f"{feature}={item['value']:.3f} is above the normal "
                f"99th-percentile reference ({item['normal_q99']:.3f})"
            )

    if status == "Normal":
        explanation = (
            f"{record['process_name']} (PID {record['pid']}) was classified "
            "as Normal by the ML model."
        )
    elif anomaly_type == "Unclassified":
        explanation = (
            f"{record['process_name']} (PID {record['pid']}) was flagged by "
            "the ML model, but the normal-baseline comparison did not isolate "
            "CPU, Memory, or Disk as the cause. Review the full feature vector "
            "and model decision; do not infer a cause from global feature importance."
        )
    elif status == "Unknown":
        explanation = (
            "No recognized binary ML prediction was supplied, so the process "
            "cannot be classified reliably."
        )
    else:
        explanation = (
            f"{record['process_name']} (PID {record['pid']}) was flagged as "
            f"anomalous. Resource(s) requiring investigation: {anomaly_type}. "
            + "; ".join(reasons)
            + ". This indicates unusual resource telemetry, not proof of malicious activity."
        )

    return {
        "pid": record["pid"],
        "process_name": record["process_name"],
        "status": status,
        "anomaly_type": anomaly_type,
        "resource_details": resource_details,
        "explanation": explanation,
        "anomaly_score": record["anomaly_score"],
        "features": assessment,
        "raw_feature_values": {feature: record[feature] for feature in FEATURES},
    }


def investigate_processes(
    records: Iterable[Mapping[str, Any]],
    baseline: FeatureBaseline,
    robust_z_limit: float = 3.5,
) -> List[Dict[str, Any]]:
    """Investigate multiple live process records."""
    return [
        investigate_process(record, baseline, robust_z_limit)
        for record in records
    ]
