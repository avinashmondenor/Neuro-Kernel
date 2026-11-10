"""Adapter that connects Member 4 investigation to NeuroKernel's existing monitor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from anomaly_investigation import FeatureBaseline, investigate_process


class Member4Investigator:
    """Build a normal-only baseline from the repository's existing training CSVs."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        x_path = self.project_root / "data" / "X_train.csv"
        y_path = self.project_root / "data" / "y_train.csv"

        if not x_path.is_file() or not y_path.is_file():
            raise FileNotFoundError(
                "Member 4 needs data/X_train.csv and data/y_train.csv "
                "to calculate a normal-only reference baseline."
            )

        x_train = pd.read_csv(x_path)
        y_train = pd.read_csv(y_path)

        if "label" in y_train.columns:
            labels = y_train["label"]
        elif y_train.shape[1] == 1:
            labels = y_train.iloc[:, 0]
        else:
            raise ValueError("y_train.csv must contain a 'label' column.")

        if len(x_train) != len(labels):
            raise ValueError("X_train.csv and y_train.csv row counts do not match.")

        normal_rows = x_train.loc[pd.to_numeric(labels, errors="coerce").eq(0)]
        if normal_rows.empty:
            raise ValueError("No label-0 normal rows found in the training data.")

        self.baseline = FeatureBaseline.from_normal_rows(normal_rows)

    def investigate(self, record: dict[str, Any]) -> dict[str, Any]:
        """Return the existing live record plus Member 4's explanation fields."""
        result = investigate_process(record, self.baseline)
        return {
            **record,
            "anomaly_type": result["anomaly_type"],
            "resource_details": result["resource_details"],
            "investigation_explanation": result["explanation"],
            "investigation_features": result["features"],
        }

    def investigate_many(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.investigate(record) for record in records]
