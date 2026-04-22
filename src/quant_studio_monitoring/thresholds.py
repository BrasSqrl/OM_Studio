"""Per-model persisted threshold configuration for monitoring tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd

from .registry import ModelBundle


@dataclass(slots=True)
class ThresholdRecord:
    test_id: str
    label: str
    category: str
    operator: str
    value: float | None
    enabled: bool
    description: str


BASE_THRESHOLD_CATALOG: list[ThresholdRecord] = [
    ThresholdRecord(
        test_id="required_columns_missing_count",
        label="Schema Missing Columns",
        category="Contract",
        operator="<=",
        value=0.0,
        enabled=True,
        description="Required raw input columns missing from the uploaded monitoring dataset.",
    ),
    ThresholdRecord(
        test_id="row_count_ratio",
        label="Row Count Ratio",
        category="Data Quality",
        operator=">=",
        value=0.50,
        enabled=True,
        description="Monitoring row count divided by the reference row count.",
    ),
    ThresholdRecord(
        test_id="max_feature_missingness_pct",
        label="Max Feature Missingness %",
        category="Data Quality",
        operator="<=",
        value=10.0,
        enabled=True,
        description="Highest missingness percentage across required raw input columns.",
    ),
    ThresholdRecord(
        test_id="score_psi",
        label="Score PSI",
        category="Drift",
        operator="<=",
        value=0.10,
        enabled=True,
        description="Population stability index comparing reference and monitoring scores.",
    ),
    ThresholdRecord(
        test_id="score_ks_p_value",
        label="Score Drift P-Value",
        category="Drift",
        operator=">=",
        value=0.05,
        enabled=True,
        description="Two-sample KS p-value comparing reference and monitoring scores.",
    ),
    ThresholdRecord(
        test_id="segment_psi",
        label="Segment PSI",
        category="Drift",
        operator="<=",
        value=0.10,
        enabled=True,
        description="Population stability index across the selected segment column.",
    ),
    ThresholdRecord(
        test_id="auc",
        label="AUC",
        category="Realized Performance",
        operator=">=",
        value=0.60,
        enabled=True,
        description="Area under the ROC curve on the monitoring population.",
    ),
    ThresholdRecord(
        test_id="ks_statistic",
        label="KS Statistic",
        category="Realized Performance",
        operator=">=",
        value=0.20,
        enabled=True,
        description="Kolmogorov-Smirnov separation between events and non-events.",
    ),
    ThresholdRecord(
        test_id="gini",
        label="Gini",
        category="Realized Performance",
        operator=">=",
        value=0.20,
        enabled=True,
        description="Gini coefficient derived from AUC.",
    ),
    ThresholdRecord(
        test_id="brier_score",
        label="Brier Score",
        category="Realized Performance",
        operator="<=",
        value=0.25,
        enabled=True,
        description="Brier score on the monitoring population.",
    ),
    ThresholdRecord(
        test_id="mean_absolute_error",
        label="MAE",
        category="Realized Performance",
        operator="<=",
        value=0.35,
        enabled=True,
        description="Mean absolute error between actual and predicted outcomes.",
    ),
    ThresholdRecord(
        test_id="hosmer_lemeshow_p_value",
        label="Hosmer-Lemeshow P-Value",
        category="Calibration",
        operator=">=",
        value=0.05,
        enabled=True,
        description="Calibration-group goodness-of-fit p-value for binary models.",
    ),
    ThresholdRecord(
        test_id="precision_at_threshold",
        label="Precision At Threshold",
        category="Threshold",
        operator=">=",
        value=0.30,
        enabled=True,
        description="Precision at the bundle's configured classification threshold.",
    ),
    ThresholdRecord(
        test_id="recall_at_threshold",
        label="Recall At Threshold",
        category="Threshold",
        operator=">=",
        value=0.30,
        enabled=True,
        description="Recall at the bundle's configured classification threshold.",
    ),
    ThresholdRecord(
        test_id="bad_rate_by_band_mae",
        label="Bad Rate By Band MAE",
        category="Calibration",
        operator="<=",
        value=0.10,
        enabled=True,
        description="Mean absolute error between predicted and observed bad rate by score band.",
    ),
]


def load_threshold_records(thresholds_root: Path, bundle: ModelBundle) -> list[ThresholdRecord]:
    defaults = recommended_threshold_records(bundle)
    path = _threshold_file_path(thresholds_root, bundle)
    if not path.exists():
        return defaults

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    persisted_by_id = {
        item["test_id"]: ThresholdRecord(
            test_id=item["test_id"],
            label=item["label"],
            category=item["category"],
            operator=item["operator"],
            value=item.get("value"),
            enabled=bool(item.get("enabled", True)),
            description=item.get("description", ""),
        )
        for item in payload.get("thresholds", [])
    }

    merged: list[ThresholdRecord] = []
    for default in defaults:
        merged.append(persisted_by_id.get(default.test_id, default))
    return merged


def save_threshold_records(
    thresholds_root: Path,
    bundle: ModelBundle,
    records: list[ThresholdRecord],
) -> Path:
    thresholds_root.mkdir(parents=True, exist_ok=True)
    path = _threshold_file_path(thresholds_root, bundle)
    payload = {
        "bundle_id": bundle.bundle_id,
        "display_name": bundle.display_name,
        "model_version": bundle.model_version,
        "thresholds": [asdict(record) for record in records],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def threshold_records_to_frame(records: list[ThresholdRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "label": record.label,
                "category": record.category,
                "operator": record.operator,
                "value": record.value,
                "enabled": record.enabled,
                "description": record.description,
                "test_id": record.test_id,
            }
            for record in records
        ]
    )


def threshold_records_from_frame(frame: pd.DataFrame) -> list[ThresholdRecord]:
    records: list[ThresholdRecord] = []
    for row in frame.to_dict(orient="records"):
        raw_value = row.get("value")
        value = float(raw_value) if raw_value not in ("", None) and pd.notna(raw_value) else None
        records.append(
            ThresholdRecord(
                test_id=str(row["test_id"]),
                label=str(row["label"]),
                category=str(row["category"]),
                operator=str(row["operator"]),
                value=value,
                enabled=bool(row.get("enabled", True)),
                description=str(row.get("description", "")),
            )
        )
    return records


def records_by_test_id(records: list[ThresholdRecord]) -> dict[str, ThresholdRecord]:
    return {record.test_id: record for record in records}


def recommended_threshold_records(bundle: ModelBundle) -> list[ThresholdRecord]:
    defaults = [ThresholdRecord(**asdict(record)) for record in BASE_THRESHOLD_CATALOG]
    if bundle.target_mode != "binary":
        binary_only = {
            "auc",
            "ks_statistic",
            "gini",
            "brier_score",
            "hosmer_lemeshow_p_value",
            "precision_at_threshold",
            "recall_at_threshold",
            "bad_rate_by_band_mae",
        }
        for record in defaults:
            if record.test_id in binary_only:
                record.enabled = False
    if bundle.reference_score_column is None:
        for record in defaults:
            if record.test_id in {"score_psi", "score_ks_p_value", "segment_psi"}:
                record.enabled = False
    return defaults


def _threshold_file_path(thresholds_root: Path, bundle: ModelBundle) -> Path:
    return thresholds_root / f"{bundle.bundle_id}.json"
