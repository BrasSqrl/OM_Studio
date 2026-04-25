"""Per-model persisted threshold configuration for monitoring tests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

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
        test_id="row_count_absolute",
        label="Minimum Row Count",
        category="Data Quality",
        operator=">=",
        value=1.0,
        enabled=True,
        description="Minimum number of rows required in the monitoring dataset.",
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
        test_id="duplicate_identifier_count",
        label="Duplicate Identifier Rows",
        category="Data Quality",
        operator="<=",
        value=0.0,
        enabled=True,
        description="Rows duplicated across the bundle identifier columns.",
    ),
    ThresholdRecord(
        test_id="identifier_null_rate_pct",
        label="Identifier Null Rate %",
        category="Data Quality",
        operator="<=",
        value=0.0,
        enabled=True,
        description="Highest null percentage across identifier columns.",
    ),
    ThresholdRecord(
        test_id="invalid_date_count",
        label="Invalid Date Count",
        category="Data Quality",
        operator="<=",
        value=0.0,
        enabled=True,
        description="Non-empty date values that could not be parsed.",
    ),
    ThresholdRecord(
        test_id="stale_as_of_date_days",
        label="Stale As-Of Date Days",
        category="Data Quality",
        operator="<=",
        value=120.0,
        enabled=True,
        description="Days between the latest monitoring date and the run date.",
    ),
    ThresholdRecord(
        test_id="unexpected_category_count",
        label="Unexpected Categories",
        category="Data Quality",
        operator="<=",
        value=0.0,
        enabled=True,
        description="Categorical feature values present in monitoring data but absent from reference data.",
    ),
    ThresholdRecord(
        test_id="numeric_range_violation_count",
        label="Numeric Range Violations",
        category="Data Quality",
        operator="<=",
        value=0.0,
        enabled=True,
        description="Numeric feature values outside the reference minimum and maximum range.",
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
        test_id="max_feature_psi",
        label="Max Feature PSI",
        category="Drift",
        operator="<=",
        value=0.10,
        enabled=True,
        description="Highest feature-level PSI comparing reference and monitoring raw inputs.",
    ),
    ThresholdRecord(
        test_id="min_numeric_feature_ks_p_value",
        label="Min Numeric Feature Drift P-Value",
        category="Drift",
        operator=">=",
        value=0.05,
        enabled=True,
        description="Lowest numeric-feature KS p-value comparing reference and monitoring raw inputs.",
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
    *,
    source: str = "programmatic",
    actor: str = "local_user",
) -> Path:
    thresholds_root.mkdir(parents=True, exist_ok=True)
    path = _threshold_file_path(thresholds_root, bundle)
    previous_records = load_threshold_records(thresholds_root, bundle)
    payload = {
        "bundle_id": bundle.bundle_id,
        "display_name": bundle.display_name,
        "model_version": bundle.model_version,
        "thresholds": [asdict(record) for record in records],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    _append_threshold_audit_event(
        thresholds_root=thresholds_root,
        bundle=bundle,
        previous_records=previous_records,
        new_records=records,
        source=source,
        actor=actor,
        threshold_file_path=path,
    )
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


def build_threshold_workbook_bytes(records: list[ThresholdRecord]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        threshold_records_to_frame(records).to_excel(
            writer,
            sheet_name="thresholds",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "field": "usage",
                    "value": (
                        "Edit value and enabled fields, then import this workbook back "
                        "into OM Studio for the same model bundle."
                    ),
                },
                {
                    "field": "required_columns",
                    "value": "label, category, operator, value, enabled, description, test_id",
                },
            ]
        ).to_excel(writer, sheet_name="usage_notes", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def threshold_records_from_workbook_bytes(workbook_bytes: bytes) -> list[ThresholdRecord]:
    workbook = pd.read_excel(BytesIO(workbook_bytes), sheet_name="thresholds")
    required_columns = {
        "label",
        "category",
        "operator",
        "value",
        "enabled",
        "description",
        "test_id",
    }
    missing = sorted(required_columns.difference(workbook.columns.astype(str)))
    if missing:
        raise ValueError(
            "Threshold workbook is missing required columns: " + ", ".join(missing)
        )
    return threshold_records_from_frame(workbook)


def records_by_test_id(records: list[ThresholdRecord]) -> dict[str, ThresholdRecord]:
    return {record.test_id: record for record in records}


def load_threshold_audit_frame(thresholds_root: Path, bundle: ModelBundle) -> pd.DataFrame:
    audit_path = _threshold_audit_file_path(thresholds_root, bundle)
    columns = [
        "changed_at_utc",
        "bundle_id",
        "model_version",
        "source",
        "actor",
        "change_count",
        "changed_tests",
        "threshold_file",
    ]
    if not audit_path.exists():
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    with audit_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            changes = payload.get("changes", [])
            rows.append(
                {
                    "changed_at_utc": payload.get("changed_at_utc"),
                    "bundle_id": payload.get("bundle_id"),
                    "model_version": payload.get("model_version"),
                    "source": payload.get("source"),
                    "actor": payload.get("actor"),
                    "change_count": len(changes) if isinstance(changes, list) else 0,
                    "changed_tests": ", ".join(
                        str(item.get("test_id"))
                        for item in changes
                        if isinstance(item, dict) and item.get("test_id")
                    ),
                    "threshold_file": payload.get("threshold_file"),
                }
            )
    return pd.DataFrame(rows, columns=columns)


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
            if record.test_id in {"score_psi", "score_ks_p_value"}:
                record.enabled = False
    if bundle.bundle_paths.reference_input_path is None:
        for record in defaults:
            if record.test_id in {
                "row_count_ratio",
                "segment_psi",
                "unexpected_category_count",
                "numeric_range_violation_count",
                "max_feature_psi",
                "min_numeric_feature_ks_p_value",
            }:
                record.enabled = False
    if not bundle.identifier_columns:
        for record in defaults:
            if record.test_id in {"duplicate_identifier_count", "identifier_null_rate_pct"}:
                record.enabled = False
    if not bundle.date_columns:
        for record in defaults:
            if record.test_id in {"invalid_date_count", "stale_as_of_date_days"}:
                record.enabled = False
    return defaults


def _threshold_file_path(thresholds_root: Path, bundle: ModelBundle) -> Path:
    return thresholds_root / f"{bundle.bundle_id}.json"


def _threshold_audit_file_path(thresholds_root: Path, bundle: ModelBundle) -> Path:
    return thresholds_root / "audit" / f"{bundle.bundle_id}.jsonl"


def _append_threshold_audit_event(
    *,
    thresholds_root: Path,
    bundle: ModelBundle,
    previous_records: list[ThresholdRecord],
    new_records: list[ThresholdRecord],
    source: str,
    actor: str,
    threshold_file_path: Path,
) -> None:
    changes = _build_threshold_changes(previous_records, new_records)
    audit_path = _threshold_audit_file_path(thresholds_root, bundle)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "changed_at_utc": datetime.now(UTC).isoformat(),
        "bundle_id": bundle.bundle_id,
        "display_name": bundle.display_name,
        "model_version": bundle.model_version,
        "source": source,
        "actor": actor,
        "threshold_file": str(threshold_file_path),
        "changes": changes,
    }
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _build_threshold_changes(
    previous_records: list[ThresholdRecord],
    new_records: list[ThresholdRecord],
) -> list[dict[str, Any]]:
    previous_by_id = records_by_test_id(previous_records)
    changes: list[dict[str, Any]] = []
    for record in new_records:
        previous = previous_by_id.get(record.test_id)
        if previous is None:
            changes.append(
                {
                    "test_id": record.test_id,
                    "label": record.label,
                    "field": "record",
                    "old_value": None,
                    "new_value": asdict(record),
                }
            )
            continue
        for field_name in ("operator", "value", "enabled", "description"):
            old_value = getattr(previous, field_name)
            new_value = getattr(record, field_name)
            if old_value == new_value:
                continue
            changes.append(
                {
                    "test_id": record.test_id,
                    "label": record.label,
                    "field": field_name,
                    "old_value": old_value,
                    "new_value": new_value,
                }
            )
    return changes
