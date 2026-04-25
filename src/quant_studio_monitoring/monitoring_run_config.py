"""Canonical saved configuration for a single monitoring run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from .config import MonitoringPerformanceConfig
from .file_cache import sha256_file_cached
from .registry import DatasetAsset, ModelBundle
from .thresholds import ThresholdRecord

RUN_CONFIG_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class MonitoringRuntimeSelection:
    artifact_profile: str
    disable_individual_visual_exports: bool
    outcome_dataset_path: str | None
    outcome_join_columns: list[str]
    segment_column: str | None
    performance: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MonitoringRunConfig:
    schema_version: str
    run_id: str
    started_at_utc: str
    bundle_id: str
    model_name: str
    model_version: str
    bundle_root: str
    dataset_id: str
    dataset_name: str
    dataset_path: str
    dataset_sha256: str | None
    dataset_modified_at_utc: str
    row_count: int
    column_count: int
    threshold_snapshot_sha256: str
    enabled_test_ids: list[str]
    disabled_test_ids: list[str]
    runtime: MonitoringRuntimeSelection

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def build_monitoring_run_config(
    *,
    run_id: str,
    started_at: datetime,
    bundle: ModelBundle,
    dataset: DatasetAsset,
    raw_dataframe: pd.DataFrame,
    thresholds: list[ThresholdRecord],
    artifact_profile: str,
    disable_individual_visual_exports: bool,
    outcome_dataset_path: Path | None,
    outcome_join_columns: list[str],
    segment_column: str | None,
    performance: MonitoringPerformanceConfig,
) -> MonitoringRunConfig:
    return MonitoringRunConfig(
        schema_version=RUN_CONFIG_SCHEMA_VERSION,
        run_id=run_id,
        started_at_utc=started_at.astimezone(UTC).isoformat(),
        bundle_id=bundle.bundle_id,
        model_name=bundle.display_name,
        model_version=bundle.model_version,
        bundle_root=str(bundle.bundle_paths.root),
        dataset_id=dataset.dataset_id,
        dataset_name=dataset.name,
        dataset_path=str(dataset.path),
        dataset_sha256=_safe_file_hash(dataset.path),
        dataset_modified_at_utc=dataset.modified_at.astimezone(UTC).isoformat()
        if dataset.modified_at.tzinfo
        else dataset.modified_at.replace(tzinfo=UTC).isoformat(),
        row_count=int(raw_dataframe.shape[0]),
        column_count=int(raw_dataframe.shape[1]),
        threshold_snapshot_sha256=threshold_records_sha256(thresholds),
        enabled_test_ids=[record.test_id for record in thresholds if record.enabled],
        disabled_test_ids=[record.test_id for record in thresholds if not record.enabled],
        runtime=MonitoringRuntimeSelection(
            artifact_profile=artifact_profile,
            disable_individual_visual_exports=disable_individual_visual_exports,
            outcome_dataset_path=str(outcome_dataset_path) if outcome_dataset_path else None,
            outcome_join_columns=list(outcome_join_columns),
            segment_column=segment_column,
            performance=asdict(performance),
        ),
    )


def build_dataset_provenance(
    *,
    dataset: DatasetAsset,
    raw_dataframe: pd.DataFrame,
    run_config: MonitoringRunConfig,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_name": dataset.name,
        "dataset_path": str(dataset.path),
        "dataset_suffix": dataset.suffix,
        "dataset_modified_at_utc": run_config.dataset_modified_at_utc,
        "dataset_sha256": run_config.dataset_sha256,
        "row_count": int(raw_dataframe.shape[0]),
        "column_count": int(raw_dataframe.shape[1]),
        "column_names": raw_dataframe.columns.astype(str).tolist(),
        "schema_sha256": dataframe_schema_sha256(raw_dataframe),
    }


def dataframe_schema_sha256(dataframe: pd.DataFrame) -> str:
    schema_payload = [
        {"column": str(column), "dtype": str(dtype)}
        for column, dtype in zip(dataframe.columns, dataframe.dtypes, strict=False)
    ]
    return sha256(json.dumps(schema_payload, sort_keys=True).encode("utf-8")).hexdigest()


def threshold_records_sha256(records: list[ThresholdRecord]) -> str:
    payload = [asdict(record) for record in records]
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _safe_file_hash(path: Path) -> str | None:
    try:
        return sha256_file_cached(path)
    except OSError:
        return None
