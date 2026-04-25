"""Lightweight persisted run manifest for reviewer traceability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .config import WorkspaceConfig

if TYPE_CHECKING:  # pragma: no cover
    from .monitoring_pipeline import MonitoringRunResult

RUN_INDEX_FILE_NAME = "index.json"


def record_run_index(workspace: WorkspaceConfig, result: MonitoringRunResult) -> Path:
    workspace.runs_root.mkdir(parents=True, exist_ok=True)
    index_path = workspace.runs_root / RUN_INDEX_FILE_NAME
    records = _load_records(index_path)
    record = _result_to_record(result)
    records = [item for item in records if item.get("run_root") != record["run_root"]]
    records.insert(0, record)
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump({"runs": records}, handle, indent=2, default=str)
    return index_path


def load_run_index(workspace: WorkspaceConfig) -> pd.DataFrame:
    records = _load_records(workspace.runs_root / RUN_INDEX_FILE_NAME)
    if not records:
        return pd.DataFrame(
            columns=[
                "run_id",
                "started_at",
                "model_name",
                "model_version",
                "dataset_name",
                "status",
                "passed",
                "failed",
                "na",
                "report",
                "workbook",
                "reviewer_package",
                "run_root",
            ]
        )
    return pd.DataFrame(records)


def _load_records(index_path: Path) -> list[dict[str, Any]]:
    if not index_path.exists():
        return []
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    records = payload.get("runs", [])
    if not isinstance(records, list):
        return []
    return [item for item in records if isinstance(item, dict)]


def _result_to_record(result: MonitoringRunResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "started_at": result.started_at.isoformat(),
        "bundle_id": result.model_bundle.bundle_id,
        "model_name": result.model_bundle.display_name,
        "model_version": result.model_bundle.model_version,
        "dataset_id": result.dataset.dataset_id,
        "dataset_name": result.dataset.name,
        "status": result.status,
        "score_column": result.score_column,
        "labels_available": result.labels_available,
        "segment_column": result.segment_column,
        "passed": result.pass_count,
        "failed": result.fail_count,
        "na": result.na_count,
        "report": _string_path(result.artifacts.get("report")),
        "workbook": _string_path(result.artifacts.get("workbook")),
        "reviewer_package": _string_path(result.artifacts.get("reviewer_package")),
        "run_root": str(result.run_root),
    }


def _string_path(path: Path | str | None) -> str | None:
    return str(path) if path is not None else None
