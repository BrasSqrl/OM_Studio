"""Artifact completeness checks for generated monitoring runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ALWAYS_REQUIRED_ARTIFACT_KEYS = frozenset(
    {
        "report",
        "workbook",
        "tests_json",
        "tests_csv",
        "threshold_snapshot",
        "bundle_metadata",
        "input_contract",
        "failure_diagnostics",
        "reviewer_notes",
        "reviewer_exceptions",
        "monitoring_run_config",
        "dataset_provenance",
        "manifest",
        "diagnostic_export",
        "artifact_completeness_csv",
        "artifact_completeness_json",
    }
)
PROFILE_SKIPPABLE_ARTIFACT_KEYS = frozenset({"reviewer_package", "current_scored_data"})
NON_ARTIFACT_MANIFEST_KEYS = frozenset(
    {
        "run_root",
        "artifact_profile",
        "scoring_output_root",
        "generated_artifacts",
        "skipped_artifacts",
        "artifact_completeness_status",
    }
)


def build_artifact_completeness_frame(
    *,
    manifest: dict[str, Any],
    planned_keys: set[str] | None = None,
) -> pd.DataFrame:
    planned_keys = planned_keys or set()
    artifact_profile = str(manifest.get("artifact_profile") or "full")
    skipped_lookup = {
        str(item.get("artifact")): str(item.get("reason"))
        for item in manifest.get("skipped_artifacts", [])
        if isinstance(item, dict)
    }
    keys = sorted(
        {
            key
            for key, value in manifest.items()
            if key not in NON_ARTIFACT_MANIFEST_KEYS
            and (isinstance(value, str) or value is None)
        }
        .union(ALWAYS_REQUIRED_ARTIFACT_KEYS)
        .union(PROFILE_SKIPPABLE_ARTIFACT_KEYS)
        .union(skipped_lookup.keys())
    )
    rows: list[dict[str, Any]] = []
    for key in keys:
        if key in NON_ARTIFACT_MANIFEST_KEYS:
            continue
        value = manifest.get(key)
        required = _is_required(key=key, artifact_profile=artifact_profile)
        skipped_reason = skipped_lookup.get(key)
        exists = False
        if isinstance(value, str):
            exists = Path(value).exists()
        if exists:
            status = "generated"
        elif key in planned_keys:
            status = "planned"
        elif skipped_reason or (not required and value is None):
            status = "skipped"
        elif required:
            status = "missing"
        else:
            status = "not_available"
        rows.append(
            {
                "artifact": key,
                "required": required,
                "status": status,
                "path": value or "",
                "reason": skipped_reason or _default_reason(key, status, artifact_profile),
            }
        )
    return pd.DataFrame(rows)


def write_artifact_completeness_artifacts(
    *,
    manifest: dict[str, Any],
    csv_path: Path,
    json_path: Path,
    planned_keys: set[str] | None = None,
) -> pd.DataFrame:
    frame = build_artifact_completeness_frame(
        manifest=manifest,
        planned_keys=planned_keys,
    )
    frame.to_csv(csv_path, index=False)
    payload = {
        "status": "complete"
        if frame.loc[frame["status"] == "missing"].empty
        else "incomplete",
        "artifacts": frame.to_dict(orient="records"),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return frame


def _is_required(*, key: str, artifact_profile: str) -> bool:
    if key in ALWAYS_REQUIRED_ARTIFACT_KEYS:
        return True
    if key == "reviewer_package":
        return artifact_profile != "minimal"
    if key == "diagnostic_export":
        return True
    if key == "current_scored_data":
        return artifact_profile != "minimal"
    return key not in PROFILE_SKIPPABLE_ARTIFACT_KEYS


def _default_reason(key: str, status: str, artifact_profile: str) -> str:
    if status == "generated":
        return "Artifact exists on disk."
    if status == "planned":
        return "Artifact is expected later in the run finalization sequence."
    if status == "skipped" and artifact_profile == "minimal":
        return "Artifact is skipped by the minimal artifact profile."
    if status == "missing":
        return "Expected artifact was not found on disk."
    return f"Artifact {key} is not applicable for this run."
