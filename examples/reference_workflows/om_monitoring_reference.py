"""Deterministic OM Studio reference monitoring workflows."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from quant_studio_monitoring import (
    create_demo_assets,
    discover_datasets,
    discover_model_bundles,
    execute_monitoring_run,
)
from quant_studio_monitoring.config import WorkspaceConfig
from quant_studio_monitoring.monitoring_pipeline import ScoringRuntimeOptions
from quant_studio_monitoring.registry import DatasetAsset
from quant_studio_monitoring.thresholds import load_threshold_records


def run_reference_workflows(workspace_root: Path | None = None) -> dict[str, dict[str, object]]:
    workspace_root = workspace_root or Path.cwd() / ".test_workspace" / f"reference_{uuid4().hex}"
    workspace = WorkspaceConfig(
        project_root=workspace_root,
        models_root=workspace_root / "models",
        incoming_data_root=workspace_root / "incoming_data",
        thresholds_root=workspace_root / "thresholds",
        runs_root=workspace_root / "runs",
    )
    workspace.ensure_directories()
    created = create_demo_assets(workspace, overwrite=True)
    bundle = discover_model_bundles(workspace)[0]
    thresholds = load_threshold_records(workspace.thresholds_root, bundle)
    base_dataset = discover_datasets(workspace)[0]
    base_frame = pd.read_csv(created["dataset_path"])

    results: dict[str, dict[str, object]] = {}
    happy = execute_monitoring_run(
        bundle=bundle,
        dataset=base_dataset,
        workspace=workspace,
        thresholds=thresholds,
        segment_column="region",
    )
    results["happy_path"] = _summarize(happy)

    missing_path = workspace.incoming_data_root / "contract_missing_columns.csv"
    base_frame[["loan_id", "default_status"]].head(3).to_csv(missing_path, index=False)
    contract_failed = execute_monitoring_run(
        bundle=bundle,
        dataset=_dataset_asset(workspace, missing_path),
        workspace=workspace,
        thresholds=thresholds,
        segment_column="region",
    )
    results["contract_failed"] = _summarize(contract_failed)

    score_only_path = workspace.incoming_data_root / "score_only.csv"
    base_frame.drop(columns=["default_status"]).to_csv(score_only_path, index=False)
    score_only = execute_monitoring_run(
        bundle=bundle,
        dataset=_dataset_asset(workspace, score_only_path),
        workspace=workspace,
        thresholds=thresholds,
        segment_column="region",
    )
    results["score_only"] = _summarize(score_only)

    outcome_path = workspace.incoming_data_root / "outcomes.csv"
    base_frame[["loan_id", "default_status"]].to_csv(outcome_path, index=False)
    outcome_join = execute_monitoring_run(
        bundle=bundle,
        dataset=_dataset_asset(workspace, score_only_path),
        workspace=workspace,
        thresholds=thresholds,
        segment_column="region",
        scoring_options=ScoringRuntimeOptions(
            outcome_dataset_path=outcome_path,
            outcome_join_columns=["loan_id"],
        ),
    )
    results["outcome_join"] = _summarize(outcome_join)

    minimal = execute_monitoring_run(
        bundle=bundle,
        dataset=base_dataset,
        workspace=workspace,
        thresholds=thresholds,
        segment_column="region",
        scoring_options=ScoringRuntimeOptions(artifact_profile="minimal"),
    )
    results["minimal_artifacts"] = _summarize(minimal)
    return results


def _dataset_asset(workspace: WorkspaceConfig, path: Path) -> DatasetAsset:
    stat = path.stat()
    return DatasetAsset(
        dataset_id=path.stem,
        name=path.name,
        path=path,
        suffix=path.suffix.lower(),
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
    )


def _summarize(result) -> dict[str, object]:
    return {
        "status": result.status,
        "labels_available": result.labels_available,
        "failure_stage": result.failure_stage,
        "report": bool(result.artifacts.get("report")),
        "workbook": bool(result.artifacts.get("workbook")),
        "reviewer_package": bool(result.artifacts.get("reviewer_package")),
        "run_config": bool(result.artifacts.get("monitoring_run_config")),
        "step_manifest": bool(result.artifacts.get("step_manifest")),
        "run_debug_trace": bool(result.artifacts.get("run_debug_trace")),
        "test_count": len(result.test_results),
        "failed_count": result.fail_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default="")
    args = parser.parse_args()
    payload = run_reference_workflows(Path(args.workspace_root) if args.workspace_root else None)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
