"""Profile an OM Studio monitoring run using the built-in demo bundle."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_studio_monitoring import (
    create_demo_assets,
    discover_datasets,
    discover_model_bundles,
    execute_monitoring_run,
)
from quant_studio_monitoring.config import WorkspaceConfig
from quant_studio_monitoring.monitoring_pipeline import ScoringRuntimeOptions
from quant_studio_monitoring.thresholds import load_threshold_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--artifact-profile", choices=["full", "reviewer", "minimal"], default="minimal")
    parser.add_argument("--workspace-root", default="")
    args = parser.parse_args()

    root = (
        Path(args.workspace_root)
        if args.workspace_root
        else Path.cwd() / ".test_workspace" / f"profile_{uuid4().hex}"
    )
    workspace = WorkspaceConfig(
        project_root=root,
        models_root=root / "models",
        incoming_data_root=root / "incoming_data",
        thresholds_root=root / "thresholds",
        runs_root=root / "runs",
    )
    workspace.ensure_directories()
    created = create_demo_assets(workspace, overwrite=True)
    _resize_demo_dataset(created["dataset_path"], args.rows)

    bundle = discover_model_bundles(workspace)[0]
    dataset = discover_datasets(workspace)[0]
    thresholds = load_threshold_records(workspace.thresholds_root, bundle)

    profile = cProfile.Profile()
    profile.enable()
    result = execute_monitoring_run(
        bundle=bundle,
        dataset=dataset,
        workspace=workspace,
        thresholds=thresholds,
        segment_column="region",
        scoring_options=ScoringRuntimeOptions(artifact_profile=args.artifact_profile),
    )
    profile.disable()

    print(
        f"run_id={result.run_id} status={result.status} "
        f"pass={result.pass_count} fail={result.fail_count} na={result.na_count}"
    )
    print(f"run_root={result.run_root}")
    print(f"events={result.artifacts.get('run_events')}")
    pstats.Stats(profile).sort_stats("cumtime").print_stats(30)
    return 0


def _resize_demo_dataset(dataset_path: Path, rows: int) -> None:
    if rows <= 0:
        raise ValueError("--rows must be greater than zero.")
    frame = pd.read_csv(dataset_path)
    if rows <= len(frame):
        resized = frame.head(rows).copy()
    else:
        resized = frame.sample(n=rows, replace=True, random_state=42).reset_index(drop=True)
    resized["loan_id"] = [f"PROFILE_{index:07d}" for index in range(1, rows + 1)]
    resized.to_csv(dataset_path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
