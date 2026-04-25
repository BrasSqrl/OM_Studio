from __future__ import annotations

import zipfile
from pathlib import Path
from uuid import uuid4

from quant_studio_monitoring import (
    create_demo_assets,
    discover_datasets,
    discover_model_bundles,
    execute_monitoring_run,
)
from quant_studio_monitoring.config import WorkspaceConfig
from quant_studio_monitoring.thresholds import load_threshold_records


def test_demo_assets_support_end_to_end_monitoring_flow() -> None:
    workspace = _build_workspace()
    created = create_demo_assets(workspace)

    assert created["bundle_root"].exists()
    assert created["dataset_path"].exists()

    bundles = discover_model_bundles(workspace)
    datasets = discover_datasets(workspace)

    assert len(bundles) == 1
    assert len(datasets) == 1

    bundle = bundles[0]
    dataset = datasets[0]
    thresholds = load_threshold_records(workspace.thresholds_root, bundle)

    result = execute_monitoring_run(
        bundle=bundle,
        dataset=dataset,
        workspace=workspace,
        thresholds=thresholds,
        segment_column="region",
    )

    assert result.status == "completed"
    assert result.error_message is None
    assert result.artifacts["report"] is not None
    assert result.artifacts["workbook"] is not None
    assert result.artifacts["reviewer_package"] is not None
    assert result.support_tables["reference_monitoring_delta_summary"].shape[0] >= 3

    package_path = Path(result.artifacts["reviewer_package"])
    with zipfile.ZipFile(package_path, "r") as archive:
        names = set(archive.namelist())

    assert "monitoring_report.html" in names
    assert "monitoring_workbook.xlsx" in names
    assert "monitoring_test_results.csv" in names
    assert "artifacts_manifest.json" in names


def _build_workspace() -> WorkspaceConfig:
    tmp_path = _make_test_root("golden_path")
    workspace = WorkspaceConfig(
        project_root=tmp_path,
        models_root=tmp_path / "models",
        incoming_data_root=tmp_path / "incoming_data",
        thresholds_root=tmp_path / "thresholds",
        runs_root=tmp_path / "runs",
    )
    workspace.ensure_directories()
    return workspace


def _make_test_root(prefix: str) -> Path:
    root = Path.cwd() / ".test_workspace" / f"{prefix}_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root
