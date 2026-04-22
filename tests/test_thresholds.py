from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from quant_studio_monitoring.config import WorkspaceConfig
from quant_studio_monitoring.registry import discover_model_bundles
from quant_studio_monitoring.thresholds import load_threshold_records, save_threshold_records


def test_threshold_round_trip() -> None:
    tmp_path = _make_test_root("thresholds")
    workspace = WorkspaceConfig(
        project_root=tmp_path,
        models_root=tmp_path / "models",
        incoming_data_root=tmp_path / "incoming_data",
        thresholds_root=tmp_path / "thresholds",
        runs_root=tmp_path / "runs",
    )
    workspace.ensure_directories()
    bundle_root = workspace.models_root / "retail_pd_v1"
    bundle_root.mkdir(parents=True)
    _write_minimal_bundle(bundle_root)
    bundle = discover_model_bundles(workspace)[0]

    records = load_threshold_records(workspace.thresholds_root, bundle)
    score_psi = next(record for record in records if record.test_id == "score_psi")
    score_psi.value = 0.2
    saved_path = save_threshold_records(workspace.thresholds_root, bundle, records)
    reloaded = load_threshold_records(workspace.thresholds_root, bundle)

    assert saved_path.exists()
    assert next(record for record in reloaded if record.test_id == "score_psi").value == 0.2


def _make_test_root(prefix: str) -> Path:
    root = Path.cwd() / ".test_workspace" / f"{prefix}_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_minimal_bundle(bundle_root) -> None:
    run_config = {
        "schema": {"column_specs": [{"name": "balance", "source_name": "balance", "enabled": True, "role": "feature"}]},
        "documentation": {"model_name": "Retail PD", "model_owner": "Model Risk"},
        "model": {"model_type": "logistic_regression", "threshold": 0.5},
        "target": {"mode": "binary", "output_column": "default_flag"},
        "diagnostics": {"default_segment_column": None},
    }
    (bundle_root / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    (bundle_root / "quant_model.joblib").write_bytes(b"placeholder")
    (bundle_root / "generated_run.py").write_text("print('stub')\n", encoding="utf-8")
    pd.DataFrame({"predicted_probability_recommended": [0.1, 0.2]}).to_csv(
        bundle_root / "predictions.csv",
        index=False,
    )
