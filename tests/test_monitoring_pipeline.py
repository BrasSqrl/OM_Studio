from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from quant_studio_monitoring.config import WorkspaceConfig
from quant_studio_monitoring.monitoring_pipeline import (
    ScoringExecutionOutput,
    ScoringRuntimeOptions,
    execute_monitoring_run,
)
from quant_studio_monitoring.registry import DatasetAsset, discover_model_bundles
from quant_studio_monitoring.run_history import load_run_index
from quant_studio_monitoring.thresholds import load_threshold_records


def test_execute_monitoring_run_contract_failure_writes_artifacts() -> None:
    workspace = _build_workspace()
    bundle_root = workspace.models_root / "retail_pd_v1"
    bundle_root.mkdir(parents=True)
    _write_bundle(bundle_root)

    dataset_path = workspace.incoming_data_root / "monitoring.csv"
    pd.DataFrame({"loan_id": ["L1"], "default_status": [1]}).to_csv(dataset_path, index=False)

    bundle = discover_model_bundles(workspace)[0]
    dataset = DatasetAsset(
        dataset_id="monitoring_csv",
        name=dataset_path.name,
        path=dataset_path,
        suffix=".csv",
        modified_at=pd.Timestamp.now("UTC").to_pydatetime(),
    )
    thresholds = load_threshold_records(workspace.thresholds_root, bundle)
    result = execute_monitoring_run(
        bundle=bundle,
        dataset=dataset,
        workspace=workspace,
        thresholds=thresholds,
        segment_column=None,
    )

    assert result.status == "contract_failed"
    assert result.fail_count >= 1
    assert result.artifacts["report"] is not None
    assert result.artifacts["workbook"] is not None
    assert result.artifacts["manifest"] is not None
    assert result.failure_stage == "input_contract"
    assert result.artifacts["failure_diagnostics"] is not None
    assert result.artifacts["run_events"] is not None
    assert result.artifacts["monitoring_run_config"] is not None
    assert result.artifacts["dataset_provenance"] is not None
    assert result.artifacts["step_manifest"] is not None
    assert result.artifacts["run_debug_trace"] is not None
    assert result.artifacts["artifact_completeness_csv"] is not None
    assert result.artifacts["artifact_completeness_json"] is not None
    assert result.artifacts["diagnostic_export"] is not None
    assert Path(result.artifacts["run_events"]).exists()
    assert Path(result.artifacts["step_manifest"]).exists()
    assert Path(result.artifacts["run_debug_trace"]).exists()
    assert Path(result.artifacts["diagnostic_export"]).exists()


def test_execute_monitoring_run_with_mocked_scoring_produces_binary_results(
    monkeypatch,
) -> None:
    workspace = _build_workspace()
    bundle_root = workspace.models_root / "retail_pd_v1"
    bundle_root.mkdir(parents=True)
    _write_bundle(bundle_root)

    dataset_path = workspace.incoming_data_root / "monitoring.csv"
    pd.DataFrame(
        {
            "as_of_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "loan_id": ["L1", "L2", "L3", "L4"],
            "balance": [100.0, 120.0, 90.0, 140.0],
            "region": ["north", "south", "north", "east"],
            "default_status": [0, 1, 0, 1],
        }
    ).to_csv(dataset_path, index=False)

    bundle = discover_model_bundles(workspace)[0]
    dataset = DatasetAsset(
        dataset_id="monitoring_csv",
        name=dataset_path.name,
        path=dataset_path,
        suffix=".csv",
        modified_at=pd.Timestamp.now("UTC").to_pydatetime(),
    )

    def _mock_scoring(**kwargs):
        output_root = kwargs["scoring_root"] / "mock_run"
        output_root.mkdir(parents=True)
        predictions = pd.DataFrame(
            {
                "loan_id": ["L1", "L2", "L3", "L4"],
                "default_flag": [0, 1, 0, 1],
                "predicted_probability_recommended": [0.1, 0.8, 0.2, 0.7],
            }
        )
        predictions.to_csv(output_root / "predictions.csv", index=False)
        return ScoringExecutionOutput(
            output_root=output_root,
            predictions=predictions,
            metrics={},
            bundle_artifacts={},
        )

    monkeypatch.setattr(
        "quant_studio_monitoring.monitoring_pipeline.run_bundle_scoring",
        _mock_scoring,
    )

    thresholds = load_threshold_records(workspace.thresholds_root, bundle)
    result = execute_monitoring_run(
        bundle=bundle,
        dataset=dataset,
        workspace=workspace,
        thresholds=thresholds,
        segment_column="region",
        scoring_options=ScoringRuntimeOptions(
            disable_individual_visual_exports=True
        ),
    )

    assert result.status == "completed"
    assert result.score_column == "predicted_probability_recommended"
    results = result.results_frame.set_index("test_id")
    assert results.loc["auc", "status"] == "pass"
    assert results.loc["score_psi", "observed_value"] is not None
    assert result.artifacts["report"] is not None
    assert result.artifacts["reviewer_package"] is not None
    assert result.artifacts["run_events"] is not None
    assert result.artifacts["monitoring_run_config"] is not None
    assert result.artifacts["dataset_provenance"] is not None
    assert result.artifacts["step_manifest"] is not None
    assert result.artifacts["run_debug_trace"] is not None
    assert result.artifacts["artifact_completeness_csv"] is not None
    assert result.artifacts["diagnostic_export"] is not None
    assert "reference_monitoring_delta_summary" in result.support_tables
    assert "data_quality_summary" in result.support_tables
    assert "feature_drift_summary" in result.support_tables
    assert "reference_baseline_diagnostics" in result.support_tables
    assert "test_applicability_matrix" in result.support_tables
    assert "artifact_completeness" in result.support_tables
    assert "max_feature_psi" in set(results.index)
    assert "duplicate_identifier_count" in set(results.index)

    persisted = load_run_index(workspace)
    assert persisted.loc[0, "run_id"] == result.run_id
    assert persisted.loc[0, "status"] == "completed"


def test_execute_monitoring_run_prepares_score_existing_model_inputs_for_optional_columns() -> None:
    workspace = _build_workspace()
    bundle_root = workspace.models_root / "retail_pd_v1"
    bundle_root.mkdir(parents=True)
    _write_bundle(bundle_root, include_ignore_column=True, runner_mode="capture")

    dataset_path = workspace.incoming_data_root / "monitoring.csv"
    pd.DataFrame(
        {
            "as_of_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "loan_id": ["L1", "L2", "L3"],
            "balance": [100.0, 120.0, 90.0],
            "region": ["north", "south", "east"],
        }
    ).to_csv(dataset_path, index=False)

    bundle = discover_model_bundles(workspace)[0]
    dataset = DatasetAsset(
        dataset_id="monitoring_csv",
        name=dataset_path.name,
        path=dataset_path,
        suffix=".csv",
        modified_at=pd.Timestamp.now("UTC").to_pydatetime(),
    )
    thresholds = load_threshold_records(workspace.thresholds_root, bundle)
    result = execute_monitoring_run(
        bundle=bundle,
        dataset=dataset,
        workspace=workspace,
        thresholds=thresholds,
        segment_column="region",
        scoring_options=ScoringRuntimeOptions(
            disable_individual_visual_exports=True
        ),
    )

    assert result.status == "completed"
    assert result.labels_available is False
    assert result.scoring_output_root is not None

    captured_config = json.loads(
        (result.scoring_output_root / "captured_config.json").read_text(encoding="utf-8")
    )
    assert captured_config["execution"]["mode"] == "score_existing_model"
    assert captured_config["execution"]["existing_model_path"]
    assert captured_config["execution"]["existing_config_path"]
    assert captured_config["diagnostics"]["interactive_visualizations"] is False
    assert captured_config["diagnostics"]["static_image_exports"] is False

    captured_input = pd.read_csv(result.scoring_output_root / "captured_input.csv")
    assert "legacy_text_field" in captured_input.columns
    assert captured_input["legacy_text_field"].eq("__monitoring_placeholder__").all()
    assert "default_status" in captured_input.columns
    assert captured_input["default_status"].isna().all()

    results = result.results_frame.set_index("test_id")
    assert results.loc["auc", "status"] == "na"
    assert result.artifacts["report"] is not None
    assert result.artifacts["reviewer_package"] is not None
    assert result.artifacts["monitoring_run_config"] is not None
    report_html = Path(result.artifacts["report"]).read_text(encoding="utf-8")
    assert "Score-Only Run" in report_html
    assert "Bundle Fingerprint" in report_html
    assert "Dataset Fingerprint" in report_html
    assert "Reference Vs Monitoring Delta" in report_html
    assert "Executive Summary" in report_html


def test_execute_monitoring_run_joins_separate_outcome_file_and_exports_notes(
    monkeypatch,
) -> None:
    workspace = _build_workspace()
    bundle_root = workspace.models_root / "retail_pd_v1"
    bundle_root.mkdir(parents=True)
    _write_bundle(bundle_root)

    dataset_path = workspace.incoming_data_root / "monitoring.csv"
    pd.DataFrame(
        {
            "as_of_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "loan_id": ["L1", "L2", "L3", "L4"],
            "balance": [100.0, 120.0, 90.0, 140.0],
            "region": ["north", "south", "north", "east"],
        }
    ).to_csv(dataset_path, index=False)
    outcome_path = workspace.incoming_data_root / "outcomes.csv"
    pd.DataFrame(
        {
            "loan_id": ["L1", "L2", "L3", "L4"],
            "default_status": [0, 1, 0, 1],
        }
    ).to_csv(outcome_path, index=False)

    bundle = discover_model_bundles(workspace)[0]
    dataset = DatasetAsset(
        dataset_id="monitoring_csv",
        name=dataset_path.name,
        path=dataset_path,
        suffix=".csv",
        modified_at=pd.Timestamp.now("UTC").to_pydatetime(),
    )

    def _mock_scoring(**kwargs):
        raw_dataframe = kwargs["raw_dataframe"]
        assert raw_dataframe["default_status"].tolist() == [0, 1, 0, 1]
        output_root = kwargs["scoring_root"] / "mock_run"
        output_root.mkdir(parents=True)
        predictions = pd.DataFrame(
            {
                "loan_id": raw_dataframe["loan_id"],
                "default_flag": raw_dataframe["default_status"],
                "predicted_probability_recommended": [0.1, 0.8, 0.2, 0.7],
            }
        )
        predictions.to_csv(output_root / "predictions.csv", index=False)
        return ScoringExecutionOutput(
            output_root=output_root,
            predictions=predictions,
            metrics={},
            bundle_artifacts={},
        )

    monkeypatch.setattr(
        "quant_studio_monitoring.monitoring_pipeline.run_bundle_scoring",
        _mock_scoring,
    )

    thresholds = load_threshold_records(workspace.thresholds_root, bundle)
    result = execute_monitoring_run(
        bundle=bundle,
        dataset=dataset,
        workspace=workspace,
        thresholds=thresholds,
        scoring_options=ScoringRuntimeOptions(
            outcome_dataset_path=outcome_path,
            outcome_join_columns=["loan_id"],
        ),
        reviewer_notes={"score_psi": "Reviewed drift exception."},
        reviewer_exceptions={
            "score_psi": {
                "disposition": "Exception Noted",
                "rationale": "Temporary portfolio mix shift.",
            }
        },
    )

    assert result.status == "completed"
    assert result.labels_available is True
    assert "outcome_join_summary" in result.support_tables
    assert result.support_tables["outcome_join_summary"].loc[0, "matched_rows"] == 4
    assert result.artifacts["reviewer_notes"] is not None
    assert result.artifacts["reviewer_exceptions"] is not None
    assert result.artifacts["run_events"] is not None
    assert result.artifacts["monitoring_run_config"] is not None
    notes_payload = json.loads(Path(result.artifacts["reviewer_notes"]).read_text(encoding="utf-8"))
    assert notes_payload["reviewer_notes"]["score_psi"] == "Reviewed drift exception."
    exceptions_payload = json.loads(
        Path(result.artifacts["reviewer_exceptions"]).read_text(encoding="utf-8")
    )
    assert exceptions_payload["reviewer_exceptions"]["score_psi"]["disposition"] == "Exception Noted"
    assert "reviewer_exception_summary" in result.support_tables


def test_execute_monitoring_run_minimal_artifact_profile_skips_reviewer_package() -> None:
    workspace = _build_workspace()
    bundle_root = workspace.models_root / "retail_pd_v1"
    bundle_root.mkdir(parents=True)
    _write_bundle(bundle_root)

    dataset_path = workspace.incoming_data_root / "monitoring.csv"
    pd.DataFrame({"loan_id": ["L1"], "default_status": [1]}).to_csv(dataset_path, index=False)

    bundle = discover_model_bundles(workspace)[0]
    dataset = DatasetAsset(
        dataset_id="monitoring_csv",
        name=dataset_path.name,
        path=dataset_path,
        suffix=".csv",
        modified_at=pd.Timestamp.now("UTC").to_pydatetime(),
    )
    thresholds = load_threshold_records(workspace.thresholds_root, bundle)
    result = execute_monitoring_run(
        bundle=bundle,
        dataset=dataset,
        workspace=workspace,
        thresholds=thresholds,
        scoring_options=ScoringRuntimeOptions(artifact_profile="minimal"),
    )

    assert result.artifacts["report"] is not None
    assert result.artifacts["workbook"] is not None
    assert result.artifacts["reviewer_package"] is None
    assert result.artifacts["diagnostic_export"] is not None
    manifest = json.loads(Path(result.artifacts["manifest"]).read_text(encoding="utf-8"))
    assert manifest["artifact_profile"] == "minimal"
    assert "generated_artifacts" in manifest
    assert any(item["artifact"] == "reviewer_package" for item in manifest["skipped_artifacts"])
    assert manifest["artifact_completeness_status"] == "complete"
    assert Path(result.artifacts["run_events"]).exists()


def _build_workspace() -> WorkspaceConfig:
    tmp_path = _make_test_root("pipeline")
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


def _write_bundle(
    bundle_root,
    *,
    include_ignore_column: bool = False,
    runner_mode: str = "stub",
) -> None:
    column_specs = [
        {"name": "as_of_date", "source_name": "as_of_date", "enabled": True, "role": "date"},
        {"name": "loan_id", "source_name": "loan_id", "enabled": True, "role": "identifier"},
        {"name": "balance", "source_name": "balance", "enabled": True, "role": "feature"},
        {"name": "region", "source_name": "region", "enabled": True, "role": "feature"},
    ]
    if include_ignore_column:
        column_specs.append(
            {
                "name": "legacy_text_field",
                "source_name": "legacy_text_field",
                "enabled": True,
                "role": "ignore",
            }
        )
    column_specs.append(
        {
            "name": "default_status",
            "source_name": "default_status",
            "enabled": True,
            "role": "target_source",
        }
    )
    run_config = {
        "schema": {
            "column_specs": column_specs
        },
        "documentation": {"model_name": "Retail PD", "model_owner": "Model Risk"},
        "model": {"model_type": "logistic_regression", "threshold": 0.5},
        "target": {"mode": "binary", "output_column": "default_flag"},
        "diagnostics": {"default_segment_column": "region"},
    }
    (bundle_root / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    (bundle_root / "quant_model.joblib").write_bytes(b"placeholder")
    if runner_mode == "capture":
        (bundle_root / "generated_run.py").write_text(
            """
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    dataframe = pd.read_csv(args.input)
    output_root = Path(args.output_root) / "captured_run"
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "captured_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    dataframe.to_csv(output_root / "captured_input.csv", index=False)

    predictions = pd.DataFrame(
        {
            "loan_id": dataframe["loan_id"],
            "predicted_probability_recommended": [0.2, 0.4, 0.6][: len(dataframe)],
            "default_flag": pd.to_numeric(dataframe.get("default_status"), errors="coerce"),
        }
    )
    predictions.to_csv(output_root / "predictions.csv", index=False)
    print(f"Run completed. Artifacts written to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""".strip()
            + "\n",
            encoding="utf-8",
        )
    else:
        (bundle_root / "generated_run.py").write_text("print('stub')\n", encoding="utf-8")
    pd.DataFrame(
        {
            "as_of_date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "loan_id": ["L1", "L2", "L3", "L4"],
            "balance": [100.0, 110.0, 95.0, 130.0],
            "region": ["north", "south", "north", "east"],
        }
    ).to_csv(bundle_root / "input_snapshot.csv", index=False)
    pd.DataFrame(
        {
            "predicted_probability_recommended": [0.12, 0.75, 0.18, 0.69],
            "default_flag": [0, 1, 0, 1],
        }
    ).to_csv(bundle_root / "predictions.csv", index=False)
