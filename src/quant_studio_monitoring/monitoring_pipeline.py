"""Monitoring workflow execution and statistical test computation."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .artifact_completeness import write_artifact_completeness_artifacts
from .config import MonitoringPerformanceConfig, WorkspaceConfig
from .file_cache import read_csv_cached
from .monitoring_metrics import (
    bad_rate_by_band_mae,
    binary_ks,
    build_data_quality_summary,
    build_feature_drift_summary,
    build_label_evaluation_frame,
    categorical_psi,
    hosmer_lemeshow_p_value,
    numeric_series,
    score_psi,
    summary_metric_value,
)
from .monitoring_reporting import write_monitoring_artifacts
from .monitoring_run_config import (
    MonitoringRunConfig,
    build_dataset_provenance,
    build_monitoring_run_config,
)
from .registry import (
    DatasetAsset,
    InputContractResult,
    ModelBundle,
    read_dataset,
    resolve_prediction_column_name,
    validate_input_contract,
)
from .run_history import record_run_index
from .support_tables import build_support_tables
from .telemetry import RunTelemetry
from .thresholds import ThresholdRecord, records_by_test_id


@dataclass(slots=True)
class BundleExecutionError(RuntimeError):
    stage: str
    detail: str
    command: str | None = None
    stdout: str | None = None
    stderr: str | None = None


@dataclass(slots=True)
class MonitoringTestResult:
    test_id: str
    label: str
    category: str
    operator: str
    threshold_value: float | None
    observed_value: float | None
    status: str
    detail: str


@dataclass(slots=True)
class MonitoringRunResult:
    run_id: str
    started_at: datetime
    model_bundle: ModelBundle
    dataset: DatasetAsset
    run_root: Path
    status: str
    score_column: str | None
    labels_available: bool
    segment_column: str | None
    contract: InputContractResult
    test_results: list[MonitoringTestResult]
    support_tables: dict[str, pd.DataFrame]
    artifacts: dict[str, Path | None] = field(default_factory=dict)
    error_message: str | None = None
    scoring_output_root: Path | None = None
    failure_stage: str | None = None
    failure_context: dict[str, Any] = field(default_factory=dict)
    reviewer_notes: dict[str, str] = field(default_factory=dict)
    reviewer_exceptions: dict[str, dict[str, str]] = field(default_factory=dict)
    run_config: MonitoringRunConfig | None = None
    dataset_provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def results_frame(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(result) for result in self.test_results)

    @property
    def pass_count(self) -> int:
        return sum(result.status == "pass" for result in self.test_results)

    @property
    def fail_count(self) -> int:
        return sum(result.status == "fail" for result in self.test_results)

    @property
    def na_count(self) -> int:
        return sum(result.status == "na" for result in self.test_results)


@dataclass(slots=True)
class ScoringExecutionOutput:
    output_root: Path
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    bundle_artifacts: dict[str, Any]


@dataclass(slots=True)
class ScoringRuntimeOptions:
    disable_individual_visual_exports: bool = False
    outcome_dataset_path: Path | None = None
    outcome_join_columns: list[str] = field(default_factory=list)
    artifact_profile: str = "full"
    performance: MonitoringPerformanceConfig = field(default_factory=MonitoringPerformanceConfig)


def execute_monitoring_run(
    *,
    bundle: ModelBundle,
    dataset: DatasetAsset,
    workspace: WorkspaceConfig,
    thresholds: list[ThresholdRecord],
    segment_column: str | None = None,
    scoring_options: ScoringRuntimeOptions | None = None,
    reviewer_notes: dict[str, str] | None = None,
    reviewer_exceptions: dict[str, dict[str, str]] | None = None,
) -> MonitoringRunResult:
    scoring_options = scoring_options or ScoringRuntimeOptions()
    reviewer_notes = reviewer_notes or {}
    reviewer_exceptions = reviewer_exceptions or {}
    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    run_root = workspace.runs_root / f"{run_id}__{bundle.bundle_id}__{dataset.dataset_id}"
    run_root.mkdir(parents=True, exist_ok=True)
    telemetry = RunTelemetry()

    with telemetry.stage("load_dataset", detail=str(dataset.path)):
        raw_dataframe = read_dataset(dataset.path)
    with telemetry.stage("outcome_join", detail=str(scoring_options.outcome_dataset_path or "")):
        raw_dataframe, outcome_join_summary = _join_outcome_data(
            raw_dataframe=raw_dataframe,
            bundle=bundle,
            outcome_dataset_path=scoring_options.outcome_dataset_path,
            requested_join_columns=scoring_options.outcome_join_columns,
        )
    with telemetry.stage("validate_contract"):
        contract = validate_input_contract(bundle, raw_dataframe, segment_column=segment_column)

    run_config = build_monitoring_run_config(
        run_id=run_id,
        started_at=started_at,
        bundle=bundle,
        dataset=dataset,
        raw_dataframe=raw_dataframe,
        thresholds=thresholds,
        artifact_profile=scoring_options.artifact_profile,
        disable_individual_visual_exports=scoring_options.disable_individual_visual_exports,
        outcome_dataset_path=scoring_options.outcome_dataset_path,
        outcome_join_columns=scoring_options.outcome_join_columns,
        segment_column=segment_column,
        performance=scoring_options.performance,
    )
    dataset_provenance = build_dataset_provenance(
        dataset=dataset,
        raw_dataframe=raw_dataframe,
        run_config=run_config,
    )

    threshold_map = records_by_test_id(thresholds)
    test_results: list[MonitoringTestResult]
    support_tables: dict[str, pd.DataFrame]
    score_column: str | None = bundle.reference_score_column
    error_message: str | None = None
    scoring_output_root: Path | None = None
    labels_available = False
    status = "completed"
    failure_stage: str | None = None
    failure_context: dict[str, Any] = {}

    if contract.missing_required_columns:
        status = "contract_failed"
        failure_stage = "input_contract"
        failure_context = {
            "failing_stage": "input_contract",
            "detail": "Missing required raw input columns blocked scoring.",
            "missing_required_columns": ", ".join(contract.missing_required_columns),
        }
        test_results = _build_contract_failure_results(threshold_map, contract)
        with telemetry.stage("load_reference"):
            reference_raw = _load_reference_raw(bundle)
            reference_predictions = _load_reference_predictions(bundle)
        with telemetry.stage("build_support_tables"):
            support_tables = build_support_tables(
                raw_dataframe=raw_dataframe,
                current_predictions=None,
                reference_raw=reference_raw,
                reference_predictions=reference_predictions,
                bundle=bundle,
                score_column=None,
                actual_column=contract.resolved_label_column,
                segment_column=segment_column if contract.segment_available else None,
                run_started_at=started_at,
            )
    else:
        try:
            with telemetry.stage("score_bundle"):
                scoring_output = run_bundle_scoring(
                    bundle=bundle,
                    scoring_root=run_root / "scoring_bundle",
                    raw_dataframe=raw_dataframe,
                    contract=contract,
                    scoring_options=scoring_options,
                )
            scoring_output_root = scoring_output.output_root
            current_predictions = scoring_output.predictions
            score_column = resolve_prediction_column_name(
                current_predictions.columns.astype(str).tolist(),
                bundle.target_mode,
            )
            if score_column is None:
                raise ValueError("Scoring output does not expose a supported prediction column.")

            with telemetry.stage("load_reference"):
                reference_raw = _load_reference_raw(bundle)
                reference_predictions = _load_reference_predictions(bundle)
            actual_column = _resolve_actual_column(current_predictions, bundle)
            applied_segment_column = segment_column if contract.segment_available else None
            with telemetry.stage("build_support_tables"):
                support_tables = build_support_tables(
                    raw_dataframe=raw_dataframe,
                    current_predictions=current_predictions,
                    reference_raw=reference_raw,
                    reference_predictions=reference_predictions,
                    bundle=bundle,
                    score_column=score_column,
                    actual_column=actual_column,
                    segment_column=applied_segment_column,
                    run_started_at=started_at,
                )
            labels_available = _labels_available_for_metrics(
                current_predictions=current_predictions,
                score_column=score_column,
                actual_column=actual_column,
            )
            with telemetry.stage("evaluate_tests"):
                test_results = _evaluate_tests(
                    threshold_map=threshold_map,
                    bundle=bundle,
                    contract=contract,
                    raw_dataframe=raw_dataframe,
                    current_predictions=current_predictions,
                    reference_raw=reference_raw,
                    reference_predictions=reference_predictions,
                    score_column=score_column,
                    actual_column=actual_column,
                    segment_column=applied_segment_column,
                    run_started_at=started_at,
                )
        except Exception as exc:
            status = "execution_failed"
            error_message, failure_stage, failure_context = _extract_failure_details(exc)
            test_results = _build_execution_failure_results(
                threshold_map=threshold_map,
                contract=contract,
                error_message=error_message,
            )
            with telemetry.stage("load_reference"):
                reference_raw = _load_reference_raw(bundle)
                reference_predictions = _load_reference_predictions(bundle)
            with telemetry.stage("build_support_tables"):
                support_tables = build_support_tables(
                    raw_dataframe=raw_dataframe,
                    current_predictions=None,
                    reference_raw=reference_raw,
                    reference_predictions=reference_predictions,
                    bundle=bundle,
                    score_column=score_column,
                    actual_column=contract.resolved_label_column,
                    segment_column=segment_column if contract.segment_available else None,
                    run_started_at=started_at,
                )

    if not outcome_join_summary.empty:
        support_tables["outcome_join_summary"] = outcome_join_summary

    result = MonitoringRunResult(
        run_id=run_id,
        started_at=started_at,
        model_bundle=bundle,
        dataset=dataset,
        run_root=run_root,
        status=status,
        score_column=score_column,
        labels_available=labels_available and status == "completed",
        segment_column=segment_column if contract.segment_available else None,
        contract=contract,
        test_results=test_results,
        support_tables=support_tables,
        error_message=error_message,
        scoring_output_root=scoring_output_root,
        failure_stage=failure_stage,
        failure_context=failure_context,
        reviewer_notes=reviewer_notes,
        reviewer_exceptions=reviewer_exceptions,
        run_config=run_config,
        dataset_provenance=dataset_provenance,
    )
    with telemetry.stage("write_artifacts", detail=scoring_options.artifact_profile):
        result.artifacts = write_monitoring_artifacts(
            result=result,
            thresholds=thresholds,
            raw_dataframe=raw_dataframe,
            artifact_profile=scoring_options.artifact_profile,
            performance=scoring_options.performance,
        )
    with telemetry.stage("record_run_index"):
        record_run_index(workspace, result)
    _persist_telemetry_artifact(result, telemetry)
    return result


def _persist_telemetry_artifact(
    result: MonitoringRunResult,
    telemetry: RunTelemetry,
) -> None:
    telemetry_path = telemetry.write_jsonl(result.run_root / "run_events.jsonl")
    step_manifest_path = result.run_root / "monitoring_step_manifest.json"
    run_debug_trace_path = result.run_root / "run_debug_trace.json"
    _write_json(step_manifest_path, _build_step_manifest_payload(result, telemetry))
    _write_json(run_debug_trace_path, _build_run_debug_trace_payload(result, telemetry))
    result.artifacts["run_events"] = telemetry_path
    result.artifacts["step_manifest"] = step_manifest_path
    result.artifacts["run_debug_trace"] = run_debug_trace_path
    diagnostic_export_path = result.run_root / "diagnostic_export.zip"
    _build_diagnostic_export_package(
        result=result,
        package_path=diagnostic_export_path,
        extra_paths=[telemetry_path, step_manifest_path, run_debug_trace_path],
    )
    result.artifacts["diagnostic_export"] = diagnostic_export_path
    manifest_path = result.artifacts.get("manifest")
    if manifest_path is not None and Path(manifest_path).exists():
        payload = _read_json(Path(manifest_path))
        payload["run_events"] = str(telemetry_path)
        payload["step_manifest"] = str(step_manifest_path)
        payload["run_debug_trace"] = str(run_debug_trace_path)
        payload["diagnostic_export"] = str(diagnostic_export_path)
        completeness_csv_path = result.run_root / "artifact_completeness.csv"
        completeness_json_path = result.run_root / "artifact_completeness.json"
        payload["artifact_completeness_csv"] = str(completeness_csv_path)
        payload["artifact_completeness_json"] = str(completeness_json_path)
        completeness_frame = write_artifact_completeness_artifacts(
            manifest=payload,
            csv_path=completeness_csv_path,
            json_path=completeness_json_path,
        )
        result.support_tables["artifact_completeness"] = completeness_frame
        result.artifacts["artifact_completeness_csv"] = completeness_csv_path
        result.artifacts["artifact_completeness_json"] = completeness_json_path
        _build_diagnostic_export_package(
            result=result,
            package_path=diagnostic_export_path,
            extra_paths=[telemetry_path, step_manifest_path, run_debug_trace_path],
        )
        missing_count = int((completeness_frame["status"] == "missing").sum())
        payload["artifact_completeness_status"] = "complete" if missing_count == 0 else "incomplete"
        generated_artifacts = dict(payload.get("generated_artifacts", {}))
        for key in (
            "run_events",
            "step_manifest",
            "run_debug_trace",
            "diagnostic_export",
            "artifact_completeness_csv",
            "artifact_completeness_json",
        ):
            value = payload.get(key)
            if isinstance(value, str) and Path(value).exists():
                generated_artifacts[key] = value
        payload["generated_artifacts"] = generated_artifacts
        with Path(manifest_path).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
    package_path = result.artifacts.get("reviewer_package")
    if package_path is not None and Path(package_path).exists():
        with zipfile.ZipFile(package_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in (
                telemetry_path,
                step_manifest_path,
                run_debug_trace_path,
                result.artifacts.get("artifact_completeness_csv"),
                result.artifacts.get("artifact_completeness_json"),
                diagnostic_export_path,
            ):
                if path is None or not Path(path).exists():
                    continue
                archive.write(path, arcname=path.name)


def _build_step_manifest_payload(
    result: MonitoringRunResult,
    telemetry: RunTelemetry,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_id": result.run_id,
        "status": result.status,
        "steps": [
            {
                "order": index,
                "name": event.stage,
                "status": event.status,
                "duration_seconds": event.duration_seconds,
                "started_at_utc": event.started_at_utc,
                "detail": event.detail,
            }
            for index, event in enumerate(telemetry.events, start=1)
        ],
    }


def _build_diagnostic_export_package(
    *,
    result: MonitoringRunResult,
    package_path: Path,
    extra_paths: list[Path],
) -> None:
    keys = [
        "manifest",
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
        "artifact_completeness_csv",
        "artifact_completeness_json",
    ]
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key in keys:
            path = result.artifacts.get(key)
            if path is None or not Path(path).exists():
                continue
            archive.write(Path(path), arcname=Path(path).name)
        for path in extra_paths:
            if path.exists():
                archive.write(path, arcname=path.name)


def _build_run_debug_trace_payload(
    result: MonitoringRunResult,
    telemetry: RunTelemetry,
) -> dict[str, Any]:
    total_seconds = round(sum(event.duration_seconds for event in telemetry.events), 6)
    return {
        "schema_version": "1.0",
        "run_id": result.run_id,
        "status": result.status,
        "failure_stage": result.failure_stage,
        "error_message": result.error_message,
        "failure_context": result.failure_context,
        "step_count": len(telemetry.events),
        "total_step_seconds": total_seconds,
        "events": [asdict(event) for event in telemetry.events],
        "summary": {
            "model_name": result.model_bundle.display_name,
            "model_version": result.model_bundle.model_version,
            "dataset_name": result.dataset.name,
            "score_column": result.score_column,
            "labels_available": result.labels_available,
            "segment_column": result.segment_column,
            "pass_count": result.pass_count,
            "fail_count": result.fail_count,
            "na_count": result.na_count,
        },
    }


def run_bundle_scoring(
    *,
    bundle: ModelBundle,
    scoring_root: Path,
    raw_dataframe: pd.DataFrame,
    contract: InputContractResult,
    scoring_options: ScoringRuntimeOptions,
) -> ScoringExecutionOutput:
    if bundle.bundle_paths.generated_runner_path is None:
        raise BundleExecutionError(
            stage="bundle_compatibility",
            detail="Bundle is missing generated_run.py and cannot rerun raw-data scoring.",
        )

    scoring_root.mkdir(parents=True, exist_ok=True)
    try:
        prepared_input_path = _prepare_bundle_input_dataset(
            bundle=bundle,
            raw_dataframe=raw_dataframe,
            contract=contract,
            scoring_root=scoring_root,
        )
        prepared_config_path = _prepare_scoring_config(
            bundle=bundle,
            scoring_root=scoring_root,
            scoring_options=scoring_options,
        )
    except Exception as exc:  # pragma: no cover - defensive wrapper
        raise BundleExecutionError(
            stage="scoring_preparation",
            detail=str(exc),
        ) from exc
    command = [
        sys.executable,
        str(bundle.bundle_paths.generated_runner_path),
        "--config",
        str(prepared_config_path),
        "--input",
        str(prepared_input_path),
        "--output-root",
        str(scoring_root),
    ]
    completed = subprocess.run(
        command,
        cwd=bundle.bundle_paths.root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BundleExecutionError(
            stage="generated_runner",
            detail=detail or "Bundle rerun failed with a non-zero exit code.",
            command=" ".join(command),
            stdout=completed.stdout.strip() or None,
            stderr=completed.stderr.strip() or None,
        )

    output_root = _extract_scoring_output_root(
        scoring_root,
        completed.stdout,
        command=" ".join(command),
    )
    predictions_path = output_root / "predictions.csv"
    metrics_path = output_root / "metrics.json"
    manifest_path = output_root / "artifact_manifest.json"
    if not predictions_path.exists():
        raise BundleExecutionError(
            stage="prediction_output",
            detail=f"Expected scored predictions at {predictions_path}.",
            command=" ".join(command),
            stdout=completed.stdout.strip() or None,
            stderr=completed.stderr.strip() or None,
        )

    return ScoringExecutionOutput(
        output_root=output_root,
        predictions=pd.read_csv(predictions_path),
        metrics=_read_json(metrics_path) if metrics_path.exists() else {},
        bundle_artifacts=_read_json(manifest_path) if manifest_path.exists() else {},
    )


def _join_outcome_data(
    *,
    raw_dataframe: pd.DataFrame,
    bundle: ModelBundle,
    outcome_dataset_path: Path | None,
    requested_join_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if outcome_dataset_path is None:
        return raw_dataframe, pd.DataFrame()

    summary_base: dict[str, object] = {
        "outcome_dataset_path": str(outcome_dataset_path),
        "input_rows": len(raw_dataframe),
        "status": "not_applied",
    }
    try:
        outcome_dataframe = read_dataset(outcome_dataset_path)
    except Exception as exc:
        return raw_dataframe, pd.DataFrame(
            [
                {
                    **summary_base,
                    "outcome_rows": None,
                    "matched_rows": None,
                    "join_columns": "",
                    "label_columns": "",
                    "detail": f"Outcome file could not be read: {exc}",
                }
            ]
        )

    join_columns = _resolve_outcome_join_columns(
        raw_dataframe=raw_dataframe,
        outcome_dataframe=outcome_dataframe,
        bundle=bundle,
        requested_join_columns=requested_join_columns,
    )
    label_columns = _resolve_outcome_label_columns(
        outcome_dataframe=outcome_dataframe,
        bundle=bundle,
    )
    if not join_columns or not label_columns:
        return raw_dataframe, pd.DataFrame(
            [
                {
                    **summary_base,
                    "outcome_rows": len(outcome_dataframe),
                    "matched_rows": None,
                    "join_columns": ", ".join(join_columns),
                    "label_columns": ", ".join(label_columns),
                    "detail": "Outcome join skipped because no valid join keys or label columns were found.",
                }
            ]
        )

    outcome_subset = outcome_dataframe[join_columns + label_columns].drop_duplicates(
        subset=join_columns,
        keep="last",
    )
    merged = raw_dataframe.merge(
        outcome_subset,
        on=join_columns,
        how="left",
        suffixes=("", "__outcome"),
        indicator=True,
    )
    matched_rows = int(merged["_merge"].eq("both").sum())
    for column in label_columns:
        outcome_column = f"{column}__outcome"
        if column == bundle.label_output_column and bundle.label_source_column:
            if bundle.label_source_column in merged.columns:
                merged[bundle.label_source_column] = merged[bundle.label_source_column].combine_first(
                    merged[column]
                )
            else:
                merged[bundle.label_source_column] = merged[column]
        elif outcome_column in merged.columns:
            merged[column] = merged[column].combine_first(merged[outcome_column])
            merged = merged.drop(columns=[outcome_column])
    merged = merged.drop(columns=["_merge"])

    return merged, pd.DataFrame(
        [
            {
                **summary_base,
                "status": "applied",
                "outcome_rows": len(outcome_dataframe),
                "matched_rows": matched_rows,
                "join_columns": ", ".join(join_columns),
                "label_columns": ", ".join(label_columns),
                "detail": "Outcome labels were joined into the scoring input before validation.",
            }
        ]
    )


def _resolve_outcome_join_columns(
    *,
    raw_dataframe: pd.DataFrame,
    outcome_dataframe: pd.DataFrame,
    bundle: ModelBundle,
    requested_join_columns: list[str],
) -> list[str]:
    common_columns = set(raw_dataframe.columns).intersection(outcome_dataframe.columns)
    requested = [column for column in requested_join_columns if column in common_columns]
    if requested:
        return requested

    identifier_columns = [column for column in bundle.identifier_columns if column in common_columns]
    if identifier_columns:
        return identifier_columns
    return [column for column in bundle.date_columns if column in common_columns]


def _resolve_outcome_label_columns(
    *,
    outcome_dataframe: pd.DataFrame,
    bundle: ModelBundle,
) -> list[str]:
    columns: list[str] = []
    for candidate in (bundle.label_source_column, bundle.label_output_column):
        if candidate and candidate in outcome_dataframe.columns and candidate not in columns:
            columns.append(candidate)
    return columns


def _extract_scoring_output_root(
    scoring_root: Path,
    stdout: str,
    *,
    command: str | None = None,
) -> Path:
    match = re.search(r"Artifacts written to (.+)", stdout)
    if match:
        candidate = Path(match.group(1).strip())
        if candidate.exists():
            return candidate

    candidate_dirs = [path for path in scoring_root.iterdir() if path.is_dir()]
    if not candidate_dirs:
        raise BundleExecutionError(
            stage="scoring_output_root",
            detail=f"No scoring output directory found beneath {scoring_root}.",
            command=command,
            stdout=stdout.strip() or None,
        )
    return max(candidate_dirs, key=lambda path: path.stat().st_mtime)


def _evaluate_tests(
    *,
    threshold_map: dict[str, ThresholdRecord],
    bundle: ModelBundle,
    contract: InputContractResult,
    raw_dataframe: pd.DataFrame,
    current_predictions: pd.DataFrame,
    reference_raw: pd.DataFrame | None,
    reference_predictions: pd.DataFrame | None,
    score_column: str,
    actual_column: str | None,
    segment_column: str | None,
    run_started_at: datetime,
) -> list[MonitoringTestResult]:
    results: list[MonitoringTestResult] = []
    current_scores = numeric_series(current_predictions[score_column])
    reference_scores = (
        numeric_series(reference_predictions[bundle.reference_score_column])
        if reference_predictions is not None
        and bundle.reference_score_column is not None
        and bundle.reference_score_column in reference_predictions.columns
        else None
    )
    data_quality = build_data_quality_summary(
        raw_dataframe=raw_dataframe,
        reference_raw=reference_raw,
        bundle=bundle,
        run_started_at=run_started_at,
    )
    feature_drift = build_feature_drift_summary(
        reference_raw=reference_raw,
        raw_dataframe=raw_dataframe,
        bundle=bundle,
    )

    results.append(
        _apply_threshold(
            threshold_map["required_columns_missing_count"],
            observed_value=float(len(contract.missing_required_columns)),
            detail=(
                "All required input columns were present."
                if not contract.missing_required_columns
                else "Missing columns: " + ", ".join(contract.missing_required_columns)
            ),
        )
    )

    row_count_ratio = None
    if reference_raw is not None and len(reference_raw):
        row_count_ratio = float(len(raw_dataframe) / len(reference_raw))
    results.append(
        _apply_threshold(
            threshold_map["row_count_ratio"],
            observed_value=row_count_ratio,
            detail="Monitoring rows divided by reference input rows.",
        )
    )
    results.append(
        _apply_threshold(
            threshold_map["row_count_absolute"],
            observed_value=summary_metric_value(data_quality, "row_count_absolute"),
            detail="Monitoring dataset row count.",
        )
    )

    max_missingness_pct = 0.0
    if contract.required_input_columns:
        max_missingness_pct = float(
            raw_dataframe[contract.required_input_columns].isna().mean().max() * 100.0
        )
    results.append(
        _apply_threshold(
            threshold_map["max_feature_missingness_pct"],
            observed_value=max_missingness_pct,
            detail="Highest missingness percentage across required raw input columns.",
        )
    )
    for test_id, detail in (
        ("duplicate_identifier_count", "Rows duplicated across identifier columns."),
        ("identifier_null_rate_pct", "Highest null percentage across identifier columns."),
        ("invalid_date_count", "Non-empty date values that could not be parsed."),
        ("stale_as_of_date_days", "Days since the latest monitoring as-of date."),
        ("unexpected_category_count", "Unexpected categorical feature values versus reference input."),
        ("numeric_range_violation_count", "Numeric feature values outside reference min/max ranges."),
    ):
        results.append(
            _apply_threshold(
                threshold_map[test_id],
                observed_value=summary_metric_value(data_quality, test_id),
                detail=detail,
            )
        )

    results.append(
        _apply_threshold(
            threshold_map["score_psi"],
            observed_value=score_psi(reference_scores, current_scores),
            detail="Population stability index on the selected score column.",
        )
    )

    score_ks_p_value = None
    if reference_scores is not None and len(reference_scores) and len(current_scores):
        score_ks_p_value = float(ks_2samp(reference_scores, current_scores).pvalue)
    results.append(
        _apply_threshold(
            threshold_map["score_ks_p_value"],
            observed_value=score_ks_p_value,
            detail="Two-sample KS p-value on the selected score column.",
        )
    )

    segment_psi = None
    if (
        segment_column
        and reference_raw is not None
        and segment_column in raw_dataframe.columns
        and segment_column in reference_raw.columns
    ):
        segment_psi = categorical_psi(reference_raw[segment_column], raw_dataframe[segment_column])
    results.append(
        _apply_threshold(
            threshold_map["segment_psi"],
            observed_value=segment_psi,
            detail="Population stability index across the selected segment column.",
        )
    )

    max_feature_psi = None
    min_numeric_ks_p_value = None
    if not feature_drift.empty:
        psi_values = pd.to_numeric(feature_drift["psi"], errors="coerce").dropna()
        ks_values = pd.to_numeric(feature_drift["ks_p_value"], errors="coerce").dropna()
        if not psi_values.empty:
            max_feature_psi = float(psi_values.max())
        if not ks_values.empty:
            min_numeric_ks_p_value = float(ks_values.min())
    results.extend(
        [
            _apply_threshold(
                threshold_map["max_feature_psi"],
                observed_value=max_feature_psi,
                detail="Highest feature-level PSI across raw input features.",
            ),
            _apply_threshold(
                threshold_map["min_numeric_feature_ks_p_value"],
                observed_value=min_numeric_ks_p_value,
                detail="Lowest numeric-feature two-sample KS p-value.",
            ),
        ]
    )

    label_frame = build_label_evaluation_frame(
        current_predictions=current_predictions,
        score_column=score_column,
        actual_column=actual_column,
    )
    if label_frame.empty:
        results.extend(
            _na_result(threshold_map[test_id], "Labels are not available for this monitoring run.")
            for test_id in (
                "auc",
                "ks_statistic",
                "gini",
                "brier_score",
                "mean_absolute_error",
                "hosmer_lemeshow_p_value",
                "precision_at_threshold",
                "recall_at_threshold",
                "bad_rate_by_band_mae",
            )
        )
        return results

    actual_values = label_frame["actual"]
    scored_values = label_frame["score"]
    if bundle.target_mode != "binary":
        results.extend(
            _na_result(threshold_map[test_id], "This test is only applicable to binary targets.")
            for test_id in (
                "auc",
                "ks_statistic",
                "gini",
                "brier_score",
                "hosmer_lemeshow_p_value",
                "precision_at_threshold",
                "recall_at_threshold",
                "bad_rate_by_band_mae",
            )
        )
        results.append(
            _apply_threshold(
                threshold_map["mean_absolute_error"],
                observed_value=float(mean_absolute_error(actual_values, scored_values)),
                detail="Mean absolute error on the monitoring population.",
            )
        )
        return results

    actual_binary = actual_values.astype(int)
    auc_value = _safe_binary_metric(lambda: float(roc_auc_score(actual_binary, scored_values)))
    ks_value = binary_ks(actual_binary, scored_values)
    gini_value = float(2 * auc_value - 1) if auc_value is not None else None
    brier_value = _safe_binary_metric(
        lambda: float(brier_score_loss(actual_binary, scored_values))
    )
    mae_value = float(mean_absolute_error(actual_binary, scored_values))
    hl_p_value = hosmer_lemeshow_p_value(actual_binary, scored_values)

    threshold = bundle.model_threshold or 0.5
    predicted_class = (scored_values >= threshold).astype(int)
    precision_value = _safe_binary_metric(
        lambda: float(precision_score(actual_binary, predicted_class, zero_division=0))
    )
    recall_value = _safe_binary_metric(
        lambda: float(recall_score(actual_binary, predicted_class, zero_division=0))
    )
    band_mae = bad_rate_by_band_mae(actual_binary, scored_values)

    results.extend(
        [
            _apply_threshold(
                threshold_map["auc"],
                observed_value=auc_value,
                detail="Area under the ROC curve on the monitoring population.",
            ),
            _apply_threshold(
                threshold_map["ks_statistic"],
                observed_value=ks_value,
                detail="Event vs non-event score separation on the monitoring population.",
            ),
            _apply_threshold(
                threshold_map["gini"],
                observed_value=gini_value,
                detail="Gini coefficient derived from monitoring AUC.",
            ),
            _apply_threshold(
                threshold_map["brier_score"],
                observed_value=brier_value,
                detail="Brier score on the monitoring population.",
            ),
            _apply_threshold(
                threshold_map["mean_absolute_error"],
                observed_value=mae_value,
                detail="Mean absolute error between actual outcomes and predicted probabilities.",
            ),
            _apply_threshold(
                threshold_map["hosmer_lemeshow_p_value"],
                observed_value=hl_p_value,
                detail="Hosmer-Lemeshow p-value using score deciles.",
            ),
            _apply_threshold(
                threshold_map["precision_at_threshold"],
                observed_value=precision_value,
                detail=f"Precision at score threshold {threshold:.4f}.",
            ),
            _apply_threshold(
                threshold_map["recall_at_threshold"],
                observed_value=recall_value,
                detail=f"Recall at score threshold {threshold:.4f}.",
            ),
            _apply_threshold(
                threshold_map["bad_rate_by_band_mae"],
                observed_value=band_mae,
                detail="Mean absolute error between predicted and observed bad rate by score band.",
            ),
        ]
    )
    return results


def _build_contract_failure_results(
    threshold_map: dict[str, ThresholdRecord],
    contract: InputContractResult,
) -> list[MonitoringTestResult]:
    results = [
        _apply_threshold(
            threshold_map["required_columns_missing_count"],
            observed_value=float(len(contract.missing_required_columns)),
            detail="Missing columns: " + ", ".join(contract.missing_required_columns),
        )
    ]
    for test_id, threshold in threshold_map.items():
        if test_id == "required_columns_missing_count":
            continue
        results.append(
            _na_result(threshold, "Run stopped before scoring because the input contract failed.")
        )
    return results


def _build_execution_failure_results(
    *,
    threshold_map: dict[str, ThresholdRecord],
    contract: InputContractResult,
    error_message: str,
) -> list[MonitoringTestResult]:
    results = [
        _apply_threshold(
            threshold_map["required_columns_missing_count"],
            observed_value=float(len(contract.missing_required_columns)),
            detail="Input contract passed before execution started.",
        )
    ]
    for test_id, threshold in threshold_map.items():
        if test_id == "required_columns_missing_count":
            continue
        results.append(_na_result(threshold, f"Bundle execution failed: {error_message}"))
    return results


def _prepare_scoring_config(
    *,
    bundle: ModelBundle,
    scoring_root: Path,
    scoring_options: ScoringRuntimeOptions,
) -> Path:
    config_payload = _read_json(bundle.bundle_paths.run_config_path)
    execution_payload = dict(config_payload.get("execution", {}))
    execution_payload["mode"] = "score_existing_model"
    execution_payload["existing_model_path"] = str(bundle.bundle_paths.model_path.resolve())
    execution_payload["existing_config_path"] = str(bundle.bundle_paths.run_config_path.resolve())
    config_payload["execution"] = execution_payload
    diagnostics_payload = dict(config_payload.get("diagnostics", {}))
    if scoring_options.disable_individual_visual_exports:
        diagnostics_payload["interactive_visualizations"] = False
        diagnostics_payload["static_image_exports"] = False
    config_payload["diagnostics"] = diagnostics_payload

    config_path = scoring_root / "monitoring_scoring_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)
    return config_path


def _prepare_bundle_input_dataset(
    *,
    bundle: ModelBundle,
    raw_dataframe: pd.DataFrame,
    contract: InputContractResult,
    scoring_root: Path,
) -> Path:
    prepared = raw_dataframe.copy(deep=True)
    for column in contract.missing_optional_columns:
        if column == bundle.label_source_column:
            prepared[column] = pd.NA
        else:
            prepared[column] = "__monitoring_placeholder__"

    input_path = scoring_root / "monitoring_input.csv"
    prepared.to_csv(input_path, index=False)
    return input_path


def _apply_threshold(
    threshold: ThresholdRecord,
    *,
    observed_value: float | None,
    detail: str,
) -> MonitoringTestResult:
    if not threshold.enabled:
        return _na_result(threshold, "Threshold disabled for this model.")
    if observed_value is None:
        return _na_result(threshold, detail)

    status = "pass"
    if threshold.value is not None:
        if threshold.operator == "<=" and observed_value > threshold.value:
            status = "fail"
        elif threshold.operator == ">=" and observed_value < threshold.value:
            status = "fail"

    return MonitoringTestResult(
        test_id=threshold.test_id,
        label=threshold.label,
        category=threshold.category,
        operator=threshold.operator,
        threshold_value=threshold.value,
        observed_value=observed_value,
        status=status,
        detail=detail,
    )


def _na_result(threshold: ThresholdRecord, detail: str) -> MonitoringTestResult:
    return MonitoringTestResult(
        test_id=threshold.test_id,
        label=threshold.label,
        category=threshold.category,
        operator=threshold.operator,
        threshold_value=threshold.value,
        observed_value=None,
        status="na",
        detail=detail,
    )


def _load_reference_raw(bundle: ModelBundle) -> pd.DataFrame | None:
    path = bundle.bundle_paths.reference_input_path
    if path is None or not path.exists():
        return None
    return read_csv_cached(path)


def _load_reference_predictions(bundle: ModelBundle) -> pd.DataFrame | None:
    path = bundle.bundle_paths.reference_predictions_path
    if path is None or not path.exists():
        return None
    return read_csv_cached(path)


def _resolve_actual_column(current_predictions: pd.DataFrame, bundle: ModelBundle) -> str | None:
    for candidate in (bundle.label_output_column, bundle.label_source_column):
        if candidate and candidate in current_predictions.columns:
            return candidate
    return None


def _labels_available_for_metrics(
    *,
    current_predictions: pd.DataFrame,
    score_column: str,
    actual_column: str | None,
) -> bool:
    return not build_label_evaluation_frame(
        current_predictions=current_predictions,
        score_column=score_column,
        actual_column=actual_column,
    ).empty


def _safe_binary_metric(metric) -> float | None:
    try:
        value = metric()
    except ValueError:
        return None
    if value is None or pd.isna(value):
        return None
    return float(value)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _extract_failure_details(exc: Exception) -> tuple[str, str | None, dict[str, Any]]:
    if isinstance(exc, BundleExecutionError):
        context = {
            "failing_stage": exc.stage,
            "detail": exc.detail,
        }
        if exc.command:
            context["command"] = exc.command
        if exc.stdout:
            context["stdout_excerpt"] = exc.stdout[:4000]
        if exc.stderr:
            context["stderr_excerpt"] = exc.stderr[:4000]
        return exc.detail, exc.stage, context
    return str(exc), "execution", {"failing_stage": "execution", "detail": str(exc)}
