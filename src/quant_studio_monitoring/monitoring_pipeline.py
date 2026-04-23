"""Monitoring workflow execution and statistical test computation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, ks_2samp
from sklearn.metrics import brier_score_loss, mean_absolute_error, precision_score, recall_score, roc_auc_score

from .config import WorkspaceConfig
from .monitoring_reporting import write_monitoring_artifacts
from .registry import DatasetAsset, InputContractResult, ModelBundle, read_dataset, resolve_prediction_column_name, validate_input_contract
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


def execute_monitoring_run(
    *,
    bundle: ModelBundle,
    dataset: DatasetAsset,
    workspace: WorkspaceConfig,
    thresholds: list[ThresholdRecord],
    segment_column: str | None = None,
    scoring_options: ScoringRuntimeOptions | None = None,
) -> MonitoringRunResult:
    scoring_options = scoring_options or ScoringRuntimeOptions()
    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    run_root = workspace.runs_root / f"{run_id}__{bundle.bundle_id}__{dataset.dataset_id}"
    run_root.mkdir(parents=True, exist_ok=True)

    raw_dataframe = read_dataset(dataset.path)
    contract = validate_input_contract(bundle, raw_dataframe, segment_column=segment_column)

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
        support_tables = _build_support_tables(
            raw_dataframe=raw_dataframe,
            current_predictions=None,
            reference_raw=_load_reference_raw(bundle),
            reference_predictions=_load_reference_predictions(bundle),
            bundle=bundle,
            score_column=None,
            actual_column=contract.resolved_label_column,
            segment_column=segment_column if contract.segment_available else None,
        )
    else:
        try:
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

            reference_raw = _load_reference_raw(bundle)
            reference_predictions = _load_reference_predictions(bundle)
            actual_column = _resolve_actual_column(current_predictions, bundle)
            applied_segment_column = segment_column if contract.segment_available else None
            support_tables = _build_support_tables(
                raw_dataframe=raw_dataframe,
                current_predictions=current_predictions,
                reference_raw=reference_raw,
                reference_predictions=reference_predictions,
                bundle=bundle,
                score_column=score_column,
                actual_column=actual_column,
                segment_column=applied_segment_column,
            )
            labels_available = _labels_available_for_metrics(
                current_predictions=current_predictions,
                score_column=score_column,
                actual_column=actual_column,
            )
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
            )
        except Exception as exc:
            status = "execution_failed"
            error_message, failure_stage, failure_context = _extract_failure_details(exc)
            test_results = _build_execution_failure_results(
                threshold_map=threshold_map,
                contract=contract,
                error_message=error_message,
            )
            support_tables = _build_support_tables(
                raw_dataframe=raw_dataframe,
                current_predictions=None,
                reference_raw=_load_reference_raw(bundle),
                reference_predictions=_load_reference_predictions(bundle),
                bundle=bundle,
                score_column=score_column,
                actual_column=contract.resolved_label_column,
                segment_column=segment_column if contract.segment_available else None,
            )

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
    )
    result.artifacts = write_monitoring_artifacts(
        result=result,
        thresholds=thresholds,
        raw_dataframe=raw_dataframe,
    )
    return result


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
) -> list[MonitoringTestResult]:
    results: list[MonitoringTestResult] = []
    current_scores = _numeric_series(current_predictions[score_column])
    reference_scores = (
        _numeric_series(reference_predictions[bundle.reference_score_column])
        if reference_predictions is not None
        and bundle.reference_score_column is not None
        and bundle.reference_score_column in reference_predictions.columns
        else None
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

    results.append(
        _apply_threshold(
            threshold_map["score_psi"],
            observed_value=_score_psi(reference_scores, current_scores),
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
        segment_psi = _categorical_psi(reference_raw[segment_column], raw_dataframe[segment_column])
    results.append(
        _apply_threshold(
            threshold_map["segment_psi"],
            observed_value=segment_psi,
            detail="Population stability index across the selected segment column.",
        )
    )

    label_frame = _build_label_evaluation_frame(
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
    ks_value = _binary_ks(actual_binary, scored_values)
    gini_value = float(2 * auc_value - 1) if auc_value is not None else None
    brier_value = _safe_binary_metric(
        lambda: float(brier_score_loss(actual_binary, scored_values))
    )
    mae_value = float(mean_absolute_error(actual_binary, scored_values))
    hl_p_value = _hosmer_lemeshow_p_value(actual_binary, scored_values)

    threshold = bundle.model_threshold or 0.5
    predicted_class = (scored_values >= threshold).astype(int)
    precision_value = _safe_binary_metric(
        lambda: float(precision_score(actual_binary, predicted_class, zero_division=0))
    )
    recall_value = _safe_binary_metric(
        lambda: float(recall_score(actual_binary, predicted_class, zero_division=0))
    )
    band_mae = _bad_rate_by_band_mae(actual_binary, scored_values)

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


def _build_support_tables(
    *,
    raw_dataframe: pd.DataFrame,
    current_predictions: pd.DataFrame | None,
    reference_raw: pd.DataFrame | None,
    reference_predictions: pd.DataFrame | None,
    bundle: ModelBundle,
    score_column: str | None,
    actual_column: str | None,
    segment_column: str | None,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    tables["model_metadata"] = pd.DataFrame(
        [
            {"field": "model_name", "value": bundle.display_name},
            {"field": "model_version", "value": bundle.model_version},
            {"field": "model_owner", "value": bundle.model_owner},
            {"field": "business_purpose", "value": bundle.business_purpose},
            {"field": "portfolio_name", "value": bundle.portfolio_name},
            {"field": "segment_name", "value": bundle.segment_name},
            {
                "field": "approval_status",
                "value": bundle.monitoring_metadata.approval_status or "n/a",
            },
            {
                "field": "approval_date",
                "value": bundle.monitoring_metadata.approval_date or "n/a",
            },
            {
                "field": "monitoring_notes",
                "value": bundle.monitoring_metadata.monitoring_notes or "n/a",
            },
            {"field": "metadata_source", "value": bundle.metadata_source},
            {"field": "model_type", "value": bundle.model_type},
            {"field": "target_mode", "value": bundle.target_mode},
            {"field": "bundle_path", "value": str(bundle.bundle_paths.root)},
            {"field": "reference_row_count", "value": bundle.reference_row_count},
        ]
    )

    missingness = raw_dataframe.isna().mean().mul(100.0).reset_index()
    missingness.columns = ["column_name", "current_missingness_pct"]
    if reference_raw is not None:
        reference_missingness = reference_raw.isna().mean().mul(100.0).reset_index()
        reference_missingness.columns = ["column_name", "reference_missingness_pct"]
        missingness = missingness.merge(reference_missingness, on="column_name", how="left")
    tables["missingness_summary"] = missingness.sort_values(
        "current_missingness_pct", ascending=False
    ).reset_index(drop=True)

    if (
        score_column is not None
        and current_predictions is not None
        and score_column in current_predictions.columns
    ):
        current_scores = _numeric_series(current_predictions[score_column])
        score_profile = pd.DataFrame(
            [
                {
                    "population": "monitoring",
                    "row_count": len(current_scores),
                    "score_mean": float(current_scores.mean()) if len(current_scores) else None,
                    "score_std": float(current_scores.std()) if len(current_scores) else None,
                    "score_min": float(current_scores.min()) if len(current_scores) else None,
                    "score_max": float(current_scores.max()) if len(current_scores) else None,
                }
            ]
        )
        if (
            reference_predictions is not None
            and bundle.reference_score_column is not None
            and bundle.reference_score_column in reference_predictions.columns
        ):
            reference_scores = _numeric_series(reference_predictions[bundle.reference_score_column])
            score_profile = pd.concat(
                [
                    pd.DataFrame(
                        [
                            {
                                "population": "reference",
                                "row_count": len(reference_scores),
                                "score_mean": float(reference_scores.mean())
                                if len(reference_scores)
                                else None,
                                "score_std": float(reference_scores.std())
                                if len(reference_scores)
                                else None,
                                "score_min": float(reference_scores.min())
                                if len(reference_scores)
                                else None,
                                "score_max": float(reference_scores.max())
                                if len(reference_scores)
                                else None,
                            }
                        ]
                    ),
                    score_profile,
                ],
                ignore_index=True,
            )
        tables["score_profile"] = score_profile

        label_frame = _build_label_evaluation_frame(
            current_predictions=current_predictions,
            score_column=score_column,
            actual_column=actual_column,
        )
        if not label_frame.empty:
            tables["score_band_summary"] = _build_score_band_summary(
                actual_values=label_frame["actual"],
                score_values=label_frame["score"],
            )

    if segment_column and segment_column in raw_dataframe.columns:
        current_segment = _build_segment_share_frame(
            raw_dataframe[segment_column],
            value_name="current_share_pct",
        )
        if reference_raw is not None and segment_column in reference_raw.columns:
            reference_segment = _build_segment_share_frame(
                reference_raw[segment_column],
                value_name="reference_share_pct",
            )
            current_segment = current_segment.merge(
                reference_segment, on="segment_value", how="outer"
            ).fillna(0.0)
        tables["segment_summary"] = current_segment.sort_values(
            "current_share_pct", ascending=False
        ).reset_index(drop=True)

    if current_predictions is not None:
        tables["scored_data_preview"] = current_predictions.head(50).copy(deep=True)
    return tables


def _build_score_band_summary(
    *,
    actual_values: pd.Series,
    score_values: pd.Series,
    band_count: int = 10,
) -> pd.DataFrame:
    quantiles = np.unique(np.quantile(score_values, np.linspace(0.0, 1.0, band_count + 1)))
    if len(quantiles) < 3:
        quantiles = np.linspace(float(score_values.min()), float(score_values.max()), band_count + 1)
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    band = pd.cut(score_values, bins=quantiles, include_lowest=True, duplicates="drop")
    frame = pd.DataFrame({"score": score_values, "actual": actual_values, "band": band})
    summary = (
        frame.groupby("band", observed=False)
        .agg(
            observation_count=("score", "size"),
            average_predicted_score=("score", "mean"),
            observed_rate=("actual", "mean"),
        )
        .reset_index()
    )
    summary["bad_rate_abs_error"] = (
        summary["average_predicted_score"] - summary["observed_rate"]
    ).abs()
    summary["band_label"] = summary["band"].astype(str)
    return summary.drop(columns=["band"])


def _build_segment_share_frame(values: pd.Series, *, value_name: str) -> pd.DataFrame:
    frame = (
        values.fillna("__MISSING__")
        .astype(str)
        .value_counts(dropna=False, normalize=True)
        .mul(100.0)
        .rename(value_name)
        .reset_index()
    )
    frame.columns = ["segment_value", value_name]
    return frame


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
    return pd.read_csv(path)


def _load_reference_predictions(bundle: ModelBundle) -> pd.DataFrame | None:
    path = bundle.bundle_paths.reference_predictions_path
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _resolve_actual_column(current_predictions: pd.DataFrame, bundle: ModelBundle) -> str | None:
    for candidate in (bundle.label_output_column, bundle.label_source_column):
        if candidate and candidate in current_predictions.columns:
            return candidate
    return None


def _build_label_evaluation_frame(
    *,
    current_predictions: pd.DataFrame,
    score_column: str,
    actual_column: str | None,
) -> pd.DataFrame:
    if actual_column is None or actual_column not in current_predictions.columns:
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "score": pd.to_numeric(current_predictions[score_column], errors="coerce"),
            "actual": pd.to_numeric(current_predictions[actual_column], errors="coerce"),
        }
    ).dropna(subset=["score", "actual"])
    return frame.reset_index(drop=True)


def _labels_available_for_metrics(
    *,
    current_predictions: pd.DataFrame,
    score_column: str,
    actual_column: str | None,
) -> bool:
    return not _build_label_evaluation_frame(
        current_predictions=current_predictions,
        score_column=score_column,
        actual_column=actual_column,
    ).empty


def _score_psi(reference_scores: pd.Series | None, current_scores: pd.Series) -> float | None:
    if reference_scores is None or not len(reference_scores) or not len(current_scores):
        return None

    quantiles = np.unique(np.quantile(reference_scores, np.linspace(0.0, 1.0, 11)))
    if len(quantiles) < 3:
        quantiles = np.linspace(float(reference_scores.min()), float(reference_scores.max()), 11)
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    reference_buckets = pd.cut(reference_scores, bins=quantiles, include_lowest=True, duplicates="drop")
    current_buckets = pd.cut(current_scores, bins=quantiles, include_lowest=True, duplicates="drop")
    reference_dist = reference_buckets.value_counts(normalize=True, sort=False)
    current_dist = current_buckets.value_counts(normalize=True, sort=False).reindex(
        reference_dist.index, fill_value=0.0
    )

    epsilon = 1e-6
    reference_arr = np.clip(reference_dist.to_numpy(dtype=float), epsilon, None)
    current_arr = np.clip(current_dist.to_numpy(dtype=float), epsilon, None)
    return float(np.sum((current_arr - reference_arr) * np.log(current_arr / reference_arr)))


def _categorical_psi(reference_values: pd.Series, current_values: pd.Series) -> float:
    reference_dist = (
        reference_values.fillna("__MISSING__").astype(str).value_counts(normalize=True, sort=False)
    )
    current_dist = (
        current_values.fillna("__MISSING__").astype(str).value_counts(normalize=True, sort=False)
    )
    all_categories = reference_dist.index.union(current_dist.index)
    epsilon = 1e-6
    reference_arr = np.clip(
        reference_dist.reindex(all_categories, fill_value=0.0).to_numpy(dtype=float), epsilon, None
    )
    current_arr = np.clip(
        current_dist.reindex(all_categories, fill_value=0.0).to_numpy(dtype=float), epsilon, None
    )
    return float(np.sum((current_arr - reference_arr) * np.log(current_arr / reference_arr)))


def _binary_ks(actual_values: pd.Series, score_values: pd.Series) -> float | None:
    positives = score_values[actual_values == 1]
    negatives = score_values[actual_values == 0]
    if not len(positives) or not len(negatives):
        return None
    all_scores = np.sort(np.unique(score_values))
    positive_cdf = np.searchsorted(np.sort(positives), all_scores, side="right") / len(positives)
    negative_cdf = np.searchsorted(np.sort(negatives), all_scores, side="right") / len(negatives)
    return float(np.max(np.abs(positive_cdf - negative_cdf)))


def _hosmer_lemeshow_p_value(
    actual_values: pd.Series,
    score_values: pd.Series,
    groups: int = 10,
) -> float | None:
    if score_values.nunique() < 2:
        return None
    frame = pd.DataFrame({"actual": actual_values, "score": score_values}).copy(deep=True)
    try:
        frame["group"] = pd.qcut(frame["score"], q=groups, duplicates="drop")
    except ValueError:
        return None

    summary = (
        frame.groupby("group", observed=False)
        .agg(observed=("actual", "sum"), expected=("score", "sum"), count=("actual", "size"))
        .reset_index(drop=True)
    )
    summary["observed_non_events"] = summary["count"] - summary["observed"]
    summary["expected_non_events"] = summary["count"] - summary["expected"]
    epsilon = 1e-6
    statistic = (
        ((summary["observed"] - summary["expected"]) ** 2)
        / np.clip(summary["expected"], epsilon, None)
        + ((summary["observed_non_events"] - summary["expected_non_events"]) ** 2)
        / np.clip(summary["expected_non_events"], epsilon, None)
    ).sum()
    degrees_freedom = max(len(summary) - 2, 1)
    return float(chi2.sf(statistic, degrees_freedom))


def _bad_rate_by_band_mae(actual_values: pd.Series, score_values: pd.Series) -> float | None:
    summary = _build_score_band_summary(actual_values=actual_values, score_values=score_values)
    if summary.empty:
        return None
    return float(summary["bad_rate_abs_error"].mean())


def _numeric_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.dropna().reset_index(drop=True)


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
