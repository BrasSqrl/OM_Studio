"""Applicability matrix for monitoring tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .thresholds import ThresholdRecord

if TYPE_CHECKING:  # pragma: no cover
    from .monitoring_pipeline import MonitoringRunResult


BINARY_ONLY_TESTS = frozenset(
    {
        "auc",
        "ks_statistic",
        "gini",
        "brier_score",
        "hosmer_lemeshow_p_value",
        "precision_at_threshold",
        "recall_at_threshold",
        "bad_rate_by_band_mae",
    }
)
LABEL_REQUIRED_TESTS = frozenset(
    {
        "auc",
        "ks_statistic",
        "gini",
        "brier_score",
        "mean_absolute_error",
        "hosmer_lemeshow_p_value",
        "precision_at_threshold",
        "recall_at_threshold",
        "bad_rate_by_band_mae",
    }
)
REFERENCE_SCORE_TESTS = frozenset({"score_psi", "score_ks_p_value"})
REFERENCE_INPUT_TESTS = frozenset(
    {
        "row_count_ratio",
        "segment_psi",
        "unexpected_category_count",
        "numeric_range_violation_count",
        "max_feature_psi",
        "min_numeric_feature_ks_p_value",
    }
)
IDENTIFIER_TESTS = frozenset({"duplicate_identifier_count", "identifier_null_rate_pct"})
DATE_TESTS = frozenset({"invalid_date_count", "stale_as_of_date_days"})


def build_test_applicability_matrix(
    *,
    result: MonitoringRunResult,
    thresholds: list[ThresholdRecord],
) -> pd.DataFrame:
    result_lookup = {item.test_id: item for item in result.test_results}
    rows = []
    for threshold in thresholds:
        test_result = result_lookup.get(threshold.test_id)
        applicability, reason = _resolve_applicability(result=result, threshold=threshold)
        if test_result is not None and test_result.status == "na":
            applicability = "not_applicable" if threshold.enabled else "disabled"
            reason = test_result.detail
        elif test_result is not None:
            applicability = "evaluated"
            reason = "Test was evaluated during this run."
        rows.append(
            {
                "test_id": threshold.test_id,
                "label": threshold.label,
                "category": threshold.category,
                "enabled": threshold.enabled,
                "applicability": applicability,
                "result_status": test_result.status if test_result is not None else "not_run",
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _resolve_applicability(
    *,
    result: MonitoringRunResult,
    threshold: ThresholdRecord,
) -> tuple[str, str]:
    if not threshold.enabled:
        return "disabled", "Threshold is disabled for this model."
    if threshold.test_id in LABEL_REQUIRED_TESTS and not result.labels_available:
        return "not_applicable", "Labels are unavailable for this monitoring run."
    if threshold.test_id in BINARY_ONLY_TESTS and result.model_bundle.target_mode != "binary":
        return "not_applicable", "Test applies only to binary target models."
    if threshold.test_id in REFERENCE_SCORE_TESTS and not result.model_bundle.reference_score_column:
        return "not_applicable", "Reference score column is unavailable."
    if (
        threshold.test_id in REFERENCE_INPUT_TESTS
        and result.model_bundle.bundle_paths.reference_input_path is None
    ):
        return "not_applicable", "Reference input snapshot is unavailable."
    if threshold.test_id == "segment_psi" and not result.segment_column:
        return "not_applicable", "No segment column was selected and available."
    if threshold.test_id in IDENTIFIER_TESTS and not result.model_bundle.identifier_columns:
        return "not_applicable", "No identifier columns are defined by the bundle."
    if threshold.test_id in DATE_TESTS and not result.model_bundle.date_columns:
        return "not_applicable", "No date columns are defined by the bundle."
    if result.status in {"contract_failed", "execution_failed"}:
        return "not_evaluated", f"Run status {result.status} prevented full evaluation."
    return "applicable", "Inputs required by this test are present."
