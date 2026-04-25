"""Monitoring test catalog and reviewer guidance."""

from __future__ import annotations

TEST_HELP_GUIDANCE: dict[str, dict[str, str]] = {
    "required_columns_missing_count": {
        "fail_implies": "The uploaded dataset is missing fields required to score the saved bundle. Monitoring should not proceed until the contract is fixed.",
        "na_when": "This is only N/A if the threshold is disabled for the selected model.",
    },
    "row_count_ratio": {
        "fail_implies": "The monitoring population is materially smaller than the reference population, which can make diagnostics unstable or unrepresentative.",
        "na_when": "This is N/A when the threshold is disabled or reference row counts are unavailable.",
    },
    "max_feature_missingness_pct": {
        "fail_implies": "At least one required raw input field is missing too often in the monitoring dataset.",
        "na_when": "This is only N/A if the threshold is disabled.",
    },
    "row_count_absolute": {
        "fail_implies": "The monitoring file has fewer observations than the saved minimum for this model.",
        "na_when": "This is only N/A if the threshold is disabled.",
    },
    "duplicate_identifier_count": {
        "fail_implies": "One or more records share the same configured identifier, which can distort monitoring diagnostics.",
        "na_when": "This is N/A when no identifier column exists for the bundle or the threshold is disabled.",
    },
    "identifier_null_rate_pct": {
        "fail_implies": "At least one identifier field has missing values above the configured tolerance.",
        "na_when": "This is N/A when no identifier column exists for the bundle or the threshold is disabled.",
    },
    "invalid_date_count": {
        "fail_implies": "The monitoring data contains configured date values that could not be parsed.",
        "na_when": "This is N/A when the bundle has no date column or the threshold is disabled.",
    },
    "stale_as_of_date_days": {
        "fail_implies": "The latest monitoring as-of date is older than the configured freshness benchmark.",
        "na_when": "This is N/A when no date values are available or the threshold is disabled.",
    },
    "unexpected_category_count": {
        "fail_implies": "Categorical feature values appeared in monitoring data that were not present in the reference baseline.",
        "na_when": "This is N/A when reference input data is unavailable or the threshold is disabled.",
    },
    "numeric_range_violation_count": {
        "fail_implies": "Numeric feature values fell outside the reference baseline range.",
        "na_when": "This is N/A when reference input data is unavailable or the threshold is disabled.",
    },
    "score_psi": {
        "fail_implies": "Score distribution drift versus the reference bundle is above the accepted stability threshold.",
        "na_when": "This is N/A when reference scores are unavailable or the threshold is disabled.",
    },
    "score_ks_p_value": {
        "fail_implies": "Reference and monitoring score distributions differ enough that the drift signal is statistically significant under the configured benchmark.",
        "na_when": "This is N/A when reference scores are unavailable or the threshold is disabled.",
    },
    "segment_psi": {
        "fail_implies": "The selected segment mix has shifted meaningfully relative to the reference bundle.",
        "na_when": "This is N/A when no valid segment column is selected, reference segment data is unavailable, or the threshold is disabled.",
    },
    "max_feature_psi": {
        "fail_implies": "At least one raw model feature has a distribution shift above the configured benchmark.",
        "na_when": "This is N/A when reference input data is unavailable or the threshold is disabled.",
    },
    "min_numeric_feature_ks_p_value": {
        "fail_implies": "At least one numeric raw feature differs materially from the reference distribution under the configured p-value benchmark.",
        "na_when": "This is N/A when numeric feature drift cannot be computed or the threshold is disabled.",
    },
    "auc": {
        "fail_implies": "Observed rank-ordering performance on the monitoring population is below the accepted benchmark.",
        "na_when": "This is N/A when labels are unavailable, the target is not binary, or the threshold is disabled.",
    },
    "ks_statistic": {
        "fail_implies": "Separation between events and non-events is weaker than expected on the monitoring population.",
        "na_when": "This is N/A when labels are unavailable, the target is not binary, or the threshold is disabled.",
    },
    "gini": {
        "fail_implies": "Discriminatory power derived from AUC is below the accepted level.",
        "na_when": "This is N/A when labels are unavailable, the target is not binary, or the threshold is disabled.",
    },
    "brier_score": {
        "fail_implies": "Prediction error on realized outcomes is higher than the allowed threshold.",
        "na_when": "This is N/A when labels are unavailable, the target is not binary, or the threshold is disabled.",
    },
    "mean_absolute_error": {
        "fail_implies": "Average absolute prediction error is above the accepted tolerance.",
        "na_when": "This is N/A when labels are unavailable or the threshold is disabled.",
    },
    "hosmer_lemeshow_p_value": {
        "fail_implies": "Calibration is weak enough that observed outcomes differ materially from grouped predicted outcomes under the configured benchmark.",
        "na_when": "This is N/A when labels are unavailable, the target is not binary, or the threshold is disabled.",
    },
    "precision_at_threshold": {
        "fail_implies": "Observed precision at the saved model threshold is weaker than the accepted level.",
        "na_when": "This is N/A when labels are unavailable, the target is not binary, the model threshold is unavailable, or the threshold is disabled.",
    },
    "recall_at_threshold": {
        "fail_implies": "Observed recall at the saved model threshold is weaker than the accepted level.",
        "na_when": "This is N/A when labels are unavailable, the target is not binary, the model threshold is unavailable, or the threshold is disabled.",
    },
    "bad_rate_by_band_mae": {
        "fail_implies": "Observed bad rates by score band diverge too far from predicted rates across the monitoring population.",
        "na_when": "This is N/A when labels are unavailable, the target is not binary, or the threshold is disabled.",
    },
}


def guidance_for_test(test_id: str) -> dict[str, str]:
    return TEST_HELP_GUIDANCE.get(
        test_id,
        {
            "fail_implies": "The configured benchmark was not met.",
            "na_when": "This is N/A when the threshold is disabled or required inputs are unavailable.",
        },
    )
