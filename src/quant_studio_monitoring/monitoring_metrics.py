"""Reusable monitoring metric profiles and statistical helpers."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import chi2, ks_2samp

if TYPE_CHECKING:  # pragma: no cover
    from .registry import ModelBundle


def build_data_quality_summary(
    *,
    raw_dataframe: pd.DataFrame,
    reference_raw: pd.DataFrame | None,
    bundle: ModelBundle,
    run_started_at: datetime,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "metric": "row_count_absolute",
            "observed_value": float(len(raw_dataframe)),
            "reference_value": float(len(reference_raw)) if reference_raw is not None else None,
            "detail": "Monitoring dataset row count.",
        }
    )

    identifier_columns = [column for column in bundle.identifier_columns if column in raw_dataframe.columns]
    duplicate_count = None
    identifier_null_rate = None
    if identifier_columns:
        duplicate_count = float(raw_dataframe.duplicated(subset=identifier_columns).sum())
        identifier_null_rate = float(raw_dataframe[identifier_columns].isna().mean().max() * 100.0)
    rows.extend(
        [
            {
                "metric": "duplicate_identifier_count",
                "observed_value": duplicate_count,
                "reference_value": None,
                "detail": (
                    "Rows duplicated across identifier columns: "
                    + ", ".join(identifier_columns or bundle.identifier_columns)
                ),
            },
            {
                "metric": "identifier_null_rate_pct",
                "observed_value": identifier_null_rate,
                "reference_value": None,
                "detail": "Highest null percentage across identifier columns.",
            },
        ]
    )

    invalid_date_count = None
    stale_as_of_date_days = None
    parsed_date_values: list[pd.Timestamp] = []
    present_date_columns = [column for column in bundle.date_columns if column in raw_dataframe.columns]
    if present_date_columns:
        invalid_count = 0
        for column in present_date_columns:
            parsed = pd.to_datetime(raw_dataframe[column], errors="coerce")
            invalid_count += int((raw_dataframe[column].notna() & parsed.isna()).sum())
            parsed_date_values.extend(parsed.dropna().tolist())
        invalid_date_count = float(invalid_count)
        if parsed_date_values:
            latest_date = max(parsed_date_values).date()
            stale_as_of_date_days = float(max((run_started_at.date() - latest_date).days, 0))
    rows.extend(
        [
            {
                "metric": "invalid_date_count",
                "observed_value": invalid_date_count,
                "reference_value": None,
                "detail": "Non-empty configured date values that could not be parsed.",
            },
            {
                "metric": "stale_as_of_date_days",
                "observed_value": stale_as_of_date_days,
                "reference_value": None,
                "detail": "Days between run date and latest configured monitoring date.",
            },
        ]
    )

    unexpected_category_count = None
    numeric_range_violation_count = None
    if reference_raw is not None:
        unexpected_total = 0
        range_violation_total = 0
        for column in bundle.feature_columns:
            if column not in raw_dataframe.columns or column not in reference_raw.columns:
                continue
            current_values = raw_dataframe[column]
            reference_values = reference_raw[column]
            if series_pair_numeric_like(reference_values, current_values):
                reference_numeric = numeric_series(reference_values)
                current_numeric = pd.to_numeric(current_values, errors="coerce")
                if len(reference_numeric):
                    lower = float(reference_numeric.min())
                    upper = float(reference_numeric.max())
                    range_violation_total += int(
                        ((current_numeric < lower) | (current_numeric > upper)).sum()
                    )
            else:
                reference_categories = set(
                    reference_values.dropna().astype(str).str.strip().loc[lambda item: item != ""]
                )
                current_categories = set(
                    current_values.dropna().astype(str).str.strip().loc[lambda item: item != ""]
                )
                unexpected_total += len(current_categories - reference_categories)
        unexpected_category_count = float(unexpected_total)
        numeric_range_violation_count = float(range_violation_total)
    rows.extend(
        [
            {
                "metric": "unexpected_category_count",
                "observed_value": unexpected_category_count,
                "reference_value": None,
                "detail": "Categorical feature values present in monitoring but absent from reference.",
            },
            {
                "metric": "numeric_range_violation_count",
                "observed_value": numeric_range_violation_count,
                "reference_value": None,
                "detail": "Numeric feature values outside the reference min/max range.",
            },
        ]
    )

    return pd.DataFrame(rows)


def build_feature_drift_summary(
    *,
    reference_raw: pd.DataFrame | None,
    raw_dataframe: pd.DataFrame,
    bundle: ModelBundle,
) -> pd.DataFrame:
    columns = [
        "feature",
        "drift_method",
        "reference_non_null",
        "monitoring_non_null",
        "reference_missingness_pct",
        "monitoring_missingness_pct",
        "psi",
        "ks_p_value",
    ]
    if reference_raw is None:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for feature in bundle.feature_columns:
        if feature not in raw_dataframe.columns or feature not in reference_raw.columns:
            continue
        reference_values = reference_raw[feature]
        current_values = raw_dataframe[feature]
        numeric_like = series_pair_numeric_like(reference_values, current_values)
        if numeric_like:
            reference_numeric = numeric_series(reference_values)
            current_numeric = numeric_series(current_values)
            psi = score_psi(reference_numeric, current_numeric)
            ks_p_value = (
                float(ks_2samp(reference_numeric, current_numeric).pvalue)
                if len(reference_numeric) and len(current_numeric)
                else None
            )
            drift_method = "numeric_psi_and_ks"
        else:
            psi = categorical_psi(reference_values, current_values)
            ks_p_value = None
            drift_method = "categorical_psi"

        rows.append(
            {
                "feature": feature,
                "drift_method": drift_method,
                "reference_non_null": int(reference_values.notna().sum()),
                "monitoring_non_null": int(current_values.notna().sum()),
                "reference_missingness_pct": float(reference_values.isna().mean() * 100.0),
                "monitoring_missingness_pct": float(current_values.isna().mean() * 100.0),
                "psi": psi,
                "ks_p_value": ks_p_value,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summary_metric_value(summary: pd.DataFrame, metric: str) -> float | None:
    if summary.empty:
        return None
    matches = summary.loc[summary["metric"] == metric, "observed_value"]
    if matches.empty or pd.isna(matches.iloc[0]):
        return None
    return float(matches.iloc[0])


def build_label_evaluation_frame(
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


def score_psi(reference_scores: pd.Series | None, current_scores: pd.Series) -> float | None:
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


def categorical_psi(reference_values: pd.Series, current_values: pd.Series) -> float:
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


def binary_ks(actual_values: pd.Series, score_values: pd.Series) -> float | None:
    positives = score_values[actual_values == 1]
    negatives = score_values[actual_values == 0]
    if not len(positives) or not len(negatives):
        return None
    all_scores = np.sort(np.unique(score_values))
    positive_cdf = np.searchsorted(np.sort(positives), all_scores, side="right") / len(positives)
    negative_cdf = np.searchsorted(np.sort(negatives), all_scores, side="right") / len(negatives)
    return float(np.max(np.abs(positive_cdf - negative_cdf)))


def hosmer_lemeshow_p_value(
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


def bad_rate_by_band_mae(actual_values: pd.Series, score_values: pd.Series) -> float | None:
    summary = build_score_band_summary(actual_values=actual_values, score_values=score_values)
    if summary.empty:
        return None
    return float(summary["bad_rate_abs_error"].mean())


def build_score_band_summary(
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


def build_segment_share_frame(values: pd.Series, *, value_name: str) -> pd.DataFrame:
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


def numeric_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.dropna().reset_index(drop=True)


def series_pair_numeric_like(left: pd.Series, right: pd.Series) -> bool:
    left_non_null = left.dropna()
    right_non_null = right.dropna()
    if left_non_null.empty and right_non_null.empty:
        return False
    left_numeric = pd.to_numeric(left_non_null, errors="coerce")
    right_numeric = pd.to_numeric(right_non_null, errors="coerce")
    left_ratio = float(left_numeric.notna().mean()) if len(left_numeric) else 1.0
    right_ratio = float(right_numeric.notna().mean()) if len(right_numeric) else 1.0
    return left_ratio >= 0.95 and right_ratio >= 0.95
