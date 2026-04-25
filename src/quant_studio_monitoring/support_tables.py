"""Support-table assembly for monitoring runs."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .monitoring_metrics import (
    build_data_quality_summary,
    build_feature_drift_summary,
    build_label_evaluation_frame,
    build_score_band_summary,
    build_segment_share_frame,
    numeric_series,
)
from .registry import (
    ModelBundle,
    build_model_bundle_intake_checklist_frame,
    build_reference_baseline_diagnostics_frame,
)


def build_support_tables(
    *,
    raw_dataframe: pd.DataFrame,
    current_predictions: pd.DataFrame | None,
    reference_raw: pd.DataFrame | None,
    reference_predictions: pd.DataFrame | None,
    bundle: ModelBundle,
    score_column: str | None,
    actual_column: str | None,
    segment_column: str | None,
    run_started_at: datetime,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    tables["reference_baseline_diagnostics"] = build_reference_baseline_diagnostics_frame(bundle)
    tables["model_bundle_intake_checklist"] = build_model_bundle_intake_checklist_frame(bundle)
    tables["data_quality_summary"] = build_data_quality_summary(
        raw_dataframe=raw_dataframe,
        reference_raw=reference_raw,
        bundle=bundle,
        run_started_at=run_started_at,
    )
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
            {
                "field": "monitoring_contract_version",
                "value": bundle.monitoring_contract_version or "not_declared",
            },
            {
                "field": "monitoring_contract_version_status",
                "value": bundle.monitoring_contract_version_status,
            },
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
        current_scores = numeric_series(current_predictions[score_column])
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
            reference_scores = numeric_series(reference_predictions[bundle.reference_score_column])
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

        label_frame = build_label_evaluation_frame(
            current_predictions=current_predictions,
            score_column=score_column,
            actual_column=actual_column,
        )
        if not label_frame.empty:
            tables["score_band_summary"] = build_score_band_summary(
                actual_values=label_frame["actual"],
                score_values=label_frame["score"],
            )

    if segment_column and segment_column in raw_dataframe.columns:
        current_segment = build_segment_share_frame(
            raw_dataframe[segment_column],
            value_name="current_share_pct",
        )
        if reference_raw is not None and segment_column in reference_raw.columns:
            reference_segment = build_segment_share_frame(
                reference_raw[segment_column],
                value_name="reference_share_pct",
            )
            current_segment = current_segment.merge(
                reference_segment, on="segment_value", how="outer"
            ).fillna(0.0)
        tables["segment_summary"] = current_segment.sort_values(
            "current_share_pct", ascending=False
        ).reset_index(drop=True)

    feature_drift = build_feature_drift_summary(
        reference_raw=reference_raw,
        raw_dataframe=raw_dataframe,
        bundle=bundle,
    )
    if not feature_drift.empty:
        tables["feature_drift_summary"] = feature_drift

    if current_predictions is not None:
        tables["scored_data_preview"] = current_predictions.head(50).copy(deep=True)
    return tables
