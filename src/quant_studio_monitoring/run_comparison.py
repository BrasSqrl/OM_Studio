"""Current-session run comparison exports."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .monitoring_pipeline import MonitoringRunResult


def build_run_comparison_summary_frame(
    left_run: MonitoringRunResult,
    right_run: MonitoringRunResult,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "baseline_run_id",
                "baseline": left_run.run_id,
                "comparison": right_run.run_id,
                "delta": "n/a",
            },
            {
                "metric": "model_name",
                "baseline": left_run.model_bundle.display_name,
                "comparison": right_run.model_bundle.display_name,
                "delta": "n/a",
            },
            {
                "metric": "dataset_name",
                "baseline": left_run.dataset.name,
                "comparison": right_run.dataset.name,
                "delta": "n/a",
            },
            {
                "metric": "status",
                "baseline": left_run.status,
                "comparison": right_run.status,
                "delta": "n/a",
            },
            {
                "metric": "pass_count",
                "baseline": left_run.pass_count,
                "comparison": right_run.pass_count,
                "delta": right_run.pass_count - left_run.pass_count,
            },
            {
                "metric": "fail_count",
                "baseline": left_run.fail_count,
                "comparison": right_run.fail_count,
                "delta": right_run.fail_count - left_run.fail_count,
            },
            {
                "metric": "na_count",
                "baseline": left_run.na_count,
                "comparison": right_run.na_count,
                "delta": right_run.na_count - left_run.na_count,
            },
        ]
    )


def build_run_comparison_frame(
    left_run: MonitoringRunResult,
    right_run: MonitoringRunResult,
) -> pd.DataFrame:
    left_frame = left_run.results_frame[
        ["test_id", "label", "status", "observed_value", "threshold_value"]
    ].rename(
        columns={
            "status": "baseline_status",
            "observed_value": "baseline_observed_value",
            "threshold_value": "baseline_threshold_value",
        }
    )
    right_frame = right_run.results_frame[
        ["test_id", "label", "status", "observed_value", "threshold_value"]
    ].rename(
        columns={
            "status": "comparison_status",
            "observed_value": "comparison_observed_value",
            "threshold_value": "comparison_threshold_value",
        }
    )
    merged = left_frame.merge(right_frame, on=["test_id", "label"], how="outer")
    merged["status_changed"] = merged["baseline_status"] != merged["comparison_status"]
    merged["observed_delta"] = (
        pd.to_numeric(merged["comparison_observed_value"], errors="coerce")
        - pd.to_numeric(merged["baseline_observed_value"], errors="coerce")
    )
    baseline_threshold = pd.to_numeric(merged["baseline_threshold_value"], errors="coerce")
    comparison_threshold = pd.to_numeric(merged["comparison_threshold_value"], errors="coerce")
    merged["threshold_changed"] = ~(
        baseline_threshold.fillna(-999999999.0) == comparison_threshold.fillna(-999999999.0)
    )
    changed = merged.loc[
        merged["status_changed"]
        | merged["threshold_changed"]
        | (merged["observed_delta"].abs() > 1e-12)
    ].copy()
    return changed.reset_index(drop=True)


def build_run_comparison_workbook_bytes(
    left_run: MonitoringRunResult,
    right_run: MonitoringRunResult,
) -> bytes:
    summary = build_run_comparison_summary_frame(left_run, right_run)
    changed_tests = build_run_comparison_frame(left_run, right_run)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="comparison_summary", index=False)
        changed_tests.to_excel(writer, sheet_name="changed_tests", index=False)
        left_run.results_frame.to_excel(writer, sheet_name="baseline_tests", index=False)
        right_run.results_frame.to_excel(writer, sheet_name="comparison_tests", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def build_run_comparison_html(
    left_run: MonitoringRunResult,
    right_run: MonitoringRunResult,
) -> str:
    summary = build_run_comparison_summary_frame(left_run, right_run)
    changed_tests = build_run_comparison_frame(left_run, right_run)
    changed_html = (
        "<p>No test-level differences were found.</p>"
        if changed_tests.empty
        else changed_tests.to_html(index=False, border=0)
    )
    return "\n".join(
        [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "  <meta charset='utf-8' />",
            "  <title>OM Studio Run Comparison</title>",
            "  <style>",
            "    body { font-family: Aptos, 'Segoe UI', sans-serif; margin: 32px; color: #172338; background: #fbfaf7; }",
            "    h1 { margin-bottom: 4px; }",
            "    table { border-collapse: collapse; width: 100%; background: white; }",
            "    th, td { border: 1px solid #d9e1ec; padding: 8px 10px; text-align: left; }",
            "    th { background: #0f5fd7; color: white; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>OM Studio Run Comparison</h1>",
            f"  <p>Baseline {left_run.run_id} vs comparison {right_run.run_id}</p>",
            "  <h2>Summary</h2>",
            summary.to_html(index=False, border=0),
            "  <h2>Changed Tests</h2>",
            changed_html,
            "</body>",
            "</html>",
        ]
    )
