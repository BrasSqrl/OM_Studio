from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from quant_studio_monitoring.run_comparison import (
    build_run_comparison_frame,
    build_run_comparison_html,
    build_run_comparison_summary_frame,
    build_run_comparison_workbook_bytes,
)


def test_run_comparison_exports_changed_tests() -> None:
    left_run = _run(
        run_id="baseline",
        dataset_name="baseline.csv",
        status="completed",
        pass_count=1,
        fail_count=1,
        na_count=0,
        results_frame=pd.DataFrame(
            [
                {
                    "test_id": "score_psi",
                    "label": "Score PSI",
                    "status": "pass",
                    "observed_value": 0.04,
                    "threshold_value": 0.10,
                },
                {
                    "test_id": "auc",
                    "label": "AUC",
                    "status": "fail",
                    "observed_value": 0.55,
                    "threshold_value": 0.60,
                },
            ]
        ),
    )
    right_run = _run(
        run_id="comparison",
        dataset_name="comparison.csv",
        status="completed",
        pass_count=2,
        fail_count=0,
        na_count=0,
        results_frame=pd.DataFrame(
            [
                {
                    "test_id": "score_psi",
                    "label": "Score PSI",
                    "status": "pass",
                    "observed_value": 0.08,
                    "threshold_value": 0.10,
                },
                {
                    "test_id": "auc",
                    "label": "AUC",
                    "status": "pass",
                    "observed_value": 0.68,
                    "threshold_value": 0.60,
                },
            ]
        ),
    )

    summary = build_run_comparison_summary_frame(left_run, right_run)
    changed = build_run_comparison_frame(left_run, right_run)
    workbook_bytes = build_run_comparison_workbook_bytes(left_run, right_run)
    html = build_run_comparison_html(left_run, right_run)

    assert summary.loc[summary["metric"] == "fail_count", "delta"].iloc[0] == -1
    assert set(changed["test_id"]) == {"score_psi", "auc"}
    assert workbook_bytes.startswith(b"PK")
    assert "OM Studio Run Comparison" in html


def _run(
    *,
    run_id: str,
    dataset_name: str,
    status: str,
    pass_count: int,
    fail_count: int,
    na_count: int,
    results_frame: pd.DataFrame,
) -> SimpleNamespace:
    return SimpleNamespace(
        run_id=run_id,
        model_bundle=SimpleNamespace(display_name="Retail PD"),
        dataset=SimpleNamespace(name=dataset_name),
        status=status,
        pass_count=pass_count,
        fail_count=fail_count,
        na_count=na_count,
        results_frame=results_frame,
    )
