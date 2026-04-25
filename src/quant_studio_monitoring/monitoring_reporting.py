"""Artifact writing and HTML/Excel report generation for monitoring runs."""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd
import plotly.express as px

from .artifact_completeness import write_artifact_completeness_artifacts
from .config import MonitoringPerformanceConfig
from .presentation import apply_fintech_figure_theme
from .registry import build_bundle_fingerprint_frame, build_dataset_fingerprint_frame
from .test_applicability import build_test_applicability_matrix
from .thresholds import ThresholdRecord, threshold_records_to_frame

if False:  # pragma: no cover
    from .monitoring_pipeline import MonitoringRunResult


def write_monitoring_artifacts(
    *,
    result: MonitoringRunResult,
    thresholds: list[ThresholdRecord],
    raw_dataframe: pd.DataFrame,
    artifact_profile: str = "full",
    performance: MonitoringPerformanceConfig | None = None,
) -> dict[str, Path | None]:
    artifact_profile = _normalize_artifact_profile(artifact_profile)
    performance = performance or MonitoringPerformanceConfig()
    performance.validate()
    report_path = result.run_root / "monitoring_report.html"
    workbook_path = result.run_root / "monitoring_workbook.xlsx"
    tests_json_path = result.run_root / "monitoring_tests.json"
    tests_csv_path = result.run_root / "monitoring_test_results.csv"
    scored_csv_path = result.run_root / "current_scored_data.csv"
    thresholds_path = result.run_root / "threshold_snapshot.json"
    manifest_path = result.run_root / "artifacts_manifest.json"
    metadata_path = result.run_root / "bundle_metadata.json"
    input_contract_path = result.run_root / "input_contract.json"
    failure_diagnostics_path = result.run_root / "failure_diagnostics.json"
    reviewer_notes_path = result.run_root / "reviewer_notes.json"
    reviewer_exceptions_path = result.run_root / "reviewer_exceptions.json"
    run_config_path = result.run_root / "monitoring_run_config.json"
    dataset_provenance_path = result.run_root / "dataset_provenance.json"
    artifact_completeness_csv_path = result.run_root / "artifact_completeness.csv"
    artifact_completeness_json_path = result.run_root / "artifact_completeness.json"
    diagnostic_export_path = result.run_root / "diagnostic_export.zip"
    reviewer_package_path = result.run_root / "reviewer_package.zip"

    result.run_root.mkdir(parents=True, exist_ok=True)

    support_tables = {name: table.copy(deep=True) for name, table in result.support_tables.items()}
    detailed_results_frame = _build_detailed_results_frame(result)
    support_tables["executive_summary"] = _build_executive_summary_frame(
        result=result,
        raw_dataframe=raw_dataframe,
        detailed_results_frame=detailed_results_frame,
    )
    support_tables["test_enablement_profile"] = _build_test_enablement_profile_frame(thresholds)
    support_tables["test_applicability_matrix"] = build_test_applicability_matrix(
        result=result,
        thresholds=thresholds,
    )
    support_tables["reviewer_exception_summary"] = _build_reviewer_exception_summary_frame(result)
    support_tables["test_result_summary"] = detailed_results_frame
    support_tables["category_status_summary"] = _build_category_status_summary(detailed_results_frame)
    support_tables["failed_test_summary"] = detailed_results_frame.loc[
        detailed_results_frame["status"] == "fail"
    ].reset_index(drop=True)
    support_tables["reference_monitoring_delta_summary"] = _build_reference_monitoring_delta_summary(
        result=result,
        raw_dataframe=raw_dataframe,
        support_tables=support_tables,
        detailed_results_frame=detailed_results_frame,
    )
    support_tables["bundle_fingerprint"] = build_bundle_fingerprint_frame(result.model_bundle)
    support_tables["dataset_fingerprint"] = build_dataset_fingerprint_frame(result.dataset)
    support_tables["run_fingerprint"] = _build_run_fingerprint_frame(
        result=result,
        thresholds=thresholds,
    )
    support_tables["failure_diagnostics"] = _build_failure_diagnostics_frame(result)
    support_tables["dataset_provenance"] = pd.DataFrame(
        [{"field": key, "value": value} for key, value in result.dataset_provenance.items()]
    )
    result.support_tables = support_tables

    results_frame = detailed_results_frame
    results_frame.to_csv(tests_csv_path, index=False)
    _write_json(tests_json_path, {"tests": detailed_results_frame.to_dict(orient="records")})
    _write_json(
        thresholds_path,
        {"thresholds": [asdict(record) for record in thresholds]},
    )
    _write_json(
        metadata_path,
        {
            "bundle_id": result.model_bundle.bundle_id,
            "display_name": result.model_bundle.display_name,
            "model_version": result.model_bundle.model_version,
            "model_owner": result.model_bundle.model_owner,
            "business_purpose": result.model_bundle.business_purpose,
            "portfolio_name": result.model_bundle.portfolio_name,
            "segment_name": result.model_bundle.segment_name,
            "approval_status": result.model_bundle.monitoring_metadata.approval_status,
            "approval_date": result.model_bundle.monitoring_metadata.approval_date,
            "monitoring_notes": result.model_bundle.monitoring_metadata.monitoring_notes,
            "model_type": result.model_bundle.model_type,
            "target_mode": result.model_bundle.target_mode,
            "monitoring_contract_version": result.model_bundle.monitoring_contract_version,
            "monitoring_contract_version_status": result.model_bundle.monitoring_contract_version_status,
            "bundle_path": str(result.model_bundle.bundle_paths.root),
            "monitoring_metadata_path": str(result.model_bundle.bundle_paths.root / "monitoring_metadata.json"),
        },
    )
    _write_json(
        input_contract_path,
        {
            "required_input_columns": result.contract.required_input_columns,
            "missing_required_columns": result.contract.missing_required_columns,
            "optional_input_columns": result.contract.optional_input_columns,
            "missing_optional_columns": result.contract.missing_optional_columns,
            "resolved_label_column": result.contract.resolved_label_column,
            "labels_available": result.contract.labels_available,
            "requested_segment_column": result.contract.requested_segment_column,
            "segment_available": result.contract.segment_available,
        },
    )
    _write_json(failure_diagnostics_path, result.failure_context or {})
    _write_json(reviewer_notes_path, {"reviewer_notes": result.reviewer_notes})
    _write_json(
        reviewer_exceptions_path,
        {"reviewer_exceptions": result.reviewer_exceptions},
    )
    _write_json(
        run_config_path,
        result.run_config.to_payload() if result.run_config is not None else {},
    )
    _write_json(dataset_provenance_path, result.dataset_provenance)

    scored_preview = support_tables.get("scored_data_preview")
    if scored_preview is not None and not scored_preview.empty and artifact_profile != "minimal":
        scored_preview.to_csv(scored_csv_path, index=False)

    for table_name, table in support_tables.items():
        if table.empty or not _should_write_support_csv(table_name, artifact_profile):
            continue
        table.to_csv(result.run_root / f"{table_name}.csv", index=False)

    _write_workbook(
        workbook_path=workbook_path,
        result=result,
        thresholds=thresholds,
        raw_dataframe=raw_dataframe,
        support_tables=support_tables,
        detailed_results_frame=detailed_results_frame,
        artifact_profile=artifact_profile,
    )
    report_path.write_text(
        _build_html_report(
            result=result,
            thresholds=thresholds,
            raw_dataframe=raw_dataframe,
            support_tables=support_tables,
            detailed_results_frame=detailed_results_frame,
            artifact_profile=artifact_profile,
            performance=performance,
        ),
        encoding="utf-8",
    )

    manifest = {
        "run_root": str(result.run_root),
        "artifact_profile": artifact_profile,
        "report": str(report_path),
        "workbook": str(workbook_path),
        "tests_json": str(tests_json_path),
        "tests_csv": str(tests_csv_path),
        "threshold_snapshot": str(thresholds_path),
        "bundle_metadata": str(metadata_path),
        "input_contract": str(input_contract_path),
        "failure_diagnostics": str(failure_diagnostics_path),
        "reviewer_notes": str(reviewer_notes_path),
        "reviewer_exceptions": str(reviewer_exceptions_path),
        "monitoring_run_config": str(run_config_path),
        "dataset_provenance": str(dataset_provenance_path),
        "artifact_completeness_csv": str(artifact_completeness_csv_path),
        "artifact_completeness_json": str(artifact_completeness_json_path),
        "manifest": str(manifest_path),
        "diagnostic_export": str(diagnostic_export_path),
        "reviewer_package": str(reviewer_package_path) if artifact_profile != "minimal" else None,
        "scoring_output_root": str(result.scoring_output_root) if result.scoring_output_root else None,
        "bundle_fingerprint": str(result.run_root / "bundle_fingerprint.csv"),
        "dataset_fingerprint": str(result.run_root / "dataset_fingerprint.csv"),
        "run_fingerprint": str(result.run_root / "run_fingerprint.csv"),
    }
    if scored_csv_path.exists():
        manifest["current_scored_data"] = str(scored_csv_path)
    manifest["generated_artifacts"] = _generated_artifact_index(
        manifest=manifest,
        planned_paths={"reviewer_package"} if artifact_profile != "minimal" else set(),
    )
    manifest["skipped_artifacts"] = _skipped_artifact_index(
        artifact_profile=artifact_profile,
        scored_csv_path=scored_csv_path,
    )
    _write_json(manifest_path, manifest)
    _build_reviewer_package(
        reviewer_package_path=diagnostic_export_path,
        files=[
            manifest_path,
            tests_csv_path,
            tests_json_path,
            thresholds_path,
            metadata_path,
            input_contract_path,
            failure_diagnostics_path,
            reviewer_notes_path,
            reviewer_exceptions_path,
            run_config_path,
            dataset_provenance_path,
        ],
    )
    completeness_frame = write_artifact_completeness_artifacts(
        manifest=manifest,
        csv_path=artifact_completeness_csv_path,
        json_path=artifact_completeness_json_path,
        planned_keys={"reviewer_package"} if artifact_profile != "minimal" else set(),
    )
    support_tables["artifact_completeness"] = completeness_frame
    result.support_tables = support_tables
    missing_count = int((completeness_frame["status"] == "missing").sum())
    manifest["artifact_completeness_status"] = "complete" if missing_count == 0 else "incomplete"
    manifest["generated_artifacts"] = _generated_artifact_index(
        manifest=manifest,
        planned_paths={"reviewer_package"} if artifact_profile != "minimal" else set(),
    )
    _write_json(manifest_path, manifest)
    _write_workbook(
        workbook_path=workbook_path,
        result=result,
        thresholds=thresholds,
        raw_dataframe=raw_dataframe,
        support_tables=support_tables,
        detailed_results_frame=detailed_results_frame,
        artifact_profile=artifact_profile,
    )
    report_path.write_text(
        _build_html_report(
            result=result,
            thresholds=thresholds,
            raw_dataframe=raw_dataframe,
            support_tables=support_tables,
            detailed_results_frame=detailed_results_frame,
            artifact_profile=artifact_profile,
            performance=performance,
        ),
        encoding="utf-8",
    )
    if artifact_profile != "minimal":
        _build_reviewer_package(
            reviewer_package_path=reviewer_package_path,
            files=[
                report_path,
                workbook_path,
                tests_csv_path,
                tests_json_path,
                thresholds_path,
                manifest_path,
                metadata_path,
                input_contract_path,
                failure_diagnostics_path,
                reviewer_notes_path,
                reviewer_exceptions_path,
                run_config_path,
                dataset_provenance_path,
                result.run_root / "executive_summary.csv",
                result.run_root / "test_enablement_profile.csv",
                result.run_root / "test_applicability_matrix.csv",
                result.run_root / "reviewer_exception_summary.csv",
                result.run_root / "model_bundle_intake_checklist.csv",
                result.run_root / "dataset_provenance.csv",
                result.run_root / "bundle_fingerprint.csv",
                result.run_root / "dataset_fingerprint.csv",
                result.run_root / "run_fingerprint.csv",
                result.run_root / "reference_baseline_diagnostics.csv",
                result.run_root / "data_quality_summary.csv",
                result.run_root / "feature_drift_summary.csv",
                result.run_root / "outcome_join_summary.csv",
                scored_csv_path if scored_csv_path.exists() else None,
            ],
        )

    return {
        "report": report_path,
        "workbook": workbook_path,
        "tests_json": tests_json_path,
        "tests_csv": tests_csv_path,
        "threshold_snapshot": thresholds_path,
        "bundle_metadata": metadata_path,
        "input_contract": input_contract_path,
        "failure_diagnostics": failure_diagnostics_path,
        "reviewer_notes": reviewer_notes_path,
        "reviewer_exceptions": reviewer_exceptions_path,
        "monitoring_run_config": run_config_path,
        "dataset_provenance": dataset_provenance_path,
        "artifact_completeness_csv": artifact_completeness_csv_path,
        "artifact_completeness_json": artifact_completeness_json_path,
        "diagnostic_export": diagnostic_export_path,
        "current_scored_data": scored_csv_path if scored_csv_path.exists() else None,
        "manifest": manifest_path,
        "reviewer_package": reviewer_package_path if reviewer_package_path.exists() else None,
    }


def _normalize_artifact_profile(value: str) -> str:
    normalized = str(value or "full").strip().lower()
    return normalized if normalized in {"full", "reviewer", "minimal"} else "full"


def _generated_artifact_index(
    *,
    manifest: dict[str, object],
    planned_paths: set[str],
) -> dict[str, str]:
    generated: dict[str, str] = {}
    for key, value in manifest.items():
        if not isinstance(value, str):
            continue
        path = Path(value)
        if path.exists() or key in planned_paths:
            generated[key] = value
    return generated


def _skipped_artifact_index(
    *,
    artifact_profile: str,
    scored_csv_path: Path,
) -> list[dict[str, str]]:
    skipped: list[dict[str, str]] = []
    if artifact_profile == "minimal":
        skipped.append(
            {
                "artifact": "reviewer_package",
                "reason": "Minimal artifact profile skips the zip package.",
            }
        )
    if not scored_csv_path.exists():
        skipped.append(
            {
                "artifact": "current_scored_data",
                "reason": (
                    "Scored data CSV is only written when a scored preview exists "
                    "and the artifact profile is not minimal."
                ),
            }
        )
    return skipped


def _should_write_support_csv(table_name: str, artifact_profile: str) -> bool:
    if artifact_profile == "full":
        return True
    if artifact_profile == "minimal":
        return table_name in {
            "executive_summary",
            "test_result_summary",
            "failure_diagnostics",
            "dataset_provenance",
            "bundle_fingerprint",
            "dataset_fingerprint",
            "run_fingerprint",
        }
    return table_name in {
        "executive_summary",
        "test_result_summary",
        "category_status_summary",
        "failed_test_summary",
        "reference_monitoring_delta_summary",
        "reference_baseline_diagnostics",
        "data_quality_summary",
        "feature_drift_summary",
        "outcome_join_summary",
        "test_enablement_profile",
        "test_applicability_matrix",
        "reviewer_exception_summary",
        "model_bundle_intake_checklist",
        "failure_diagnostics",
        "dataset_provenance",
        "bundle_fingerprint",
        "dataset_fingerprint",
        "run_fingerprint",
    }


def _should_write_workbook_sheet(table_name: str, artifact_profile: str) -> bool:
    if artifact_profile in {"full", "reviewer"}:
        return True
    return table_name in {
        "test_result_summary",
        "category_status_summary",
        "failed_test_summary",
        "reference_monitoring_delta_summary",
        "data_quality_summary",
        "feature_drift_summary",
        "test_applicability_matrix",
        "reviewer_exception_summary",
        "model_bundle_intake_checklist",
        "artifact_completeness",
        "failure_diagnostics",
        "dataset_provenance",
    }


def _write_workbook(
    *,
    workbook_path: Path,
    result: MonitoringRunResult,
    thresholds: list[ThresholdRecord],
    raw_dataframe: pd.DataFrame,
    support_tables: dict[str, pd.DataFrame],
    detailed_results_frame: pd.DataFrame,
    artifact_profile: str,
) -> None:
    summary_frame = pd.DataFrame(
        [
            {
                "run_id": result.run_id,
                "model_name": result.model_bundle.display_name,
                "model_version": result.model_bundle.model_version,
                "dataset_name": result.dataset.name,
                "status": result.status,
                "score_column": result.score_column,
                "labels_available": result.labels_available,
                "segment_column": result.segment_column,
                "pass_count": result.pass_count,
                "fail_count": result.fail_count,
                "na_count": result.na_count,
                "raw_row_count": len(raw_dataframe),
                "raw_column_count": len(raw_dataframe.columns),
            }
        ]
    )

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        support_tables.get("executive_summary", pd.DataFrame()).to_excel(
            writer,
            sheet_name="executive_summary",
            index=False,
        )
        summary_frame.to_excel(writer, sheet_name="summary", index=False)
        detailed_results_frame.to_excel(writer, sheet_name="test_results", index=False)
        threshold_records_to_frame(thresholds).to_excel(writer, sheet_name="thresholds", index=False)
        pd.DataFrame(
            [
                {
                    "required_input_columns": ", ".join(result.contract.required_input_columns),
                    "missing_required_columns": ", ".join(result.contract.missing_required_columns),
                    "optional_input_columns": ", ".join(result.contract.optional_input_columns),
                    "missing_optional_columns": ", ".join(result.contract.missing_optional_columns),
                    "resolved_label_column": result.contract.resolved_label_column,
                    "labels_available": result.contract.labels_available,
                    "requested_segment_column": result.contract.requested_segment_column,
                    "segment_available": result.contract.segment_available,
                }
            ]
        ).to_excel(writer, sheet_name="input_contract", index=False)
        for table_name, table in support_tables.items():
            if (
                table.empty
                or table_name in {"executive_summary"}
                or not _should_write_workbook_sheet(table_name, artifact_profile)
            ):
                continue
            table.to_excel(writer, sheet_name=_sanitize_sheet_name(table_name), index=False)


def _build_html_report(
    *,
    result: MonitoringRunResult,
    thresholds: list[ThresholdRecord],
    raw_dataframe: pd.DataFrame,
    support_tables: dict[str, pd.DataFrame],
    detailed_results_frame: pd.DataFrame,
    artifact_profile: str,
    performance: MonitoringPerformanceConfig,
) -> str:
    results_frame = detailed_results_frame.copy(deep=True)
    if not results_frame.empty:
        results_frame["observed_value"] = results_frame["observed_value"].map(_format_numeric)
        results_frame["threshold_value"] = results_frame["threshold_value"].map(_format_numeric)

    figures_html: list[str] = []
    score_profile = support_tables.get("score_profile", pd.DataFrame())
    if artifact_profile != "minimal" and not score_profile.empty:
        figure = px.bar(
            score_profile,
            x="population",
            y="score_mean",
            color="population",
            title="Reference vs Monitoring Score Mean",
            text="row_count",
        )
        figures_html.append(
            apply_fintech_figure_theme(
                figure,
                title="Reference vs Monitoring Score Mean",
            ).to_html(full_html=False, include_plotlyjs="cdn")
        )

    missingness = support_tables.get("missingness_summary", pd.DataFrame()).head(12)
    if artifact_profile != "minimal" and not missingness.empty:
        figure = px.bar(
            missingness.sort_values("current_missingness_pct", ascending=True),
            x="current_missingness_pct",
            y="column_name",
            orientation="h",
            title="Current Missingness By Column",
        )
        figures_html.append(
            apply_fintech_figure_theme(
                figure,
                title="Current Missingness By Column",
            ).to_html(full_html=False, include_plotlyjs=False)
        )

    score_band_summary = support_tables.get("score_band_summary", pd.DataFrame())
    if artifact_profile != "minimal" and not score_band_summary.empty:
        figure = px.line(
            score_band_summary,
            x="band_label",
            y=["average_predicted_score", "observed_rate"],
            markers=True,
            title="Predicted vs Observed By Score Band",
        )
        figures_html.append(
            apply_fintech_figure_theme(
                figure,
                title="Predicted vs Observed By Score Band",
            ).to_html(full_html=False, include_plotlyjs=False)
        )

    segment_summary = support_tables.get("segment_summary", pd.DataFrame())
    if artifact_profile != "minimal" and not segment_summary.empty:
        y_values = ["current_share_pct"]
        if "reference_share_pct" in segment_summary.columns:
            y_values.append("reference_share_pct")
        figure = px.bar(
            segment_summary,
            x="segment_value",
            y=y_values,
            barmode="group",
            title="Segment Mix Comparison",
        )
        figures_html.append(
            apply_fintech_figure_theme(
                figure,
                title="Segment Mix Comparison",
            ).to_html(full_html=False, include_plotlyjs=False)
        )

    rows = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8' />",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0' />",
        "  <title>Monitoring Report</title>",
        "  <style>",
        "    body { font-family: Aptos, 'Segoe UI', sans-serif; background: #fcfaf6; color: #112033; margin: 0; }",
        "    main { max-width: 1200px; margin: 0 auto; padding: 32px 24px 48px; }",
        "    .hero { border: 1px solid rgba(17,32,51,0.08); border-radius: 28px; padding: 28px 30px; background: linear-gradient(135deg, #fffdfc, #f3eee5); box-shadow: 0 22px 60px rgba(17,32,51,0.08); }",
        "    .kicker { color: #c28a2c; text-transform: uppercase; letter-spacing: 0.16em; font-size: 12px; margin-bottom: 8px; }",
        "    h1 { margin: 0; font-size: 42px; line-height: 1; }",
        "    h2 { margin-top: 34px; font-size: 24px; }",
        "    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 18px; }",
        "    .card { background: rgba(255,252,249,0.94); border: 1px solid rgba(17,32,51,0.08); border-radius: 18px; padding: 16px; }",
        "    .card-label { color: #607089; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }",
        "    .card-value { font-size: 28px; font-weight: 700; margin-top: 8px; }",
        "    table { width: 100%; border-collapse: collapse; background: white; border-radius: 18px; overflow: hidden; }",
        "    th, td { padding: 10px 12px; border-bottom: 1px solid rgba(17,32,51,0.08); text-align: left; vertical-align: top; }",
        "    th { background: #16324f; color: white; }",
        "    .status-pass { color: #116149; font-weight: 700; }",
        "    .status-fail { color: #a43333; font-weight: 700; }",
        "    .status-na { color: #607089; font-weight: 700; }",
        "    .muted { color: #607089; }",
        "    .error { background: #fff1f1; border: 1px solid #f0c5c5; border-radius: 18px; padding: 14px 16px; }",
        "    .info { background: #eef7ff; border: 1px solid #c6def5; border-radius: 18px; padding: 14px 16px; }",
        "    .spacer { height: 18px; }",
        "  </style>",
        "</head>",
        "<body>",
        "<main>",
        "  <section class='hero'>",
        "    <div class='kicker'>OM Studio</div>",
        f"    <h1>{_escape(result.model_bundle.display_name)}</h1>",
        f"    <p class='muted'>Version {_escape(result.model_bundle.model_version)} | Dataset {_escape(result.dataset.name)} | Run {result.run_id}</p>",
        "    <div class='cards'>",
        f"      <div class='card'><div class='card-label'>Status</div><div class='card-value'>{_escape(result.status)}</div></div>",
        f"      <div class='card'><div class='card-label'>Passed</div><div class='card-value'>{result.pass_count}</div></div>",
        f"      <div class='card'><div class='card-label'>Failed</div><div class='card-value'>{result.fail_count}</div></div>",
        f"      <div class='card'><div class='card-label'>N/A</div><div class='card-value'>{result.na_count}</div></div>",
        f"      <div class='card'><div class='card-label'>Rows</div><div class='card-value'>{len(raw_dataframe):,}</div></div>",
        f"      <div class='card'><div class='card-label'>Score Column</div><div class='card-value'>{_escape(result.score_column or 'n/a')}</div></div>",
        "    </div>",
        "  </section>",
    ]

    if result.error_message:
        rows.extend(
            [
                "<div class='spacer'></div>",
                f"<div class='error'><strong>Execution Error:</strong> {_escape(result.error_message)}</div>",
            ]
        )
    elif not result.labels_available:
        rows.extend(
            [
                "<div class='spacer'></div>",
                "<div class='info'><strong>Score-Only Run:</strong> Labels were unavailable in the monitoring dataset, so label-dependent tests were marked N/A.</div>",
            ]
        )

    rows.extend(
        [
            "<h2>Executive Summary</h2>",
            _table_preview_html(support_tables["executive_summary"], performance),
            "<h2>Reviewer Summary</h2>",
            _table_preview_html(support_tables["category_status_summary"], performance),
            "<h2>Reviewer Exceptions</h2>",
            _table_preview_html(support_tables["reviewer_exception_summary"], performance),
            "<h2>Model Bundle Intake Checklist</h2>",
            _table_preview_html(support_tables["model_bundle_intake_checklist"], performance),
            "<h2>Reference Vs Monitoring Delta</h2>",
            _table_preview_html(
                support_tables["reference_monitoring_delta_summary"],
                performance,
            ),
            "<h2>Per-Test Results</h2>",
            _results_table_html(results_frame.head(performance.html_table_preview_rows)),
        ]
    )

    failed_tests = support_tables.get("failed_test_summary", pd.DataFrame())
    if not failed_tests.empty:
        rows.extend(
            [
                "<h2>Failed Tests</h2>",
                _table_preview_html(failed_tests, performance),
            ]
        )

    rows.extend(
        [
            "<h2>Threshold Snapshot</h2>",
            threshold_records_to_frame(thresholds)
            .drop(columns=["test_id"])
            .head(performance.html_table_preview_rows)
            .to_html(index=False, border=0),
            "<h2>Model Metadata</h2>",
            _table_preview_html(support_tables["model_metadata"], performance),
            "<h2>Input Contract</h2>",
            pd.DataFrame(
                [
                    {
                        "required_input_columns": ", ".join(result.contract.required_input_columns),
                        "missing_required_columns": ", ".join(result.contract.missing_required_columns),
                        "optional_input_columns": ", ".join(result.contract.optional_input_columns),
                        "missing_optional_columns": ", ".join(result.contract.missing_optional_columns),
                        "resolved_label_column": result.contract.resolved_label_column,
                        "labels_available": result.contract.labels_available,
                        "segment_column": result.segment_column,
                    }
                ]
            ).head(performance.html_table_preview_rows).to_html(index=False, border=0),
            "<h2>Run Fingerprint</h2>",
            _table_preview_html(support_tables["run_fingerprint"], performance),
            "<h2>Dataset Fingerprint</h2>",
            _table_preview_html(support_tables["dataset_fingerprint"], performance),
            "<h2>Bundle Fingerprint</h2>",
            _table_preview_html(support_tables["bundle_fingerprint"], performance),
        ]
    )

    failure_diagnostics = support_tables.get("failure_diagnostics", pd.DataFrame())
    if not failure_diagnostics.empty:
        rows.extend(
            [
                "<h2>Failure Diagnostics</h2>",
                _table_preview_html(failure_diagnostics, performance),
            ]
        )

    if figures_html:
        rows.append("<h2>Monitoring Figures</h2>")
        rows.extend(figures_html[: performance.html_max_figures])
        if len(figures_html) > performance.html_max_figures:
            rows.append(
                f"<p class='muted'>Showing first {performance.html_max_figures:,} "
                f"of {len(figures_html):,} figures in this HTML report.</p>"
            )

    for table_name in (
        "reference_baseline_diagnostics",
        "test_applicability_matrix",
        "artifact_completeness",
        "data_quality_summary",
        "feature_drift_summary",
        "outcome_join_summary",
        "test_enablement_profile",
        "missingness_summary",
        "score_band_summary",
        "segment_summary",
    ):
        table = support_tables.get(table_name, pd.DataFrame())
        if table.empty:
            continue
        rows.append(f"<h2>{_escape(table_name.replace('_', ' ').title())}</h2>")
        rows.append(_table_preview_html(table, performance))

    rows.extend(["</main>", "</body>", "</html>"])
    return "\n".join(rows)


def _table_preview_html(frame: pd.DataFrame, performance: MonitoringPerformanceConfig) -> str:
    if len(frame) <= performance.html_table_preview_rows:
        return frame.to_html(index=False, border=0)
    preview = frame.head(performance.html_table_preview_rows)
    return (
        preview.to_html(index=False, border=0)
        + f"<p class='muted'>Showing first {performance.html_table_preview_rows:,} "
        f"of {len(frame):,} rows in the HTML report. See the workbook for the full table.</p>"
    )


def _results_table_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p class='muted'>No test results were produced.</p>"

    rows = [
        "<table>",
        "<thead><tr><th>Test</th><th>Category</th><th>Status</th><th>Observed</th><th>Benchmark</th><th>Interpretation</th><th>Detail</th><th>Reviewer Exception</th><th>Reviewer Note</th></tr></thead>",
        "<tbody>",
    ]
    for item in frame.to_dict(orient="records"):
        rows.append(
            "<tr>"
            f"<td>{_escape(str(item['label']))}</td>"
            f"<td>{_escape(str(item['category']))}</td>"
            f"<td class='status-{_escape(str(item['status']))}'>{_escape(str(item['status']))}</td>"
            f"<td>{_escape(str(item['observed_value']))}</td>"
            f"<td>{_escape(str(item['benchmark']))}</td>"
            f"<td>{_escape(str(item['interpretation']))}</td>"
            f"<td>{_escape(str(item['detail']))}</td>"
            f"<td>{_escape(str(item.get('reviewer_exception', '')))}</td>"
            f"<td>{_escape(str(item.get('reviewer_note', '')))}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def _build_detailed_results_frame(result: MonitoringRunResult) -> pd.DataFrame:
    frame = result.results_frame.copy(deep=True)
    if frame.empty:
        return frame

    frame["benchmark"] = frame.apply(_benchmark_text, axis=1)
    frame["interpretation"] = frame.apply(_interpret_result_row, axis=1)
    frame["reviewer_note"] = frame["test_id"].map(result.reviewer_notes).fillna("")
    frame["reviewer_exception"] = frame["test_id"].map(
        lambda test_id: result.reviewer_exceptions.get(str(test_id), {}).get(
            "disposition",
            "",
        )
    )
    frame["reviewer_exception_rationale"] = frame["test_id"].map(
        lambda test_id: result.reviewer_exceptions.get(str(test_id), {}).get(
            "rationale",
            "",
        )
    )
    ordered_columns = [
        "test_id",
        "label",
        "category",
        "status",
        "observed_value",
        "operator",
        "threshold_value",
        "benchmark",
        "interpretation",
        "detail",
        "reviewer_exception",
        "reviewer_exception_rationale",
        "reviewer_note",
    ]
    return frame[ordered_columns]


def _build_executive_summary_frame(
    *,
    result: MonitoringRunResult,
    raw_dataframe: pd.DataFrame,
    detailed_results_frame: pd.DataFrame,
) -> pd.DataFrame:
    failed_tests = []
    if not detailed_results_frame.empty:
        failed_tests = detailed_results_frame.loc[
            detailed_results_frame["status"] == "fail",
            "label",
        ].astype(str).tolist()
    return pd.DataFrame(
        [
            {
                "run_id": result.run_id,
                "started_at_utc": result.started_at.isoformat(),
                "model_name": result.model_bundle.display_name,
                "model_version": result.model_bundle.model_version,
                "dataset_name": result.dataset.name,
                "run_status": result.status,
                "monitoring_mode": "full_monitoring" if result.labels_available else "score_only",
                "score_column": result.score_column or "n/a",
                "row_count": len(raw_dataframe),
                "test_count": len(detailed_results_frame),
                "pass_count": result.pass_count,
                "fail_count": result.fail_count,
                "na_count": result.na_count,
                "failed_tests": ", ".join(failed_tests) if failed_tests else "none",
                "reviewer_exception_count": len(result.reviewer_exceptions),
            }
        ]
    )


def _build_reviewer_exception_summary_frame(result: MonitoringRunResult) -> pd.DataFrame:
    rows = []
    result_lookup = (
        result.results_frame.set_index("test_id").to_dict(orient="index")
        if not result.results_frame.empty
        else {}
    )
    for test_id, payload in result.reviewer_exceptions.items():
        test_result = result_lookup.get(test_id, {})
        rows.append(
            {
                "test_id": test_id,
                "label": test_result.get("label", test_id),
                "status": test_result.get("status", "n/a"),
                "disposition": payload.get("disposition", ""),
                "rationale": payload.get("rationale", ""),
                "reviewer_note": result.reviewer_notes.get(test_id, ""),
            }
        )
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {
                "test_id": "none",
                "label": "No reviewer exceptions recorded",
                "status": "n/a",
                "disposition": "none",
                "rationale": "",
                "reviewer_note": "",
            }
        ]
    )


def _build_test_enablement_profile_frame(thresholds: list[ThresholdRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "test_id": threshold.test_id,
                "label": threshold.label,
                "category": threshold.category,
                "enabled": threshold.enabled,
                "operator": threshold.operator,
                "threshold_value": threshold.value,
                "description": threshold.description,
            }
            for threshold in thresholds
        ]
    )


def _build_category_status_summary(results_frame: pd.DataFrame) -> pd.DataFrame:
    if results_frame.empty:
        return pd.DataFrame(columns=["category", "pass", "fail", "na"])
    summary = (
        results_frame.groupby(["category", "status"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column_name in ("pass", "fail", "na"):
        if column_name not in summary.columns:
            summary[column_name] = 0
    return summary[["category", "pass", "fail", "na"]]


def _build_reference_monitoring_delta_summary(
    *,
    result: MonitoringRunResult,
    raw_dataframe: pd.DataFrame,
    support_tables: dict[str, pd.DataFrame],
    detailed_results_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    status_lookup = detailed_results_frame.set_index("test_id")["status"].to_dict()

    reference_row_count = result.model_bundle.reference_row_count
    monitoring_row_count = float(len(raw_dataframe))
    rows.append(
        _delta_row(
            metric="row_count",
            reference_value=float(reference_row_count) if reference_row_count is not None else None,
            monitoring_value=monitoring_row_count,
            status=status_lookup.get("row_count_ratio", "na"),
        )
    )

    missingness = support_tables.get("missingness_summary", pd.DataFrame())
    if not missingness.empty:
        reference_missingness = (
            float(missingness["reference_missingness_pct"].max())
            if "reference_missingness_pct" in missingness.columns
            and missingness["reference_missingness_pct"].notna().any()
            else None
        )
        monitoring_missingness = (
            float(missingness["current_missingness_pct"].max())
            if "current_missingness_pct" in missingness.columns
            and missingness["current_missingness_pct"].notna().any()
            else None
        )
        rows.append(
            _delta_row(
                metric="max_missingness_pct",
                reference_value=reference_missingness,
                monitoring_value=monitoring_missingness,
                status=status_lookup.get("max_feature_missingness_pct", "na"),
            )
        )

    score_profile = support_tables.get("score_profile", pd.DataFrame())
    if not score_profile.empty:
        reference_row = score_profile.loc[score_profile["population"] == "reference"]
        monitoring_row = score_profile.loc[score_profile["population"] == "monitoring"]
        reference_score_mean = (
            float(reference_row["score_mean"].iloc[0])
            if not reference_row.empty and pd.notna(reference_row["score_mean"].iloc[0])
            else None
        )
        monitoring_score_mean = (
            float(monitoring_row["score_mean"].iloc[0])
            if not monitoring_row.empty and pd.notna(monitoring_row["score_mean"].iloc[0])
            else None
        )
        rows.append(
            _delta_row(
                metric="score_mean",
                reference_value=reference_score_mean,
                monitoring_value=monitoring_score_mean,
                status=status_lookup.get("score_psi", "na"),
            )
        )
        reference_score_std = (
            float(reference_row["score_std"].iloc[0])
            if not reference_row.empty and pd.notna(reference_row["score_std"].iloc[0])
            else None
        )
        monitoring_score_std = (
            float(monitoring_row["score_std"].iloc[0])
            if not monitoring_row.empty and pd.notna(monitoring_row["score_std"].iloc[0])
            else None
        )
        rows.append(
            _delta_row(
                metric="score_std",
                reference_value=reference_score_std,
                monitoring_value=monitoring_score_std,
                status=status_lookup.get("score_psi", "na"),
            )
        )

    return pd.DataFrame(rows)


def _delta_row(
    *,
    metric: str,
    reference_value: float | None,
    monitoring_value: float | None,
    status: str,
) -> dict[str, object]:
    delta_value = None
    if reference_value is not None and monitoring_value is not None:
        delta_value = monitoring_value - reference_value
    return {
        "metric": metric,
        "reference_value": reference_value,
        "monitoring_value": monitoring_value,
        "delta_value": delta_value,
        "status": status,
    }


def _build_run_fingerprint_frame(
    *,
    result: MonitoringRunResult,
    thresholds: list[ThresholdRecord],
) -> pd.DataFrame:
    threshold_payload = [asdict(record) for record in thresholds]
    threshold_hash = sha256(
        json.dumps(threshold_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    dataset_fingerprint = build_dataset_fingerprint_frame(result.dataset)
    dataset_hash = (
        dataset_fingerprint["sha256"].iloc[0]
        if not dataset_fingerprint.empty
        else None
    )
    model_hash = _extract_model_hash(build_bundle_fingerprint_frame(result.model_bundle))
    run_payload = {
        "run_id": result.run_id,
        "started_at": result.started_at.isoformat(),
        "bundle_id": result.model_bundle.bundle_id,
        "dataset_id": result.dataset.dataset_id,
        "dataset_sha256": dataset_hash,
        "model_sha256": model_hash,
        "threshold_snapshot_sha256": threshold_hash,
        "status": result.status,
    }
    run_hash = sha256(
        json.dumps(run_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return pd.DataFrame(
        [
            {"field": "run_id", "value": result.run_id},
            {"field": "started_at_utc", "value": result.started_at.isoformat()},
            {"field": "run_root", "value": str(result.run_root)},
            {"field": "bundle_id", "value": result.model_bundle.bundle_id},
            {"field": "dataset_id", "value": result.dataset.dataset_id},
            {"field": "dataset_sha256", "value": dataset_hash},
            {"field": "model_sha256", "value": model_hash},
            {"field": "threshold_snapshot_sha256", "value": threshold_hash},
            {"field": "run_sha256", "value": run_hash},
            {"field": "report_generated_at_utc", "value": datetime.now(UTC).isoformat()},
        ]
    )


def _build_failure_diagnostics_frame(result: MonitoringRunResult) -> pd.DataFrame:
    if not result.failure_context and not result.error_message:
        return pd.DataFrame(columns=["field", "value"])
    rows = [
        {"field": "status", "value": result.status},
        {"field": "failure_stage", "value": result.failure_stage or "n/a"},
        {"field": "error_message", "value": result.error_message or "n/a"},
    ]
    for key, value in result.failure_context.items():
        rows.append({"field": key, "value": value})
    return pd.DataFrame(rows)


def _extract_model_hash(bundle_fingerprint: pd.DataFrame) -> str | None:
    if bundle_fingerprint.empty:
        return None
    match = bundle_fingerprint.loc[
        bundle_fingerprint["artifact_name"] == "model_artifact", "sha256"
    ]
    if match.empty:
        return None
    return str(match.iloc[0]) if pd.notna(match.iloc[0]) else None


def _benchmark_text(row: pd.Series) -> str:
    threshold_value = row.get("threshold_value")
    operator = str(row.get("operator") or "").strip()
    if threshold_value is None or pd.isna(threshold_value):
        return "n/a"
    return f"{operator} {_format_numeric(float(threshold_value))}"


def _interpret_result_row(row: pd.Series) -> str:
    status = str(row.get("status") or "")
    observed_value = row.get("observed_value")
    benchmark = _benchmark_text(row)
    if status == "pass":
        return f"Observed {_format_numeric(observed_value)} met the {benchmark} benchmark."
    if status == "fail":
        return f"Observed {_format_numeric(observed_value)} breached the {benchmark} benchmark."
    return str(row.get("detail") or "No evaluation was produced.")


def _sanitize_sheet_name(name: str) -> str:
    cleaned = "".join(character if character.isalnum() or character == "_" else "_" for character in name)
    return cleaned[:31] or "sheet"


def _format_numeric(value: float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    if abs(value) >= 100 or value == 0:
        return f"{value:.4f}"
    return f"{value:.6f}"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _build_reviewer_package(
    *,
    reviewer_package_path: Path,
    files: list[Path | None],
) -> None:
    with zipfile.ZipFile(reviewer_package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path is None or not path.exists():
                continue
            archive.write(path, arcname=path.name)


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
