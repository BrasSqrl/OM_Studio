"""Streamlit UI for narrow approved-model monitoring workflows."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from quant_studio_monitoring import (
    create_demo_assets,
    discover_datasets,
    discover_model_bundles,
    execute_monitoring_run,
    load_workspace_config,
)
from quant_studio_monitoring.monitoring_pipeline import MonitoringRunResult, ScoringRuntimeOptions
from quant_studio_monitoring.presentation import inject_styles, render_header
from quant_studio_monitoring.registry import (
    MonitoringMetadata,
    build_bundle_compatibility_frame,
    build_bundle_fingerprint_frame,
    build_dataset_fingerprint_frame,
    build_input_template_workbook_bytes,
    build_model_bundle_intake_checklist_frame,
    build_reference_baseline_diagnostics_frame,
    build_review_completeness_frame,
    load_monitoring_metadata,
    read_dataset,
    read_dataset_columns,
    save_monitoring_metadata,
    summarize_dataset_contract,
)
from quant_studio_monitoring.run_comparison import (
    build_run_comparison_frame,
    build_run_comparison_html,
    build_run_comparison_summary_frame,
    build_run_comparison_workbook_bytes,
)
from quant_studio_monitoring.run_history import load_run_index
from quant_studio_monitoring.streamlit_ui.state import initialize_session_state
from quant_studio_monitoring.test_catalog import TEST_HELP_GUIDANCE
from quant_studio_monitoring.thresholds import (
    build_threshold_workbook_bytes,
    load_threshold_audit_frame,
    load_threshold_records,
    recommended_threshold_records,
    save_threshold_records,
    threshold_records_from_frame,
    threshold_records_from_workbook_bytes,
    threshold_records_to_frame,
)


@st.cache_data(show_spinner=False)
def _load_dataset_frame(path_text: str) -> pd.DataFrame:
    return read_dataset(Path(path_text))


def _render_sidebar(workspace, bundles, datasets) -> tuple[bool, str]:
    latest_run = st.session_state.get("latest_run")
    runnable_bundles = sum(bundle.is_compliant for bundle in bundles)
    session_runs = len(st.session_state["run_history"])
    st.session_state.setdefault("disable_individual_visual_exports", False)
    st.session_state.setdefault("artifact_profile", "full")

    with st.sidebar:
        st.markdown(
            f"""
            <section class="sidepanel-card">
              <div class="sidepanel-kicker">Workspace</div>
              <h3>Monitoring Inbox</h3>
              <p>Drop approved bundles and monitoring datasets into the watched folders, then refresh.</p>
              <div class="mini-stat-grid">
                <div class="mini-stat">
                  <span class="mini-stat-label">Models</span>
                  <strong class="mini-stat-value">{len(bundles)}</strong>
                </div>
                <div class="mini-stat">
                  <span class="mini-stat-label">Runnable</span>
                  <strong class="mini-stat-value">{runnable_bundles}</strong>
                </div>
                <div class="mini-stat">
                  <span class="mini-stat-label">Data Files</span>
                  <strong class="mini-stat-value">{len(datasets)}</strong>
                </div>
              </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Refresh Registry", type="primary", use_container_width=True):
            _load_dataset_frame.clear()
            st.rerun()

        if st.button("Generate Demo Assets", type="secondary", use_container_width=True):
            created = create_demo_assets(workspace)
            _load_dataset_frame.clear()
            st.session_state["flash_message"] = (
                f"Demo bundle created at {created['bundle_root']} and demo dataset created at {created['dataset_path']}."
            )
            st.rerun()

        st.markdown(
            """
            <section class="sidepanel-card">
              <div class="sidepanel-kicker">Runtime</div>
              <h3>Artifact Export Controls</h3>
              <p>Use this to skip per-figure HTML and PNG exports from the scoring bundle while keeping stitched reports and workbooks.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        disable_individual_visual_exports = st.toggle(
            "Disable Individual PNG/HTML Figure Files",
            value=bool(st.session_state["disable_individual_visual_exports"]),
            help=(
                "Turns off per-figure HTML and PNG exports from the scoring bundle. "
                "The stitched report and workbook remain enabled."
            ),
        )
        st.session_state["disable_individual_visual_exports"] = (
            disable_individual_visual_exports
        )
        st.caption(
            "Recommended for faster reruns when standalone figure files are not needed."
        )
        artifact_profile = st.selectbox(
            "Artifact Profile",
            options=["full", "reviewer", "minimal"],
            index=["full", "reviewer", "minimal"].index(
                st.session_state.get("artifact_profile", "full")
            ),
            help=(
                "Full writes every support artifact. Reviewer keeps the main reviewer package. "
                "Minimal keeps HTML and Excel but skips low-value support files and the zip package."
            ),
        )
        st.session_state["artifact_profile"] = artifact_profile

        st.markdown(
            f"""
            <section class="sidepanel-card">
              <div class="sidepanel-kicker">Session</div>
              <h3>Current Review State</h3>
              <p>{session_runs} monitoring run(s) executed in this app session.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        if isinstance(latest_run, MonitoringRunResult):
            _render_status_chip_row(
                [
                    (latest_run.status.title(), latest_run.status),
                    (f"Failed {latest_run.fail_count}", "fail"),
                    (f"N/A {latest_run.na_count}", "na"),
                ]
            )
            st.caption(
                f"Latest run: {latest_run.model_bundle.display_name} on {latest_run.dataset.name}"
            )

        with st.expander("Watched Directories", expanded=False):
            st.markdown(
                "\n".join(
                    [
                        f"- `models/`: `{workspace.models_root}`",
                        f"- `incoming_data/`: `{workspace.incoming_data_root}`",
                        f"- `thresholds/`: `{workspace.thresholds_root}`",
                        f"- `runs/`: `{workspace.runs_root}`",
                    ]
                )
            )
    return bool(st.session_state["disable_individual_visual_exports"]), str(
        st.session_state["artifact_profile"]
    )


def _render_readiness_banner(bundle, dataset, contract_summary) -> None:
    state, title, reason = _resolve_readiness_state(bundle, dataset, contract_summary)
    chips: list[tuple[str, str]] = [
        (f"Bundle {bundle.display_name}", "na"),
        (f"Version {bundle.model_version}", "na"),
    ]
    if dataset is not None:
        chips.append((f"Dataset {dataset.name}", "na"))
    if bundle.warnings and state == "ready":
        chips.append((f"Intake warnings {len(bundle.warnings)}", "warning"))
    if contract_summary is not None and contract_summary.score_only_run and state == "score_only":
        chips.append(("Realized-performance tests N/A", "score_only"))

    st.markdown(
        f"""
        <section class="readiness-shell">
          <div class="readiness-banner readiness-banner--{state}">
            <div class="readiness-kicker">Execution Readiness</div>
            <h3>{escape(title)}</h3>
            <p>{escape(reason)}</p>
            <div class="status-chip-row">
              {''.join(_build_status_chip_html(label, tone) for label, tone in chips)}
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _resolve_readiness_state(bundle, dataset, contract_summary) -> tuple[str, str, str]:
    if not bundle.is_compliant:
        return (
            "blocked",
            "Blocked",
            "The selected bundle is not runnable for raw-data monitoring reruns.",
        )
    if dataset is None:
        return (
            "blocked",
            "Blocked",
            "Select a monitoring dataset from the watched inbox to enable execution.",
        )
    if contract_summary is None:
        return (
            "blocked",
            "Blocked",
            "The dataset contract has not been evaluated yet.",
        )
    if contract_summary.hard_failures:
        return (
            "blocked",
            "Blocked",
            contract_summary.hard_failures[0],
        )
    if contract_summary.score_only_run:
        return (
            "score_only",
            "Score-Only",
            "The dataset can run, but labels are unavailable so realized-performance tests will be N/A.",
        )
    if contract_summary.warnings:
        return (
            "ready",
            "Ready To Run",
            contract_summary.warnings[0],
        )
    return (
        "ready",
        "Ready To Run",
        "The selected bundle and dataset satisfy the saved monitoring contract.",
    )


def _render_status_chip_row(chips: list[tuple[str, str]]) -> None:
    visible_chips = [item for item in chips if item[0]]
    if not visible_chips:
        return
    st.markdown(
        f"""
        <div class="status-chip-row">
          {''.join(_build_status_chip_html(label, tone) for label, tone in visible_chips)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_status_chip_html(label: str, tone: str) -> str:
    return (
        f'<span class="status-chip status-chip--{escape(_status_tone(tone))}">'
        f"{escape(label)}</span>"
    )


def _status_tone(value: str) -> str:
    normalized = str(value).strip().lower().replace(" ", "_")
    return {
        "pass": "pass",
        "ready": "ready",
        "completed": "ready",
        "warning": "warning",
        "ready_with_warnings": "warning",
        "score_only": "score_only",
        "fail": "fail",
        "error": "fail",
        "not_ready": "blocked",
        "blocked": "blocked",
        "failed": "blocked",
        "na": "na",
        "n/a": "na",
        "extra": "warning",
        "missing_required": "fail",
        "missing_optional": "warning",
    }.get(normalized, "na")


def _status_chip_pairs_from_series(series: pd.Series) -> list[tuple[str, str]]:
    if series.empty:
        return []
    counts = series.astype(str).str.lower().value_counts()
    order = ("pass", "warning", "fail", "na")
    pairs: list[tuple[str, str]] = []
    for status in order:
        count = int(counts.get(status, 0))
        if count:
            pairs.append((f"{_status_label(status)} {count}", status))
    return pairs


def _status_label(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", " ")
    return {
        "pass": "Pass",
        "warning": "Warning",
        "fail": "Fail",
        "na": "N/A",
        "ready_with_warnings": "Ready With Warnings",
        "not_ready": "Not Ready",
        "score_only": "Score-Only",
    }.get(normalized, normalized.title())


def _render_empty_state(*, title: str, message: str, bullets: list[str]) -> None:
    bullet_html = "".join(f"<li>{escape(item)}</li>" for item in bullets)
    st.markdown(
        f"""
        <section class="empty-state-card">
          <div class="empty-state-kicker">Guidance</div>
          <h3>{escape(title)}</h3>
          <p>{escape(message)}</p>
          <ul>{bullet_html}</ul>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_stage_intro(*, number: int, title: str, description: str) -> None:
    st.markdown(
        f"""
        <section class="workflow-stage-card">
          <div class="workflow-stage-badge">{number}</div>
          <div class="workflow-stage-copy">
            <div class="workflow-stage-kicker">Decision Workflow</div>
            <h3>{escape(title)}</h3>
            <p>{escape(description)}</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_summary_tiles(tiles: list[dict[str, str]]) -> None:
    card_html = "".join(
        f"""
        <div class="summary-tile summary-tile--{escape(tile.get("tone", "neutral"))}">
          <span>{escape(tile["label"])}</span>
          <strong>{escape(tile["value"])}</strong>
          <p>{escape(tile["detail"])}</p>
        </div>
        """
        for tile in tiles
    )
    st.markdown(
        f"""
        <section class="summary-tile-grid">
          {card_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_workspace_navigation_hint() -> None:
    st.markdown(
        """
        <section class="workspace-nav-strip">
          <div>
            <div class="workspace-nav-kicker">Primary Navigation</div>
            <p>
              Use the task tabs below as the main workspace: Overview for context, Validate for
              contract checks, Thresholds for saved benchmarks, Run for execution, and History for
              session comparison.
            </p>
          </div>
          <div class="workspace-nav-tip">Select -> Validate -> Run -> Download</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_control_deck(*, workspace, bundles, datasets):
    st.markdown(
        """
        <section class="workspace-toolbar-card">
          <div class="workspace-toolbar-kicker">Monitoring Workspace</div>
          <h3>Configure the active workflow.</h3>
        </section>
        """,
        unsafe_allow_html=True,
    )

    selection_columns = st.columns([1.2, 1.2, 0.9, 1.05])
    bundle_lookup = {f"{bundle.display_name} | {bundle.model_version}": bundle for bundle in bundles}
    with selection_columns[0]:
        bundle = bundle_lookup[st.selectbox("Model Bundle", list(bundle_lookup.keys()))]

    dataset = None
    dataset_columns: list[str] = []
    contract_summary = None
    selected_segment = None
    with selection_columns[1]:
        if datasets:
            dataset_lookup = {
                f"{dataset_item.name} | {dataset_item.modified_at:%Y-%m-%d %H:%M}": dataset_item
                for dataset_item in datasets
            }
            dataset = dataset_lookup[
                st.selectbox("Monitoring Dataset", list(dataset_lookup.keys()))
            ]
            dataset_columns = read_dataset_columns(dataset.path)
        else:
            st.selectbox(
                "Monitoring Dataset",
                options=["No watched dataset available"],
                index=0,
                disabled=True,
            )

    with selection_columns[2]:
        if dataset_columns:
            segment_options = [""] + sorted(dataset_columns)
            default_index = 0
            if bundle.default_segment_column and bundle.default_segment_column in segment_options:
                default_index = segment_options.index(bundle.default_segment_column)
            selected_segment = st.selectbox(
                "Segment Column",
                options=segment_options,
                index=default_index,
                help="Optional. Used for segment PSI and segment mix reporting.",
            ) or None
        else:
            st.selectbox(
                "Segment Column",
                options=["No dataset loaded"],
                index=0,
                disabled=True,
            )

    toolbar_run_requested = False
    run_disabled = True
    with selection_columns[3]:
        st.caption("Execution")
        if dataset is not None:
            dataset_frame = _load_dataset_frame(str(dataset.path))
            contract_summary = summarize_dataset_contract(
                bundle,
                dataset_frame,
                segment_column=selected_segment,
                performance=workspace.performance,
            )
        run_disabled = (
            not bundle.is_compliant
            or dataset is None
            or contract_summary is None
            or bool(contract_summary.hard_failures)
        )
        toolbar_run_requested = st.button(
            "Run Monitoring Workflow",
            type="primary",
            disabled=run_disabled,
            use_container_width=True,
        )

    if dataset is not None and contract_summary is None:
        dataset_frame = _load_dataset_frame(str(dataset.path))
        contract_summary = summarize_dataset_contract(
            bundle,
            dataset_frame,
            segment_column=selected_segment,
            performance=workspace.performance,
        )

    toolbar_chips = [
        (bundle.display_name, "na"),
        (
            dataset.name if dataset is not None else "Dataset Needed",
            "ready" if dataset is not None else "blocked",
        ),
        (
            "Ready To Run"
            if contract_summary is not None and not contract_summary.hard_failures
            else "Validation Needed",
            "ready"
            if contract_summary is not None and not contract_summary.hard_failures
            else "warning",
        ),
    ]
    _render_status_chip_row(toolbar_chips)

    if not bundle.is_compliant:
        _render_empty_state(
            title="Selected Asset Is Not Runnable",
            message="This model asset was discovered, but it is not yet a compliant rerunnable bundle for OM Studio.",
            bullets=[
                f"Selected folder: {bundle.bundle_paths.root}",
                "Minimum files: quant_model.joblib, run_config.json, generated_run.py",
                "Use a full exported bundle directory instead of a standalone model file.",
            ],
        )
    elif dataset is None:
        _render_empty_state(
            title="No Monitoring Dataset Available",
            message="You can still inspect the selected bundle and configure thresholds, but monitoring cannot run until a watched CSV or Excel dataset is present.",
            bullets=[
                f"Watched folder: {workspace.incoming_data_root}",
                "Accepted formats: .csv, .xlsx, .xls",
                "The dataset should contain the raw input columns required by the selected bundle.",
            ],
        )

    active_run = _selected_run(bundle, dataset)
    _render_decision_card(bundle, dataset, contract_summary, active_run)
    return (
        bundle,
        dataset,
        dataset_columns,
        contract_summary,
        selected_segment,
        toolbar_run_requested,
    )


def _render_decision_card(bundle, dataset, contract_summary, active_run: MonitoringRunResult | None):
    state, title, reason = _resolve_decision_state(
        bundle=bundle,
        dataset=dataset,
        contract_summary=contract_summary,
        active_run=active_run,
    )
    next_action = _resolve_next_action(
        bundle=bundle,
        dataset=dataset,
        contract_summary=contract_summary,
        active_run=active_run,
    )
    chips = [
        (bundle.display_name, "na"),
        (bundle.model_version, "na"),
    ]
    if dataset is not None:
        chips.append((dataset.name, "na"))
    if contract_summary is not None and contract_summary.score_only_run:
        chips.append(("Score-Only", "score_only"))
    elif contract_summary is not None and not contract_summary.hard_failures:
        chips.append(("Ready", "ready"))

    st.markdown(
        f"""
        <section class="decision-card decision-card--{state}">
          <div class="decision-kicker">Primary Message</div>
          <div class="decision-grid">
            <div class="decision-main">
              <h3>{escape(title)}</h3>
              <p>{escape(reason)}</p>
              <div class="status-chip-row">
                {''.join(_build_status_chip_html(label, tone) for label, tone in chips)}
              </div>
            </div>
            <div class="decision-next">
              <span>Next Action</span>
              <strong>{escape(next_action)}</strong>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _resolve_decision_state(
    *,
    bundle,
    dataset,
    contract_summary,
    active_run: MonitoringRunResult | None,
) -> tuple[str, str, str]:
    if active_run is not None and active_run.status == "completed":
        return (
            "completed",
            "Run Complete",
            "Artifacts are available for download and detailed results can be reviewed in the Run tab.",
        )
    if not bundle.is_compliant:
        return (
            "blocked",
            "Bundle Not Runnable",
            "The selected asset does not meet the compliant rerunnable bundle contract.",
        )
    if dataset is None:
        return (
            "blocked",
            "Dataset Needed",
            "Select a watched monitoring dataset before validation and execution can begin.",
        )
    if contract_summary is None:
        return (
            "current",
            "Validation Pending",
            "The selected dataset has not been evaluated against the saved bundle contract yet.",
        )
    if contract_summary.hard_failures:
        return ("blocked", "Validation Blocked", contract_summary.hard_failures[0])
    if contract_summary.score_only_run:
        return (
            "score_only",
            "Score-Only Monitoring",
            "The dataset is runnable, but label-based tests will be N/A because labels are not available.",
        )
    return (
        "ready",
        "Ready To Run",
        "The selected bundle and dataset satisfy the saved monitoring contract.",
    )


def _resolve_next_action(*, bundle, dataset, contract_summary, active_run) -> str:
    if not bundle.is_compliant:
        return "Replace this asset with a full compliant bundle."
    if dataset is None:
        return "Drop a CSV or Excel file into incoming_data/, refresh, and select it."
    if contract_summary is None:
        return "Open Validate to confirm the dataset contract."
    if contract_summary.hard_failures:
        return "Open Validate and fix the blocking columns or schema issues."
    if active_run is not None and active_run.status == "completed":
        return "Open Run to download the report or workbook."
    if contract_summary.score_only_run:
        return "Run monitoring now, or add labels if realized-performance tests are needed."
    return "Review thresholds if needed, then run monitoring."


def _render_step_tracker(
    *,
    bundle,
    dataset,
    contract_summary,
    active_run: MonitoringRunResult | None,
    bundle_assets_available: bool,
) -> None:
    steps = _build_step_tracker_steps(
        bundle=bundle,
        dataset=dataset,
        contract_summary=contract_summary,
        active_run=active_run,
        bundle_assets_available=bundle_assets_available,
    )
    st.markdown(
        f"""
        <section class="step-tracker-card">
          <div class="step-tracker-kicker">Workflow</div>
          <h3>Monitoring Path</h3>
          <div class="step-tracker-grid">
            {''.join(_build_step_html(index, step) for index, step in enumerate(steps, start=1))}
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _build_step_tracker_steps(
    *,
    bundle,
    dataset,
    contract_summary,
    active_run: MonitoringRunResult | None,
    bundle_assets_available: bool,
) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    if not bundle_assets_available:
        return [
            {
                "label": "Select Bundle",
                "state": "blocked",
                "detail": "Drop a compliant bundle into models/ to begin.",
            },
            {
                "label": "Select Dataset",
                "state": "blocked",
                "detail": "The workflow stays blocked until a bundle is available.",
            },
            {
                "label": "Validate",
                "state": "blocked",
                "detail": "Validation starts after bundle and dataset selection.",
            },
            {
                "label": "Run",
                "state": "blocked",
                "detail": "Execution is unavailable until earlier steps are complete.",
            },
            {
                "label": "Download",
                "state": "blocked",
                "detail": "Reports and workbook downloads appear after a completed run.",
            },
        ]

    bundle_ready = bundle is not None
    dataset_ready = dataset is not None
    validation_ready = (
        contract_summary is not None and not contract_summary.hard_failures
    )
    run_ready = validation_ready and bundle is not None and bundle.is_compliant
    download_ready = (
        active_run is not None
        and active_run.status == "completed"
        and (
            (active_run.artifacts.get("report") and Path(active_run.artifacts["report"]).exists())
            or (
                active_run.artifacts.get("workbook")
                and Path(active_run.artifacts["workbook"]).exists()
            )
        )
    )

    steps.append(
        {
            "label": "Select Bundle",
            "state": "complete" if bundle_ready else "current",
            "detail": (
                f"{bundle.display_name} | {bundle.model_version}"
                if bundle_ready
                else "Choose a discovered compliant bundle."
            ),
        }
    )
    steps.append(
        {
            "label": "Select Dataset",
            "state": (
                "blocked"
                if not bundle_ready
                else "complete"
                if dataset_ready
                else "current"
            ),
            "detail": dataset.name if dataset_ready else "Choose a watched CSV or Excel dataset.",
        }
    )
    if contract_summary is None:
        validation_state = "blocked" if not dataset_ready else "current"
        validation_detail = "Validation starts after a dataset is selected."
    elif contract_summary.hard_failures:
        validation_state = "blocked"
        validation_detail = contract_summary.hard_failures[0]
    elif contract_summary.score_only_run:
        validation_state = "complete"
        validation_detail = "Valid for score-only monitoring. Label-based tests will be N/A."
    elif contract_summary.warnings:
        validation_state = "complete"
        validation_detail = contract_summary.warnings[0]
    else:
        validation_state = "complete"
        validation_detail = "Dataset satisfies the saved monitoring contract."
    steps.append(
        {
            "label": "Validate",
            "state": validation_state,
            "detail": validation_detail,
        }
    )
    steps.append(
        {
            "label": "Run",
            "state": (
                "complete"
                if active_run is not None
                else "current"
                if run_ready
                else "blocked"
            ),
            "detail": (
                f"Latest run {active_run.run_id}"
                if active_run is not None
                else "Execute monitoring for the current bundle and dataset."
            ),
        }
    )
    steps.append(
        {
            "label": "Download",
            "state": "complete" if download_ready else "blocked" if active_run is None else "current",
            "detail": (
                "Artifacts are ready to download."
                if download_ready
                else "Run monitoring to create the HTML report and workbook."
            ),
        }
    )
    return steps


def _build_step_html(index: int, step: dict[str, str]) -> str:
    state = step["state"]
    marker = {"complete": "Done", "current": "Next", "blocked": "Blocked"}.get(state, "Step")
    return f"""
    <div class="step step--{escape(state)}">
      <div class="step-number">{index}</div>
      <div class="step-body">
        <div class="step-label-row">
          <strong>{escape(step['label'])}</strong>
          <span class="step-marker">{escape(marker)}</span>
        </div>
        <span>{escape(step['detail'])}</span>
      </div>
    </div>
    """


def _selected_run(bundle, dataset) -> MonitoringRunResult | None:
    latest_run = st.session_state.get("latest_run")
    if not isinstance(latest_run, MonitoringRunResult):
        return None
    if bundle is None or dataset is None:
        return None
    if latest_run.model_bundle.bundle_id != bundle.bundle_id:
        return None
    if latest_run.dataset.dataset_id != dataset.dataset_id:
        return None
    return latest_run


def _apply_column_search(frame: pd.DataFrame, search_text: str, columns: list[str]) -> pd.DataFrame:
    if frame.empty or not search_text.strip():
        return frame
    pattern = search_text.strip().lower()
    mask = pd.Series(False, index=frame.index)
    for column in columns:
        if column not in frame.columns:
            continue
        mask = mask | frame[column].astype(str).str.lower().str.contains(pattern, na=False)
    return frame.loc[mask].copy()


def _sort_dataset_preview_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    priority = {
        "missing_required": 0,
        "required": 1,
        "missing_optional": 2,
        "extra": 3,
        "optional": 4,
    }
    status_priority = {"fail": 0, "warning": 1, "pass": 2, "na": 3}
    sorted_frame = frame.copy()
    sorted_frame["_contract_priority"] = (
        sorted_frame["contract_match"].map(priority).fillna(99)
    )
    sorted_frame["_status_priority"] = sorted_frame["status"].map(status_priority).fillna(99)
    sorted_frame = sorted_frame.sort_values(
        by=["_contract_priority", "_status_priority", "dataset_column"],
        kind="stable",
    )
    return sorted_frame.drop(columns=["_contract_priority", "_status_priority"])


def _sort_column_checks_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    status_priority = {"fail": 0, "warning": 1, "pass": 2, "na": 3}
    sorted_frame = frame.copy()
    sorted_frame["_required_priority"] = (~sorted_frame["required_for_run"]).astype(int)
    sorted_frame["_status_priority"] = sorted_frame["status"].map(status_priority).fillna(99)
    name_column = "column_name" if "column_name" in sorted_frame.columns else sorted_frame.columns[0]
    sorted_frame = sorted_frame.sort_values(
        by=["_required_priority", "_status_priority", name_column],
        kind="stable",
    )
    return sorted_frame.drop(columns=["_required_priority", "_status_priority"])


def _frames_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left.columns) != list(right.columns):
        return False
    left_normalized = left.reset_index(drop=True).astype(object).where(pd.notna(left), "__NA__")
    right_normalized = right.reset_index(drop=True).astype(object).where(pd.notna(right), "__NA__")
    return left_normalized.equals(right_normalized)


def _build_inferred_dtype_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column_name in dataframe.columns.astype(str):
        series = dataframe[column_name]
        non_null_series = series.dropna()
        rows.append(
            {
                "column_name": column_name,
                "inferred_dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
                "sample_value": str(non_null_series.iloc[0]) if not non_null_series.empty else "n/a",
            }
        )
    return pd.DataFrame(rows)


def _render_test_help_drawer(records, *, key_prefix: str) -> None:
    with st.expander("What These Tests Mean", expanded=False):
        search_text = st.text_input(
            "Find Test",
            key=f"{key_prefix}_test_help_search",
            help="Search by test name, category, or concept.",
        )
        for record in records:
            searchable = " ".join(
                [record.label, record.category, record.description, record.test_id]
            ).lower()
            if search_text.strip() and search_text.strip().lower() not in searchable:
                continue
            guidance = TEST_HELP_GUIDANCE.get(record.test_id, {})
            fail_implies = guidance.get(
                "fail_implies",
                "A fail means the observed monitoring value falls outside the saved benchmark for this model.",
            )
            na_when = guidance.get(
                "na_when",
                "This is N/A when the threshold is disabled or the required inputs for the test are unavailable.",
            )
            with st.expander(f"{record.label} ({record.category})", expanded=False):
                st.markdown(f"**Measures**: {record.description}")
                st.markdown(f"**Fail Means**: {fail_implies}")
                st.markdown(f"**N/A When**: {na_when}")


def main() -> None:
    st.set_page_config(
        page_title="OM Studio",
        page_icon="Q",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    render_header()
    initialize_session_state()

    flash_message = st.session_state.pop("flash_message", None)
    if flash_message:
        st.success(flash_message)

    workspace = load_workspace_config()
    bundles = discover_model_bundles(workspace)
    datasets = discover_datasets(workspace)
    disable_individual_visual_exports, artifact_profile = _render_sidebar(
        workspace,
        bundles,
        datasets,
    )

    if not bundles:
        _render_empty_state(
            title="No Compliant Bundle Available",
            message="OM Studio only monitors compliant rerunnable bundles. Drop a full exported bundle into the watched models folder, then refresh.",
            bullets=[
                f"Watched folder: {workspace.models_root}",
                "Minimum files: quant_model.joblib, run_config.json, generated_run.py",
                "Recommended extras: predictions.csv, input_snapshot.csv, artifact_manifest.json",
            ],
        )
        return

    # Focused Workspace Pattern: one control deck, one decision card, task tabs below.
    (
        bundle,
        dataset,
        dataset_columns,
        contract_summary,
        selected_segment,
        toolbar_run_requested,
    ) = _render_control_deck(
        workspace=workspace,
        bundles=bundles,
        datasets=datasets,
    )

    tabs = st.tabs(
        [
            "Overview",
            "Validate",
            "Thresholds",
            "Run",
            "History",
        ]
    )

    with tabs[0]:
        _render_bundle_intake(bundle, dataset, dataset_columns, contract_summary)

    with tabs[1]:
        _render_pre_run_contract(bundle, dataset, contract_summary)

    with tabs[2]:
        _render_threshold_editor(workspace, bundle)

    with tabs[3]:
        _render_run_tab(
            workspace=workspace,
            bundle=bundle,
            dataset=dataset,
            datasets=datasets,
            contract_summary=contract_summary,
            selected_segment=selected_segment,
            disable_individual_visual_exports=disable_individual_visual_exports,
            artifact_profile=artifact_profile,
            toolbar_run_requested=toolbar_run_requested,
        )

    with tabs[4]:
        _render_session_history(workspace)


def _render_bundle_intake(bundle, dataset, dataset_columns: list[str], contract_summary) -> None:
    # Focused Workspace Pattern: one overview summary first, detail in expanders second.
    _render_stage_intro(
        number=1,
        title="Review The Selected Bundle And Monitoring Dataset",
        description=(
            "Keep the current bundle, selected dataset, and reviewer context visible here. "
            "Detailed contract checks stay in Validate and execution stays in Run."
        ),
    )
    _render_summary_tiles(
        [
            {
                "label": "Model",
                "value": bundle.display_name,
                "detail": bundle.model_version,
                "tone": "neutral",
            },
            {
                "label": "Bundle Status",
                "value": bundle.readiness_label,
                "detail": bundle.compatibility_label,
                "tone": "ready" if bundle.is_compliant else "blocked",
            },
            {
                "label": "Dataset",
                "value": dataset.name if dataset is not None else "Not Selected",
                "detail": (
                    "Monitoring data selected."
                    if dataset is not None
                    else "Choose a watched CSV or Excel file."
                ),
                "tone": "neutral" if dataset is not None else "warning",
            },
        ]
    )

    overview_rows = [
        {"field": "Model name", "value": bundle.display_name},
        {"field": "Model version", "value": bundle.model_version},
        {"field": "Bundle status", "value": bundle.readiness_label},
        {"field": "Compatibility", "value": bundle.compatibility_label},
        {
            "field": "Contract version",
            "value": (
                f"{bundle.monitoring_contract_version or 'not declared'} "
                f"({bundle.monitoring_contract_version_status})"
            ),
        },
        {"field": "Review readiness", "value": bundle.review_label},
        {"field": "Target mode", "value": bundle.target_mode or "n/a"},
        {"field": "Model type", "value": bundle.model_type or "n/a"},
        {"field": "Dataset", "value": dataset.name if dataset is not None else "not selected"},
    ]
    if contract_summary is not None:
        execution_mode = "score_only" if contract_summary.score_only_run else "full_monitoring"
        overview_rows.extend(
            [
                {"field": "Validation state", "value": contract_summary.overall_status},
                {"field": "Execution mode", "value": execution_mode},
            ]
        )
    st.dataframe(pd.DataFrame(overview_rows), hide_index=True, use_container_width=True)
    _render_status_chip_row(
        [
            (bundle.readiness_label, bundle.readiness_status),
            (bundle.compatibility_label, "warning" if bundle.compatibility_status == "compatible_with_warnings" else "blocked" if not bundle.is_compatible else "ready"),
            (bundle.review_label, "warning" if not bundle.is_review_complete else "ready"),
            (
                "Dataset Selected" if dataset is not None else "Dataset Needed",
                "ready" if dataset is not None else "blocked",
            ),
            (
                "Score-Only"
                if contract_summary is not None and contract_summary.score_only_run
                else "Full Monitoring"
                if contract_summary is not None and not contract_summary.hard_failures
                else "",
                "score_only"
                if contract_summary is not None and contract_summary.score_only_run
                else "ready",
            ),
        ]
    )

    action_columns = st.columns([1.1, 1.0])
    template_bytes = build_input_template_workbook_bytes(bundle, include_example_row=True)
    with action_columns[0]:
        st.download_button(
            "Download Input Template",
            data=template_bytes,
            file_name=f"{bundle.bundle_id}_input_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with action_columns[1]:
        st.caption(
            "Use the template when you need a bundle-specific starter file with the required raw input columns."
        )

    with st.expander("Monitoring Metadata", expanded=False):
        _render_metadata_editor(bundle)

    with st.expander("Bundle Details", expanded=False):
        intake = pd.DataFrame(
            [
                {"field": "Metadata source", "value": bundle.metadata_source},
                {"field": "Export version", "value": bundle.export_version or "unknown"},
                {
                    "field": "Monitoring contract version",
                    "value": bundle.monitoring_contract_version or "not declared",
                },
                {
                    "field": "Monitoring contract version status",
                    "value": bundle.monitoring_contract_version_status,
                },
                {"field": "Model owner", "value": bundle.model_owner or "n/a"},
                {"field": "Business purpose", "value": bundle.business_purpose or "n/a"},
                {"field": "Portfolio", "value": bundle.portfolio_name or "n/a"},
                {"field": "Segment name", "value": bundle.segment_name or "n/a"},
                {
                    "field": "Approval status",
                    "value": bundle.monitoring_metadata.approval_status or "n/a",
                },
                {
                    "field": "Approval date",
                    "value": bundle.monitoring_metadata.approval_date or "n/a",
                },
                {"field": "Bundle path", "value": str(bundle.bundle_paths.root)},
                {"field": "Model artifact", "value": str(bundle.bundle_paths.model_path)},
                {"field": "Reference rows", "value": bundle.reference_row_count or "n/a"},
                {
                    "field": "Reference score column",
                    "value": bundle.reference_score_column or "n/a",
                },
            ]
        )
        st.dataframe(intake, hide_index=True, use_container_width=True)

    with st.expander("Compatibility And Review Readiness", expanded=False):
        compatibility_frame = build_bundle_compatibility_frame(bundle)
        review_frame = build_review_completeness_frame(bundle)
        intake_checklist = build_model_bundle_intake_checklist_frame(bundle)
        st.markdown("#### Intake Checklist")
        st.dataframe(intake_checklist, hide_index=True, use_container_width=True)
        st.markdown("#### Compatibility Findings")
        if compatibility_frame.empty:
            st.caption("No compatibility findings were recorded for this bundle.")
        else:
            st.dataframe(compatibility_frame, hide_index=True, use_container_width=True)
        st.markdown("#### Metadata Completeness")
        st.dataframe(review_frame, hide_index=True, use_container_width=True)
        if bundle.review_metadata_gaps:
            st.warning(
                "Review-incomplete metadata fields: " + ", ".join(bundle.review_metadata_gaps)
            )

    with st.expander("Reference Baseline Diagnostics", expanded=False):
        st.dataframe(
            build_reference_baseline_diagnostics_frame(bundle),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Input Contract Snapshot", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown("#### Required Raw Input Columns")
            st.dataframe(
                pd.DataFrame({"required_input_column": bundle.expected_input_columns}),
                hide_index=True,
                use_container_width=True,
            )
        with right:
            st.markdown("#### Optional Input Columns")
            st.dataframe(
                pd.DataFrame({"optional_input_column": bundle.optional_input_columns}),
                hide_index=True,
                use_container_width=True,
            )
        if dataset is not None:
            st.caption(f"Selected dataset exposes {len(dataset_columns)} column(s).")

    with st.expander("Bundle Readiness Findings", expanded=False):
        if bundle.readiness_checks:
            checks_frame = pd.DataFrame(
                [
                    {
                        "severity": check.severity,
                        "item": check.item,
                        "detail": check.detail,
                    }
                    for check in bundle.readiness_checks
                ]
            )
            st.dataframe(checks_frame, hide_index=True, use_container_width=True)
        else:
            st.caption("No readiness findings were recorded for this bundle.")

    with st.expander("Bundle Fingerprint", expanded=False):
        st.dataframe(
            build_bundle_fingerprint_frame(bundle),
            hide_index=True,
            use_container_width=True,
        )


def _build_dataset_column_preview_frame(bundle, dataset_columns: list[str]) -> pd.DataFrame:
    frame_columns = [
        "dataset_column",
        "present_in_dataset",
        "contract_match",
        "status",
        "role",
        "expected_dtype",
        "detail",
    ]
    spec_lookup = {spec.source_name: spec for spec in bundle.column_specs}
    rows = []
    for column_name in dataset_columns:
        spec = spec_lookup.get(column_name)
        if spec is None:
            rows.append(
                {
                    "dataset_column": column_name,
                    "present_in_dataset": True,
                    "contract_match": "extra",
                    "status": "warning",
                    "role": "not_in_bundle",
                    "expected_dtype": "n/a",
                    "detail": "Available in the dataset but not referenced by the saved bundle contract.",
                }
            )
            continue

        rows.append(
            {
                "dataset_column": column_name,
                "present_in_dataset": True,
                "contract_match": "required" if spec.required_for_run else "optional",
                "status": "pass",
                "role": spec.role,
                "expected_dtype": spec.dtype or "n/a",
                "detail": (
                    "Required by the saved bundle contract."
                    if spec.required_for_run
                    else "Optional for monitoring; missing values may downgrade the run to score-only."
                    if spec.role == "target_source"
                    else "Optional schema column for this bundle."
                ),
            }
        )

    present_columns = set(dataset_columns)
    for spec in bundle.column_specs:
        if spec.source_name in present_columns or not spec.required_for_run:
            continue
        rows.append(
            {
                "dataset_column": spec.source_name,
                "present_in_dataset": False,
                "contract_match": "missing_required",
                "status": "fail",
                "role": spec.role,
                "expected_dtype": spec.dtype or "n/a",
                "detail": "Required by the saved bundle contract but missing from the dataset preview.",
            }
        )

    label_column = bundle.label_source_column
    if label_column and label_column not in present_columns:
        label_spec = spec_lookup.get(label_column)
        rows.append(
            {
                "dataset_column": label_column,
                "present_in_dataset": False,
                "contract_match": "missing_optional",
                "status": "warning",
                "role": "target_source",
                "expected_dtype": label_spec.dtype if label_spec is not None else "n/a",
                "detail": "Optional label column is missing. The run can continue in score-only mode.",
            }
        )
    return pd.DataFrame(rows, columns=frame_columns)


def _render_metadata_editor(bundle) -> None:
    metadata = load_monitoring_metadata(bundle)
    st.caption(
        "This manifest is managed by the monitoring app and stored beside the selected bundle."
    )

    with st.form(key=f"metadata_{bundle.bundle_id}"):
        model_name = st.text_input("Model Name", value=metadata.model_name)
        model_version = st.text_input("Model Version", value=metadata.model_version)
        model_owner = st.text_input("Model Owner", value=metadata.model_owner)
        business_purpose = st.text_input("Business Purpose", value=metadata.business_purpose)
        portfolio_name = st.text_input("Portfolio Name", value=metadata.portfolio_name)
        segment_name = st.text_input("Segment Name", value=metadata.segment_name)
        default_segment_column = st.text_input(
            "Default Segment Column",
            value=metadata.default_segment_column,
        )
        approval_status = st.text_input("Approval Status", value=metadata.approval_status)
        approval_date = st.text_input(
            "Approval Date (YYYY-MM-DD)",
            value=metadata.approval_date,
        )
        monitoring_notes = st.text_area("Monitoring Notes", value=metadata.monitoring_notes)
        if st.form_submit_button("Save Metadata", type="primary"):
            payload = MonitoringMetadata(
                model_name=model_name.strip(),
                model_version=model_version.strip(),
                model_owner=model_owner.strip(),
                business_purpose=business_purpose.strip(),
                portfolio_name=portfolio_name.strip(),
                segment_name=segment_name.strip(),
                default_segment_column=default_segment_column.strip(),
                approval_status=approval_status.strip(),
                approval_date=approval_date.strip(),
                monitoring_notes=monitoring_notes.strip(),
            )
            saved_path = save_monitoring_metadata(bundle, payload)
            st.session_state["flash_message"] = f"Saved monitoring metadata to {saved_path}."
            st.rerun()


def _render_threshold_editor(workspace, bundle) -> None:
    _render_stage_intro(
        number=3,
        title="Set The Saved Benchmarks For Monitoring Decisions",
        description=(
            "Adjust per-test thresholds for the selected bundle, then save or revert them "
            "without leaving the workflow."
        ),
    )

    thresholds = load_threshold_records(workspace.thresholds_root, bundle)
    saved_threshold_frame = threshold_records_to_frame(thresholds)
    override_key = f"threshold_editor_override_{bundle.bundle_id}"
    revision_key = f"threshold_editor_revision_{bundle.bundle_id}"
    st.session_state.setdefault(revision_key, 0)
    editor_key = f"thresholds_{bundle.bundle_id}_{st.session_state[revision_key]}"
    threshold_frame = (
        pd.DataFrame(st.session_state[override_key])
        if override_key in st.session_state
        else saved_threshold_frame.copy(deep=True)
    )

    edited = st.data_editor(
        threshold_frame.drop(columns=["test_id"]),
        hide_index=True,
        use_container_width=True,
        key=editor_key,
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "operator": st.column_config.TextColumn("Operator", disabled=True),
            "value": st.column_config.NumberColumn("Threshold", step=0.01),
            "description": st.column_config.TextColumn("Description", disabled=True, width="large"),
            "label": st.column_config.TextColumn("Test", disabled=True, width="medium"),
            "category": st.column_config.TextColumn("Category", disabled=True),
        },
    )
    edited_with_ids = edited.copy(deep=True)
    edited_with_ids["test_id"] = threshold_frame["test_id"]
    threshold_records = threshold_records_from_frame(edited_with_ids)
    baseline_frame = saved_threshold_frame[
        ["label", "category", "operator", "value", "enabled", "description", "test_id"]
    ].copy()
    current_frame = edited_with_ids[
        ["label", "category", "operator", "value", "enabled", "description", "test_id"]
    ].copy()
    is_dirty = not _frames_match(current_frame, baseline_frame)

    _render_status_chip_row(
        [
            ("Modified" if is_dirty else "Saved", "warning" if is_dirty else "ready"),
            (f"Enabled {sum(record.enabled for record in threshold_records)}", "na"),
            (f"Tests {len(threshold_records)}", "na"),
        ]
    )

    actions = st.columns(3)
    if actions[0].button(
        "Save Thresholds",
        type="primary",
        key=f"save_{bundle.bundle_id}",
        disabled=not is_dirty,
        use_container_width=True,
    ):
        saved_path = save_threshold_records(
            workspace.thresholds_root,
            bundle,
            threshold_records,
            source="ui_threshold_editor",
        )
        st.session_state.pop(override_key, None)
        st.session_state[revision_key] += 1
        st.session_state["flash_message"] = f"Saved thresholds to {saved_path}."
        st.rerun()
    if actions[1].button(
        "Reset To Saved",
        key=f"revert_{bundle.bundle_id}",
        disabled=not is_dirty,
        use_container_width=True,
    ):
        st.session_state.pop(override_key, None)
        st.session_state[revision_key] += 1
        st.session_state["flash_message"] = "Reverted unsaved threshold edits."
        st.rerun()
    if actions[2].button(
        "Apply Recommended Defaults",
        key=f"defaults_{bundle.bundle_id}",
        use_container_width=True,
    ):
        default_frame = threshold_records_to_frame(recommended_threshold_records(bundle))
        st.session_state[override_key] = default_frame.to_dict(orient="records")
        st.session_state[revision_key] += 1
        st.session_state["flash_message"] = (
            "Loaded the recommended threshold defaults. Save to persist them for this bundle."
        )
        st.rerun()

    with st.expander("Offline Threshold Workbook", expanded=False):
        st.caption(
            "Export the current threshold profile, edit it offline, then import it back "
            "for this model bundle."
        )
        workbook_columns = st.columns(2)
        with workbook_columns[0]:
            st.download_button(
                "Download Threshold Workbook",
                data=build_threshold_workbook_bytes(threshold_records),
                file_name=f"{bundle.bundle_id}_thresholds.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with workbook_columns[1]:
            uploaded_thresholds = st.file_uploader(
                "Import Threshold Workbook",
                type=["xlsx"],
                key=f"threshold_workbook_upload_{bundle.bundle_id}",
                label_visibility="collapsed",
            )
        if uploaded_thresholds is not None:
            try:
                imported_records = threshold_records_from_workbook_bytes(
                    uploaded_thresholds.read()
                )
            except Exception as exc:
                st.error(f"Could not import threshold workbook: {exc}")
            else:
                imported_frame = threshold_records_to_frame(imported_records)
                st.session_state[override_key] = imported_frame.to_dict(orient="records")
                st.session_state[revision_key] += 1
                st.session_state["flash_message"] = (
                    "Imported thresholds from workbook. Save to persist them for this bundle."
                )
                st.rerun()

    with st.expander("Threshold Change Audit", expanded=False):
        audit_frame = load_threshold_audit_frame(workspace.thresholds_root, bundle)
        if audit_frame.empty:
            st.caption("No saved threshold changes have been recorded for this bundle.")
        else:
            st.dataframe(audit_frame.head(25), hide_index=True, use_container_width=True)

    _render_test_help_drawer(thresholds, key_prefix=f"threshold_help_{bundle.bundle_id}")


def _render_pre_run_contract(bundle, dataset, contract_summary) -> None:
    if dataset is None or contract_summary is None:
        _render_empty_state(
            title="Pre-Run Contract Review Is Waiting On A Dataset",
            message="Select a monitoring dataset to review execution readiness, column-level checks, and date coverage before scoring.",
            bullets=[
                "Choose a dataset from the watched inbox near the top of the page.",
                "Missing required columns will be pinned to the top of the validation table.",
                "Date coverage appears only when the saved bundle defines a date column.",
            ],
        )
        return

    _render_stage_intro(
        number=2,
        title="Validate The Configured Monitoring Workflow Before Execution",
        description=(
            "Review hard failures, warnings, dataset preview coverage, and date support before "
            "running the selected bundle."
        ),
    )
    _render_summary_tiles(
        [
            {
                "label": "Validation State",
                "value": contract_summary.overall_status.replace("_", " ").title(),
                "detail": "Current dataset contract result.",
                "tone": contract_summary.overall_status,
            },
            {
                "label": "Hard Failures",
                "value": str(len(contract_summary.hard_failures)),
                "detail": "Blocking issues that stop execution.",
                "tone": "blocked" if contract_summary.hard_failures else "ready",
            },
            {
                "label": "Warnings",
                "value": str(len(contract_summary.warnings)),
                "detail": "Non-blocking monitoring caveats.",
                "tone": "warning" if contract_summary.warnings else "ready",
            },
        ]
    )

    _render_status_chip_row(
        [
            (contract_summary.overall_status.replace("_", " ").title(), contract_summary.overall_status),
            (f"Hard Failures {len(contract_summary.hard_failures)}", "fail"),
            (f"Warnings {len(contract_summary.warnings)}", "warning"),
        ]
    )

    if contract_summary.hard_failures:
        st.error("\n".join(contract_summary.hard_failures))
    elif contract_summary.warnings:
        st.warning("\n".join(contract_summary.warnings))
    else:
        st.success("The dataset satisfies the saved bundle contract and is ready to run.")

    if contract_summary.score_only_run and not contract_summary.hard_failures:
        st.info(
            "This dataset will run in score-only mode. Realized-performance tests will be marked N/A."
        )

    with st.expander("Validation Summary", expanded=False):
        st.dataframe(contract_summary.summary_frame, hide_index=True, use_container_width=True)

    review_tabs = st.tabs(["Issues", "Dataset Preview", "Date Coverage", "Guardrails"])

    with review_tabs[0]:
        findings_frame = contract_summary.findings_frame
        if not findings_frame.empty:
            st.dataframe(findings_frame, hide_index=True, use_container_width=True)

        _render_status_chip_row(_status_chip_pairs_from_series(contract_summary.column_checks["status"]))
        search_text = st.text_input(
            "Find Column",
            key=f"contract_column_search_{bundle.bundle_id}_{dataset.dataset_id}",
            help="Search contract checks by column name, role, or detail.",
        )
        show_only_problems = st.toggle(
            "Show Only Problems",
            key=f"contract_problem_filter_{bundle.bundle_id}_{dataset.dataset_id}",
            help="Focus on warnings and hard failures in the saved bundle contract.",
        )
        column_checks = _sort_column_checks_frame(contract_summary.column_checks.copy())
        column_checks = _apply_column_search(
            column_checks,
            search_text,
            ["column_name", "configured_name", "role", "detail"],
        )
        if show_only_problems:
            column_checks = column_checks.loc[column_checks["status"] != "pass"]
        if column_checks.empty:
            st.caption(
                "No contract rows matched the current search or problem filter."
                if search_text.strip() or show_only_problems
                else "No contract issues are active for the selected dataset."
            )
        else:
            st.dataframe(column_checks, hide_index=True, use_container_width=True)

    with review_tabs[1]:
        dataset_frame = _load_dataset_frame(str(dataset.path))
        search_text = st.text_input(
            "Find Preview Column",
            key=f"preview_column_search_{bundle.bundle_id}_{dataset.dataset_id}",
            help="Search the dataset preview by column name, role, or match type.",
        )
        show_only_problems = st.toggle(
            "Show Only Problems",
            key=f"preview_problem_filter_{bundle.bundle_id}_{dataset.dataset_id}",
            help="Focus on missing required columns, warnings, and extras.",
        )
        preview_frame = _sort_dataset_preview_frame(
            _build_dataset_column_preview_frame(
                bundle,
                contract_summary.contract.detected_columns,
            )
        )
        preview_frame = _apply_column_search(
            preview_frame,
            search_text,
            ["dataset_column", "role", "detail", "contract_match"],
        )
        if show_only_problems:
            preview_frame = preview_frame.loc[preview_frame["status"] != "pass"]
        if preview_frame.empty:
            st.caption(
                "No dataset-preview rows matched the current search or problem filter."
                if search_text.strip() or show_only_problems
                else "No dataset-preview issues are active for the selected dataset."
            )
        else:
            st.dataframe(preview_frame, hide_index=True, use_container_width=True)

        preview_tabs = st.tabs(["Sample Rows", "Inferred Dtypes"])
        with preview_tabs[0]:
            st.dataframe(dataset_frame.head(12), hide_index=True, use_container_width=True)
        with preview_tabs[1]:
            st.dataframe(
                _build_inferred_dtype_frame(dataset_frame),
                hide_index=True,
                use_container_width=True,
            )

    with review_tabs[2]:
        if contract_summary.date_coverage.empty:
            st.caption("The selected bundle does not define a date column.")
        else:
            st.dataframe(contract_summary.date_coverage, hide_index=True, use_container_width=True)
        with st.expander("Dataset Fingerprint", expanded=False):
            st.dataframe(
                build_dataset_fingerprint_frame(dataset),
                hide_index=True,
                use_container_width=True,
            )

    with review_tabs[3]:
        st.caption(
            "Dataset guardrails protect the local app from unexpectedly large monitoring files."
        )
        st.dataframe(contract_summary.guardrails, hide_index=True, use_container_width=True)


def _render_run_tab(
    *,
    workspace,
    bundle,
    dataset,
    datasets,
    contract_summary,
    selected_segment: str | None,
    disable_individual_visual_exports: bool,
    artifact_profile: str,
    toolbar_run_requested: bool,
) -> None:
    _render_stage_intro(
        number=4,
        title="Execute The Current Monitoring Run And Review Outputs",
        description=(
            "Run the selected bundle against the selected dataset, then review the outcome card, "
            "test results, and downloadable reviewer artifacts."
        ),
    )

    thresholds = load_threshold_records(workspace.thresholds_root, bundle)
    st.caption("Runs use the currently saved threshold set for the selected model bundle.")
    if disable_individual_visual_exports:
        st.info(
            "Individual PNG and per-figure HTML exports are disabled from the left pane for faster reruns. "
            "The stitched monitoring report and workbook remain enabled."
        )
    if artifact_profile != "full":
        st.info(
            f"Artifact profile `{artifact_profile}` is active. HTML and Excel outputs remain enabled."
        )

    if not bundle.is_compliant:
        st.error("This bundle is not runnable for raw-data monitoring reruns.")
    elif bundle.warnings:
        st.warning("This bundle is runnable, but intake warnings should be reviewed before use.")

    if dataset is None:
        _render_empty_state(
            title="Monitoring Run Is Waiting On A Dataset",
            message="Select a watched dataset before executing the monitoring workflow.",
            bullets=[
                f"Watched folder: {workspace.incoming_data_root}",
                "Accepted formats: .csv, .xlsx, .xls",
                "The dataset should contain the raw input fields required by the selected bundle.",
            ],
        )
    elif contract_summary is not None and contract_summary.hard_failures:
        st.error(
            "The selected dataset does not satisfy the pre-run contract. Fix the hard failures "
            "before executing monitoring."
        )
    elif contract_summary is not None and contract_summary.score_only_run:
        st.info(
            "This run will proceed in score-only mode because labels are unavailable in the dataset."
        )

    outcome_dataset_path, outcome_join_columns = _render_outcome_join_controls(
        bundle=bundle,
        dataset=dataset,
        datasets=datasets,
    )
    reviewer_notes, reviewer_exceptions = _render_reviewer_notes_editor(
        thresholds,
        key_prefix=f"reviewer_notes_{bundle.bundle_id}_{dataset.dataset_id if dataset else 'none'}",
    )

    run_disabled = (
        not bundle.is_compliant
        or dataset is None
        or contract_summary is None
        or bool(contract_summary.hard_failures)
    )
    run_tab_requested = st.button("Run Monitoring", type="primary", disabled=run_disabled)
    if toolbar_run_requested or run_tab_requested:
        assert dataset is not None
        with st.spinner("Running scoring bundle and monitoring diagnostics..."):
            run_result = execute_monitoring_run(
                bundle=bundle,
                dataset=dataset,
                workspace=workspace,
                thresholds=thresholds,
                segment_column=selected_segment,
                scoring_options=ScoringRuntimeOptions(
                    disable_individual_visual_exports=disable_individual_visual_exports,
                    outcome_dataset_path=outcome_dataset_path,
                    outcome_join_columns=outcome_join_columns,
                    artifact_profile=artifact_profile,
                ),
                reviewer_notes=reviewer_notes,
                reviewer_exceptions=reviewer_exceptions,
            )
        st.session_state["latest_run"] = run_result
        st.session_state["run_history"].insert(0, run_result)

    selected_run = _selected_run(bundle, dataset)
    if selected_run is not None:
        _render_run_result(selected_run)
    elif isinstance(st.session_state.get("latest_run"), MonitoringRunResult):
        st.caption(
            "The latest session run belongs to a different bundle or dataset selection. Choose that selection to review it here."
        )

    _render_test_help_drawer(thresholds, key_prefix=f"run_test_help_{bundle.bundle_id}")


def _render_outcome_join_controls(*, bundle, dataset, datasets) -> tuple[Path | None, list[str]]:
    if dataset is None or not datasets:
        return None, []

    with st.expander("Optional Outcome File Join", expanded=False):
        st.caption(
            "Use this only when labels live in a separate watched CSV or Excel file. "
            "OM Studio joins the label into the scoring input before validation and execution."
        )
        available_outcomes = [
            dataset_item for dataset_item in datasets if dataset_item.dataset_id != dataset.dataset_id
        ]
        if not available_outcomes:
            st.caption("No separate watched dataset is available for outcome joining.")
            return None, []

        outcome_lookup = {"No separate outcome file": None}
        outcome_lookup.update(
            {
                f"{dataset_item.name} | {dataset_item.modified_at:%Y-%m-%d %H:%M}": dataset_item
                for dataset_item in available_outcomes
            }
        )
        selected_label = st.selectbox(
            "Outcome File",
            list(outcome_lookup.keys()),
            key=f"outcome_file_{bundle.bundle_id}_{dataset.dataset_id}",
        )
        selected_outcome = outcome_lookup[selected_label]
        if selected_outcome is None:
            return None, []

        input_columns = read_dataset_columns(dataset.path)
        outcome_columns = read_dataset_columns(selected_outcome.path)
        common_columns = sorted(set(input_columns).intersection(outcome_columns))
        preferred_join_columns = [
            column
            for column in bundle.identifier_columns + bundle.date_columns
            if column in common_columns
        ]
        join_columns = st.multiselect(
            "Join Columns",
            options=common_columns,
            default=preferred_join_columns[:1] or common_columns[:1],
            key=f"outcome_join_columns_{bundle.bundle_id}_{dataset.dataset_id}",
            help="Use stable identifiers where possible. Date columns can be added if needed.",
        )
        label_candidates = [
            column
            for column in (bundle.label_source_column, bundle.label_output_column)
            if column and column in outcome_columns
        ]
        _render_status_chip_row(
            [
                (f"Join Keys {len(join_columns)}", "ready" if join_columns else "blocked"),
                (
                    f"Outcome Labels {len(label_candidates)}",
                    "ready" if label_candidates else "blocked",
                ),
            ]
        )
        if not label_candidates:
            st.warning(
                "The selected outcome file does not contain the bundle label column. "
                "The run can continue, but labels will not be joined."
            )
        return selected_outcome.path, join_columns


def _render_reviewer_notes_editor(records, *, key_prefix: str) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    with st.expander("Reviewer Exceptions And Notes", expanded=False):
        st.caption(
            "Optional. Dispositions and notes are exported with this run only and do not create approval workflow state."
        )
        notes_frame = pd.DataFrame(
            [
                {
                    "test_id": record.test_id,
                    "test": record.label,
                    "category": record.category,
                    "exception_disposition": "None",
                    "exception_rationale": "",
                    "reviewer_note": "",
                }
                for record in records
            ]
        )
        edited_notes = st.data_editor(
            notes_frame.drop(columns=["test_id"]),
            hide_index=True,
            use_container_width=True,
            key=f"{key_prefix}_editor",
            column_config={
                "test": st.column_config.TextColumn("Test", disabled=True, width="medium"),
                "category": st.column_config.TextColumn("Category", disabled=True),
                "exception_disposition": st.column_config.SelectboxColumn(
                    "Exception",
                    options=[
                        "None",
                        "Exception Noted",
                        "Remediation Planned",
                        "Accepted Limitation",
                    ],
                ),
                "exception_rationale": st.column_config.TextColumn(
                    "Exception Rationale",
                    width="large",
                ),
                "reviewer_note": st.column_config.TextColumn("Reviewer Note", width="large"),
            },
        )
        edited_notes["test_id"] = notes_frame["test_id"]
        notes = {
            str(row["test_id"]): str(row["reviewer_note"]).strip()
            for row in edited_notes.to_dict(orient="records")
            if str(row.get("reviewer_note", "")).strip()
        }
        exceptions = {
            str(row["test_id"]): {
                "disposition": str(row["exception_disposition"]).strip(),
                "rationale": str(row.get("exception_rationale", "")).strip(),
            }
            for row in edited_notes.to_dict(orient="records")
            if str(row.get("exception_disposition", "None")).strip() != "None"
            or str(row.get("exception_rationale", "")).strip()
        }
        return notes, exceptions


def _render_session_history(workspace) -> None:
    _render_stage_intro(
        number=5,
        title="Review Current-Session Monitoring History",
        description=(
            "Use this area to review completed runs from the current app session and compare "
            "two runs side by side."
        ),
    )
    persisted_history = load_run_index(workspace)
    if not st.session_state["run_history"]:
        st.caption("No monitoring runs have been executed in this session yet.")
        if not persisted_history.empty:
            st.markdown("### Persisted Run Manifest")
            st.dataframe(persisted_history, hide_index=True, use_container_width=True)
        return

    _render_summary_tiles(
        [
            {
                "label": "Session Runs",
                "value": str(len(st.session_state["run_history"])),
                "detail": "Monitoring runs captured in this session.",
                "tone": "neutral",
            },
            {
                "label": "Latest Status",
                "value": st.session_state["run_history"][0].status.replace("_", " ").title(),
                "detail": st.session_state["run_history"][0].dataset.name,
                "tone": st.session_state["run_history"][0].status,
            },
            {
                "label": "Comparisons",
                "value": "Enabled" if len(st.session_state["run_history"]) >= 2 else "Waiting",
                "detail": (
                    "Select two runs to compare."
                    if len(st.session_state["run_history"]) >= 2
                    else "Run at least two monitoring jobs."
                ),
                "tone": "ready" if len(st.session_state["run_history"]) >= 2 else "warning",
            },
        ]
    )

    history = pd.DataFrame(
        [
            {
                "run_id": run.run_id,
                "model_name": run.model_bundle.display_name,
                "model_version": run.model_bundle.model_version,
                "dataset_name": run.dataset.name,
                "status": run.status,
                "passed": run.pass_count,
                "failed": run.fail_count,
                "na": run.na_count,
                "run_root": str(run.run_root),
            }
            for run in st.session_state["run_history"]
        ]
    )
    st.dataframe(history, hide_index=True, use_container_width=True)
    if not persisted_history.empty:
        with st.expander("Persisted Run Manifest", expanded=False):
            st.dataframe(persisted_history, hide_index=True, use_container_width=True)
    if len(st.session_state["run_history"]) >= 2:
        _render_run_comparison(st.session_state["run_history"])


def _render_run_comparison(run_history: list[MonitoringRunResult]) -> None:
    st.markdown("### Compare Two Session Runs")
    comparison_lookup = {
        f"{run.run_id} | {run.model_bundle.display_name} | {run.dataset.name}": run
        for run in run_history
    }
    labels = list(comparison_lookup.keys())
    selectors = st.columns(2)
    with selectors[0]:
        left_label = st.selectbox("Baseline Run", labels, key="compare_left_run")
    with selectors[1]:
        default_index = 1 if len(labels) > 1 else 0
        right_label = st.selectbox(
            "Comparison Run",
            labels,
            index=default_index,
            key="compare_right_run",
        )

    left_run = comparison_lookup[left_label]
    right_run = comparison_lookup[right_label]
    if left_run.run_id == right_run.run_id:
        st.caption("Choose two different runs to compare.")
        return

    summary = build_run_comparison_summary_frame(left_run, right_run)
    st.dataframe(summary, hide_index=True, use_container_width=True)

    changed_tests = build_run_comparison_frame(left_run, right_run)
    if changed_tests.empty:
        st.caption("No test-level differences were found between the selected runs.")
    else:
        st.dataframe(changed_tests, hide_index=True, use_container_width=True)
    export_columns = st.columns(2)
    with export_columns[0]:
        st.download_button(
            "Download Comparison Workbook",
            data=build_run_comparison_workbook_bytes(left_run, right_run),
            file_name=f"run_comparison_{left_run.run_id}_vs_{right_run.run_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with export_columns[1]:
        st.download_button(
            "Download Comparison HTML",
            data=build_run_comparison_html(left_run, right_run).encode("utf-8"),
            file_name=f"run_comparison_{left_run.run_id}_vs_{right_run.run_id}.html",
            mime="text/html",
            use_container_width=True,
        )


def _render_run_result(run_result: MonitoringRunResult) -> None:
    st.markdown("### Latest Run")
    st.markdown(_build_run_outcome_card_html(run_result), unsafe_allow_html=True)

    if run_result.status == "completed" and not run_result.labels_available:
        st.info("This completed as a score-only run. Label-based tests were marked N/A.")
    if run_result.error_message:
        st.error(run_result.error_message)
        diagnostics = run_result.support_tables.get("failure_diagnostics", pd.DataFrame())
        if not diagnostics.empty:
            st.dataframe(diagnostics, hide_index=True, use_container_width=True)
        with st.expander("Execution Debug Trace", expanded=False):
            for artifact_key in ("run_debug_trace", "step_manifest", "run_events"):
                artifact_path = run_result.artifacts.get(artifact_key)
                if artifact_path and Path(artifact_path).exists():
                    st.markdown(f"**{artifact_key.replace('_', ' ').title()}**")
                    if Path(artifact_path).suffix == ".jsonl":
                        st.code(Path(artifact_path).read_text(encoding="utf-8")[:6000])
                    else:
                        st.json(json.loads(Path(artifact_path).read_text(encoding="utf-8")))

    report_path = run_result.artifacts.get("report")
    workbook_path = run_result.artifacts.get("workbook")
    package_path = run_result.artifacts.get("reviewer_package")
    diagnostic_path = run_result.artifacts.get("diagnostic_export")
    downloads = st.columns(4)
    with downloads[0]:
        if report_path and Path(report_path).exists():
            st.download_button(
                "Download HTML Report",
                data=Path(report_path).read_bytes(),
                file_name=Path(report_path).name,
                mime="text/html",
                use_container_width=True,
            )
    with downloads[1]:
        if workbook_path and Path(workbook_path).exists():
            st.download_button(
                "Download Workbook",
                data=Path(workbook_path).read_bytes(),
                file_name=Path(workbook_path).name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    with downloads[2]:
        if package_path and Path(package_path).exists():
            st.download_button(
                "Download Reviewer Package",
                data=Path(package_path).read_bytes(),
                file_name=Path(package_path).name,
                mime="application/zip",
                use_container_width=True,
            )
    with downloads[3]:
        if diagnostic_path and Path(diagnostic_path).exists():
            st.download_button(
                "Download Diagnostic Export",
                data=Path(diagnostic_path).read_bytes(),
                file_name=Path(diagnostic_path).name,
                mime="application/zip",
                use_container_width=True,
            )

    exception_summary = run_result.support_tables.get("reviewer_exception_summary", pd.DataFrame())
    if not exception_summary.empty and run_result.reviewer_exceptions:
        st.markdown("### Reviewer Exceptions")
        st.dataframe(exception_summary, hide_index=True, use_container_width=True)

    completeness = run_result.support_tables.get("artifact_completeness", pd.DataFrame())
    if not completeness.empty:
        with st.expander("Artifact Completeness", expanded=False):
            st.dataframe(completeness, hide_index=True, use_container_width=True)

    applicability = run_result.support_tables.get("test_applicability_matrix", pd.DataFrame())
    if not applicability.empty:
        with st.expander("Test Applicability Matrix", expanded=False):
            st.dataframe(applicability, hide_index=True, use_container_width=True)

    delta_summary = run_result.support_tables.get("reference_monitoring_delta_summary", pd.DataFrame())
    if not delta_summary.empty:
        st.markdown("### Reference Vs Monitoring Delta")
        st.dataframe(delta_summary, hide_index=True, use_container_width=True)

    diagnostic_tabs = {
        "Data Quality": run_result.support_tables.get("data_quality_summary", pd.DataFrame()),
        "Feature Drift": run_result.support_tables.get("feature_drift_summary", pd.DataFrame()),
        "Outcome Join": run_result.support_tables.get("outcome_join_summary", pd.DataFrame()),
    }
    visible_diagnostics = {name: frame for name, frame in diagnostic_tabs.items() if not frame.empty}
    if visible_diagnostics:
        st.markdown("### Run Diagnostics")
        tabs = st.tabs(list(visible_diagnostics.keys()))
        for tab, frame in zip(tabs, visible_diagnostics.values(), strict=False):
            with tab:
                st.dataframe(frame, hide_index=True, use_container_width=True)

    if not run_result.results_frame.empty:
        st.markdown("### Test Results")
        _render_status_chip_row(_status_chip_pairs_from_series(run_result.results_frame["status"]))
        show_only_non_pass = st.toggle(
            "Show Only Non-Pass Tests",
            key=f"run_problem_filter_{run_result.run_id}",
            help="Focus on failed and N/A monitoring tests first.",
        )
        results_frame = run_result.results_frame.copy()
        if run_result.reviewer_notes:
            results_frame["reviewer_note"] = results_frame["test_id"].map(
                run_result.reviewer_notes
            ).fillna("")
        if run_result.reviewer_exceptions:
            results_frame["reviewer_exception"] = results_frame["test_id"].map(
                lambda test_id: run_result.reviewer_exceptions.get(str(test_id), {}).get(
                    "disposition",
                    "",
                )
            )
        if show_only_non_pass:
            results_frame = results_frame.loc[results_frame["status"] != "pass"]
        if results_frame.empty:
            st.caption("No non-pass tests are active for this monitoring run.")
        else:
            st.dataframe(
                results_frame,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "detail": st.column_config.TextColumn("Detail", width="large"),
                    "observed_value": st.column_config.NumberColumn("Observed"),
                    "threshold_value": st.column_config.NumberColumn("Threshold"),
                },
            )

    with st.expander("Artifact Paths", expanded=False):
        for label, path in run_result.artifacts.items():
            if path is None:
                continue
            st.markdown(f"- `{label}`: `{path}`")


def _build_run_outcome_card_html(run_result: MonitoringRunResult) -> str:
    chips = [
        (run_result.status.replace("_", " ").title(), run_result.status),
        ("Labels Available" if run_result.labels_available else "Score-Only", "ready" if run_result.labels_available else "score_only"),
    ]
    if run_result.score_column:
        chips.append((f"Score Column {run_result.score_column}", "na"))
    if run_result.segment_column:
        chips.append((f"Segment {run_result.segment_column}", "na"))

    return f"""
    <section class="outcome-card">
      <div class="outcome-kicker">Run Outcome</div>
      <h3>{escape(run_result.model_bundle.display_name)} on {escape(run_result.dataset.name)}</h3>
      <p>
        Run ID {escape(run_result.run_id)} started {escape(_format_run_timestamp(run_result.started_at))}
        and wrote artifacts under {escape(str(run_result.run_root))}.
      </p>
      <div class="status-chip-row">
        {''.join(_build_status_chip_html(label, tone) for label, tone in chips)}
      </div>
      <div class="outcome-grid">
        <div class="outcome-metric">
          <strong>{escape(run_result.model_bundle.model_version)}</strong>
          <span>model version</span>
        </div>
        <div class="outcome-metric">
          <strong>{run_result.pass_count}</strong>
          <span>tests passed</span>
        </div>
        <div class="outcome-metric">
          <strong>{run_result.fail_count}</strong>
          <span>tests failed</span>
        </div>
        <div class="outcome-metric">
          <strong>{run_result.na_count}</strong>
          <span>tests N/A</span>
        </div>
      </div>
    </section>
    """


def _format_run_timestamp(value) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S %Z")


if __name__ == "__main__":
    main()
