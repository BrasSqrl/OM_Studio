"""UI regression checklist for the narrow OM Studio workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class UiRegressionScenario:
    scenario_id: str
    title: str
    expected_state: str
    checklist: tuple[str, ...]


UI_REGRESSION_SCENARIOS: tuple[UiRegressionScenario, ...] = (
    UiRegressionScenario(
        scenario_id="empty_workspace",
        title="Empty Workspace",
        expected_state="No compliant bundle guidance is visible.",
        checklist=(
            "Sidebar shows watched directory context.",
            "Main panel explains minimum bundle files.",
            "No run button is enabled.",
        ),
    ),
    UiRegressionScenario(
        scenario_id="bundle_selected_no_dataset",
        title="Bundle Selected Without Dataset",
        expected_state="Bundle context is visible and execution waits for data.",
        checklist=(
            "Overview shows model metadata.",
            "Decision card says dataset is needed.",
            "Validate and Run tabs show dataset guidance.",
        ),
    ),
    UiRegressionScenario(
        scenario_id="validation_ready",
        title="Runnable Dataset Selected",
        expected_state="Validation shows ready or score-only state.",
        checklist=(
            "Toolbar selectors are readable.",
            "Validate tab highlights hard failures and warnings.",
            "Thresholds tab shows enabled tests and saved state.",
        ),
    ),
    UiRegressionScenario(
        scenario_id="completed_run",
        title="Completed Run",
        expected_state="Run outcome card and artifact downloads are visible.",
        checklist=(
            "Run tab shows pass/fail/N/A counts.",
            "HTML report, workbook, and reviewer package downloads are available.",
            "Data-quality and feature-drift diagnostics are readable when present.",
        ),
    ),
    UiRegressionScenario(
        scenario_id="execution_failure",
        title="Execution Failure",
        expected_state="Failure stage and diagnostics are visible.",
        checklist=(
            "Run status is not hidden by the results table.",
            "Failure diagnostics table appears.",
            "Artifacts still include failure diagnostics for reviewer inspection.",
        ),
    ),
)


def write_ui_regression_checklist(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"scenarios": [asdict(scenario) for scenario in UI_REGRESSION_SCENARIOS]}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path
