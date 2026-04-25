from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from quant_studio_monitoring.ui_regression import (
    UI_REGRESSION_SCENARIOS,
    write_ui_regression_checklist,
)


def test_ui_regression_checklist_has_core_workflow_states() -> None:
    scenario_ids = {scenario.scenario_id for scenario in UI_REGRESSION_SCENARIOS}

    assert {
        "empty_workspace",
        "bundle_selected_no_dataset",
        "validation_ready",
        "completed_run",
        "execution_failure",
    }.issubset(scenario_ids)


def test_write_ui_regression_checklist() -> None:
    output_path = Path.cwd() / ".test_workspace" / f"ui_{uuid4().hex}" / "checklist.json"

    written_path = write_ui_regression_checklist(output_path)
    payload = json.loads(written_path.read_text(encoding="utf-8"))

    assert written_path == output_path
    assert len(payload["scenarios"]) == len(UI_REGRESSION_SCENARIOS)
