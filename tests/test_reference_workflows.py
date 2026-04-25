from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from examples.reference_workflows.om_monitoring_reference import run_reference_workflows


def test_reference_monitoring_workflows_match_expected_contracts() -> None:
    workspace_root = Path.cwd() / ".test_workspace" / f"reference_{uuid4().hex}"
    actual = run_reference_workflows(workspace_root)
    expected = json.loads(
        Path("examples/reference_workflows/expected/om_monitoring_reference.json").read_text(
            encoding="utf-8"
        )
    )

    for scenario_id, expected_fields in expected.items():
        assert scenario_id in actual
        for field_name, expected_value in expected_fields.items():
            assert actual[scenario_id][field_name] == expected_value
        assert actual[scenario_id]["test_count"] > 0
