"""Application-level workspace configuration for monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ALLOWED_DATASET_SUFFIXES = {".csv", ".xlsx", ".xls"}


@dataclass(slots=True)
class WorkspaceConfig:
    """Local directories the monitoring app watches and writes to."""

    project_root: Path
    models_root: Path
    incoming_data_root: Path
    thresholds_root: Path
    runs_root: Path

    def ensure_directories(self) -> None:
        for path in (
            self.models_root,
            self.incoming_data_root,
            self.thresholds_root,
            self.runs_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_workspace_config(project_root: Path | None = None) -> WorkspaceConfig:
    resolved_root = project_root or get_project_root()
    config = WorkspaceConfig(
        project_root=resolved_root,
        models_root=resolved_root / "models",
        incoming_data_root=resolved_root / "incoming_data",
        thresholds_root=resolved_root / "thresholds",
        runs_root=resolved_root / "runs",
    )
    config.ensure_directories()
    return config
