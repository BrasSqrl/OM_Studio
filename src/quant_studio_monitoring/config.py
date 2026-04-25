"""Application-level workspace configuration for monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_DATASET_SUFFIXES = {".csv", ".xlsx", ".xls"}


@dataclass(slots=True)
class MonitoringPerformanceConfig:
    """Runtime caps that keep monitoring runs and reports responsive."""

    ui_preview_rows: int = 50
    html_table_preview_rows: int = 100
    html_max_figures: int = 4
    support_csv_row_cap: int = 100_000
    feature_drift_feature_cap: int = 200
    large_dataset_warning_rows: int = 100_000
    large_dataset_block_rows: int = 1_000_000
    large_dataset_warning_columns: int = 250
    large_dataset_block_columns: int = 1_000

    def validate(self) -> None:
        for field_name, value in {
            "ui_preview_rows": self.ui_preview_rows,
            "html_table_preview_rows": self.html_table_preview_rows,
            "html_max_figures": self.html_max_figures,
            "support_csv_row_cap": self.support_csv_row_cap,
            "feature_drift_feature_cap": self.feature_drift_feature_cap,
            "large_dataset_warning_rows": self.large_dataset_warning_rows,
            "large_dataset_block_rows": self.large_dataset_block_rows,
            "large_dataset_warning_columns": self.large_dataset_warning_columns,
            "large_dataset_block_columns": self.large_dataset_block_columns,
        }.items():
            if value <= 0:
                raise ValueError(f"MonitoringPerformanceConfig.{field_name} must be positive.")
        if self.large_dataset_warning_rows > self.large_dataset_block_rows:
            raise ValueError(
                "MonitoringPerformanceConfig.large_dataset_warning_rows cannot exceed "
                "large_dataset_block_rows."
            )
        if self.large_dataset_warning_columns > self.large_dataset_block_columns:
            raise ValueError(
                "MonitoringPerformanceConfig.large_dataset_warning_columns cannot exceed "
                "large_dataset_block_columns."
            )


@dataclass(slots=True)
class WorkspaceConfig:
    """Local directories the monitoring app watches and writes to."""

    project_root: Path
    models_root: Path
    incoming_data_root: Path
    thresholds_root: Path
    runs_root: Path
    performance: MonitoringPerformanceConfig = field(default_factory=MonitoringPerformanceConfig)

    def ensure_directories(self) -> None:
        self.performance.validate()
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
        performance=MonitoringPerformanceConfig(),
    )
    config.ensure_directories()
    return config
