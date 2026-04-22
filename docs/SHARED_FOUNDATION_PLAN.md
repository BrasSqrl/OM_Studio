# Shared Foundation Plan

## Objective

Keep the monitoring application separate while preserving the same product feel and reporting quality as Quant Studio.

## Share First

The highest-value reusable pieces are:
- Streamlit CSS theme and hero layout
- Plotly figure theme
- KPI card patterns
- report page shell
- artifact bundle naming patterns
- high-level config serialization patterns
- audit-oriented documentation style

## Do Not Share Blindly

Do not automatically port these areas:
- development orchestration
- training logic
- challenger-development logic
- subset-search logic
- development-only documentation packs

## Shared Package Candidate Modules

A future shared internal package could eventually contain:
- `ui_theme.py`
- `figure_theme.py`
- `report_shell.py`
- `artifact_helpers.py`
- `config_io_helpers.py`
- `display_table_helpers.py`

## Monitoring-Specific Package Modules

This app should own:
- `registry.py`
- `monitoring_pipeline.py`
- `drift.py`
- `performance_monitoring.py`
- `monitoring_reporting.py`
- `alerts.py`

## Rule

If a utility exists mainly to preserve look, feel, layout, or export style, it is a good candidate for sharing.
If it exists mainly to implement development or monitoring business logic, it should stay in the application that owns that workflow.
