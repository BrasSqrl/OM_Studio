# OM Studio

OM Studio is a separate application for narrow, ongoing monitoring of approved quantitative models.

It does not train models. It discovers approved model bundles, discovers monitoring datasets, reruns the selected approved model on raw uploaded data, applies monitoring tests, and exports reviewer-ready monitoring artifacts.

## MVP Scope

The implemented scope is intentionally narrow:
- discover compliant approved-model bundles from `models/`
- discover monitoring datasets from `incoming_data/`
- show model metadata in the UI
- persist per-model monitoring thresholds in `thresholds/`
- rerun approved bundles on raw monitoring data
- compute per-test pass/fail outcomes
- export HTML and Excel monitoring artifacts under `runs/`
- show current-session run results in the UI

Out of scope:
- model training
- challenger development workflow
- feature subset search
- scheduling
- external databases
- authentication
- persistent run-history registry beyond the current session

## Compliant Bundle Contract

For MVP, a supported model must be a compliant rerunnable bundle.

Minimum required files in a bundle directory:
- `quant_model.joblib`
- `run_config.json`
- `generated_run.py`

Reference files used for richer monitoring diagnostics:
- `input_snapshot.csv`
- `predictions.csv`
- `artifact_manifest.json`

Optional metadata override:
- `monitoring_metadata.json`

## Workflow

1. Drop approved model bundles into `models/`.
2. Drop monitoring datasets into `incoming_data/`.
3. Launch the app.
4. Select a model bundle and dataset.
5. Review or save per-model thresholds.
6. Run monitoring.
7. Review per-test pass/fail results and exported artifacts.

If labels are present in the monitoring dataset, realized-performance tests run.
If labels are absent, label-dependent tests are marked `N/A`.

## Demo Path

OM Studio includes a built-in demo path for smoke testing.

You can use either of these approaches:
- click `Generate Demo Assets` in the app sidebar
- call `quant_studio_monitoring.create_demo_assets(...)` in code

The demo path creates:
- a compliant demo bundle in `models/`
- a matching monitoring dataset in `incoming_data/`

This lets a fresh workspace exercise the full narrow monitoring flow without any external bundle or dataset.

## Watched Directories

The app uses these local directories:
- `models/`
- `incoming_data/`
- `thresholds/`
- `runs/`

These directories are created automatically on launch.

## Monitoring Tests

Always available when reference artifacts exist:
- required-column schema validation
- row-count review
- missingness review
- score PSI
- score drift p-value
- segment PSI when a segment column is available

Available for binary labeled monitoring data:
- AUC
- KS
- Gini
- Brier score
- MAE
- Hosmer-Lemeshow p-value
- precision at threshold
- recall at threshold
- bad-rate-by-band MAE

Available for continuous labeled monitoring data:
- MAE

## Project Layout

- `app/streamlit_app.py`: Streamlit monitoring UI
- `src/quant_studio_monitoring/config.py`: workspace directory configuration
- `src/quant_studio_monitoring/registry.py`: bundle and dataset discovery
- `src/quant_studio_monitoring/thresholds.py`: per-model threshold persistence
- `src/quant_studio_monitoring/monitoring_pipeline.py`: scoring orchestration and monitoring tests
- `src/quant_studio_monitoring/monitoring_reporting.py`: HTML/Excel artifact generation
- `src/quant_studio_monitoring/presentation.py`: shared UI/theme helpers
- `docs/ROADMAP.md`: strict MVP roadmap

## Launch

First-time setup:

```powershell
.\setup_gui.bat
```

Launch:

```powershell
.\launch_gui.bat
```
