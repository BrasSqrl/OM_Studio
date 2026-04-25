# Sister Repo Alignment Roadmap

This roadmap keeps OM Studio aligned with the Quant Studio sister project while
preserving the narrow monitoring scope: consume exported model bundles, validate
monitoring data, run tests, and produce reviewer artifacts.

## Implementation Items

1. Support the sister-project `model_bundle_for_monitoring/` directory as the preferred intake shape.
2. Add a formal monitoring bundle contract validator.
3. Save a canonical `monitoring_run_config.json` for every run.
4. Export `monitoring_step_manifest.json`, `run_debug_trace.json`, and `run_events.jsonl`.
5. Strengthen `artifacts_manifest.json` with generated and skipped artifact indexes.
6. Split the Streamlit UI into a thin `app/streamlit_app.py` entrypoint and package-owned controller/state modules.
7. Move per-test reviewer guidance into a reusable test catalog.
8. Add threshold workbook export/import.
9. Add monitoring performance caps for report/table/figure previews.
10. Add failed-stage debug trace visibility in the UI.
11. Persist dataset provenance with file hash, shape, column list, and schema hash.
12. Add deterministic reference monitoring workflows and expected outputs.
13. Add CI for Ruff, pytest, reference workflows, and a profile smoke test.
14. Align Ruff lint selection with the sister repo.
15. Document GUI-to-code traceability for reviewer auditability.

## Status

All items above are implemented in the current codebase. Future changes should
extend this roadmap only if they keep the product focused on ongoing monitoring
of existing exported models.
