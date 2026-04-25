# GUI-To-Code Traceability Guide

OM Studio keeps the UI narrow. Each control either selects an exported model
bundle, selects monitoring data, edits thresholds, runs monitoring, or opens
review artifacts.

| UI Surface | User Choice | Backend Target | Export Evidence |
| --- | --- | --- | --- |
| Monitoring Inbox | Refresh registry, generate demo assets | `discover_model_bundles`, `discover_datasets`, `create_demo_assets` | bundle and dataset fingerprints |
| Model Bundle | Select discovered bundle | `ModelBundle`, `validate_monitoring_bundle_contract` | `bundle_metadata.json`, `bundle_fingerprint.csv` |
| Dataset | Select watched CSV/Excel file | `DatasetAsset`, `summarize_dataset_contract` | `dataset_provenance.json`, `dataset_fingerprint.csv` |
| Validate | Review required columns and preview | `validate_input_contract` | `input_contract.json` |
| Thresholds | Edit, save, import, export thresholds | `ThresholdRecord`, threshold workbook helpers | `threshold_snapshot.json`, `test_enablement_profile.csv` |
| Run | Execute scoring bundle and tests | `execute_monitoring_run` | `monitoring_run_config.json`, `monitoring_tests.json` |
| Runtime Controls | Artifact profile and figure-file toggle | `ScoringRuntimeOptions` | `artifacts_manifest.json`, `monitoring_run_config.json` |
| History | Review current-session and persisted runs | `run_history.py` | `runs/run_index.csv` |

Traceability chain:

`Streamlit control -> typed runtime/config object -> monitoring pipeline -> artifact manifest -> reviewer report`

The UI does not train, retrain, approve, or schedule models. It only consumes
compliant exported bundles and produces monitoring evidence.
