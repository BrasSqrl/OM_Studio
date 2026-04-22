"""Built-in demo asset generation for OM Studio smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import WorkspaceConfig

DEMO_BUNDLE_ID = "om_demo_bundle_v1"
DEMO_DATASET_NAME = "om_demo_monitoring.csv"
DEMO_EXPORT_VERSION = "om-demo-1.0"


def create_demo_assets(
    workspace: WorkspaceConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    workspace.ensure_directories()
    bundle_root = workspace.models_root / DEMO_BUNDLE_ID
    dataset_path = workspace.incoming_data_root / DEMO_DATASET_NAME

    if overwrite and bundle_root.exists():
        for path in sorted(bundle_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    bundle_root.mkdir(parents=True, exist_ok=True)

    reference_raw = _build_demo_reference_raw()
    reference_predictions = _build_demo_predictions(reference_raw)
    monitoring_raw = _build_demo_monitoring_raw()

    _write_if_missing_or_overwrite(
        bundle_root / "run_config.json",
        json.dumps(_demo_run_config(), indent=2),
        overwrite=overwrite,
    )
    _write_if_missing_or_overwrite(
        bundle_root / "monitoring_metadata.json",
        json.dumps(_demo_monitoring_metadata(), indent=2),
        overwrite=overwrite,
    )
    _write_if_missing_or_overwrite(
        bundle_root / "artifact_manifest.json",
        json.dumps(_demo_manifest(bundle_root), indent=2),
        overwrite=overwrite,
    )
    _write_if_missing_or_overwrite(
        bundle_root / "generated_run.py",
        _demo_generated_runner_script(),
        overwrite=overwrite,
    )
    _write_if_missing_or_overwrite(
        bundle_root / "HOW_TO_RERUN.md",
        "Run the bundle through OM Studio or execute generated_run.py with --config, --input, and --output-root.\n",
        overwrite=overwrite,
    )
    _write_if_missing_or_overwrite(
        bundle_root / "quant_model.joblib",
        "",
        overwrite=overwrite,
        binary=True,
    )

    code_snapshot = bundle_root / "code_snapshot"
    code_snapshot.mkdir(parents=True, exist_ok=True)
    _write_if_missing_or_overwrite(
        code_snapshot / "README.txt",
        "Demo code snapshot for OM Studio smoke tests.\n",
        overwrite=overwrite,
    )

    _write_dataframe(bundle_root / "input_snapshot.csv", reference_raw, overwrite=overwrite)
    _write_dataframe(bundle_root / "predictions.csv", reference_predictions, overwrite=overwrite)
    _write_dataframe(dataset_path, monitoring_raw, overwrite=overwrite)

    return {"bundle_root": bundle_root, "dataset_path": dataset_path}


def _demo_run_config() -> dict[str, object]:
    return {
        "export_version": DEMO_EXPORT_VERSION,
        "schema": {
            "column_specs": [
                {"name": "as_of_date", "source_name": "as_of_date", "enabled": True, "role": "date"},
                {"name": "loan_id", "source_name": "loan_id", "enabled": True, "role": "identifier"},
                {"name": "annual_income", "source_name": "annual_income", "enabled": True, "role": "feature", "dtype": "float"},
                {"name": "debt_to_income", "source_name": "debt_to_income", "enabled": True, "role": "feature", "dtype": "float"},
                {"name": "utilization", "source_name": "utilization", "enabled": True, "role": "feature", "dtype": "float"},
                {"name": "delinquency_count", "source_name": "delinquency_count", "enabled": True, "role": "feature", "dtype": "int"},
                {"name": "region", "source_name": "region", "enabled": True, "role": "feature", "dtype": "string"},
                {"name": "employment_type", "source_name": "employment_type", "enabled": True, "role": "feature", "dtype": "string"},
                {"name": "default_status", "source_name": "default_status", "enabled": True, "role": "target_source", "dtype": "int"},
            ]
        },
        "documentation": {
            "model_name": "OM Demo Retail PD",
            "model_owner": "Model Risk",
            "business_purpose": "Built-in ongoing monitoring smoke test",
            "portfolio_name": "Demo Retail",
        },
        "model": {"model_type": "demo_logistic_scorecard", "threshold": 0.45},
        "target": {"mode": "binary", "output_column": "default_flag"},
        "diagnostics": {"default_segment_column": "region"},
        "execution": {"mode": "fit_new_model"},
    }


def _demo_monitoring_metadata() -> dict[str, str]:
    return {
        "model_name": "OM Demo Retail PD",
        "model_version": "2026.04.demo",
        "model_owner": "Model Risk",
        "business_purpose": "Built-in ongoing monitoring smoke test",
        "portfolio_name": "Demo Retail",
        "segment_name": "Region",
        "default_segment_column": "region",
        "approval_status": "approved",
        "approval_date": "2026-04-21",
        "monitoring_notes": "Generated by the OM Studio demo asset path.",
    }


def _demo_manifest(bundle_root: Path) -> dict[str, object]:
    return {
        "export_version": DEMO_EXPORT_VERSION,
        "bundle_id": DEMO_BUNDLE_ID,
        "bundle_root": str(bundle_root),
        "artifacts": [
            "quant_model.joblib",
            "run_config.json",
            "generated_run.py",
            "input_snapshot.csv",
            "predictions.csv",
            "monitoring_metadata.json",
        ],
    }


def _build_demo_reference_raw() -> pd.DataFrame:
    rows = []
    regions = ["north", "south", "east", "west"]
    employment = ["salaried", "hourly", "contract"]
    for index in range(1, 25):
        rows.append(
            {
                "as_of_date": f"2025-12-{(index % 28) + 1:02d}",
                "loan_id": f"REF_{index:04d}",
                "annual_income": 52000 + index * 1800,
                "debt_to_income": round(0.18 + (index % 7) * 0.045, 3),
                "utilization": round(0.22 + (index % 6) * 0.08, 3),
                "delinquency_count": index % 3,
                "region": regions[index % len(regions)],
                "employment_type": employment[index % len(employment)],
                "default_status": 1 if index % 6 in {0, 1} else 0,
            }
        )
    return pd.DataFrame(rows)


def _build_demo_monitoring_raw() -> pd.DataFrame:
    rows = []
    regions = ["north", "south", "east", "west"]
    employment = ["salaried", "hourly", "contract"]
    for index in range(1, 31):
        rows.append(
            {
                "as_of_date": f"2026-03-{(index % 28) + 1:02d}",
                "loan_id": f"MON_{index:04d}",
                "annual_income": 50500 + index * 1650,
                "debt_to_income": round(0.21 + (index % 8) * 0.043, 3),
                "utilization": round(0.28 + (index % 5) * 0.085, 3),
                "delinquency_count": (index + 1) % 4,
                "region": regions[(index + 1) % len(regions)],
                "employment_type": employment[(index + 1) % len(employment)],
                "default_status": 1 if index % 5 in {0, 1} else 0,
            }
        )
    return pd.DataFrame(rows)


def _build_demo_predictions(raw: pd.DataFrame) -> pd.DataFrame:
    score = _score_probability(raw)
    return pd.DataFrame(
        {
            "loan_id": raw["loan_id"],
            "predicted_probability_recommended": score,
            "default_flag": raw["default_status"],
        }
    )


def _score_probability(raw: pd.DataFrame) -> pd.Series:
    region_effect = raw["region"].map({"north": -0.08, "south": 0.06, "east": 0.04, "west": 0.0}).fillna(0.0)
    employment_effect = raw["employment_type"].map({"salaried": -0.05, "hourly": 0.03, "contract": 0.07}).fillna(0.0)
    linear = (
        -2.35
        + (pd.to_numeric(raw["debt_to_income"], errors="coerce").fillna(0.0) * 2.1)
        + (pd.to_numeric(raw["utilization"], errors="coerce").fillna(0.0) * 1.55)
        + (pd.to_numeric(raw["delinquency_count"], errors="coerce").fillna(0.0) * 0.42)
        - (pd.to_numeric(raw["annual_income"], errors="coerce").fillna(0.0) / 100000.0 * 0.65)
        + region_effect
        + employment_effect
    )
    probability = 1.0 / (1.0 + np.exp(-linear))
    return probability.clip(0.01, 0.99).round(6)


def _demo_generated_runner_script() -> str:
    return """
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _score_probability(raw: pd.DataFrame) -> pd.Series:
    region_effect = raw["region"].map({"north": -0.08, "south": 0.06, "east": 0.04, "west": 0.0}).fillna(0.0)
    employment_effect = raw["employment_type"].map({"salaried": -0.05, "hourly": 0.03, "contract": 0.07}).fillna(0.0)
    linear = (
        -2.35
        + (pd.to_numeric(raw["debt_to_income"], errors="coerce").fillna(0.0) * 2.1)
        + (pd.to_numeric(raw["utilization"], errors="coerce").fillna(0.0) * 1.55)
        + (pd.to_numeric(raw["delinquency_count"], errors="coerce").fillna(0.0) * 0.42)
        - (pd.to_numeric(raw["annual_income"], errors="coerce").fillna(0.0) / 100000.0 * 0.65)
        + region_effect
        + employment_effect
    )
    probability = 1.0 / (1.0 + np.exp(-linear))
    return probability.clip(0.01, 0.99).round(6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    raw = pd.read_csv(args.input)
    output_root = Path(args.output_root) / "demo_run"
    output_root.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame(
        {
            "loan_id": raw["loan_id"],
            "predicted_probability_recommended": _score_probability(raw),
            "default_flag": pd.to_numeric(raw.get("default_status"), errors="coerce"),
        }
    )
    predictions.to_csv(output_root / "predictions.csv", index=False)
    (output_root / "metrics.json").write_text(json.dumps({"row_count": len(predictions)}, indent=2), encoding="utf-8")
    (output_root / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "export_version": config.get("export_version", "om-demo-1.0"),
                "artifacts": ["predictions.csv", "metrics.json"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Artifacts written to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""".strip() + "\n"


def _write_if_missing_or_overwrite(
    path: Path,
    payload: str,
    *,
    overwrite: bool,
    binary: bool = False,
) -> None:
    if path.exists() and not overwrite:
        return
    if binary:
        path.write_bytes(payload.encode("utf-8"))
        return
    path.write_text(payload, encoding="utf-8")


def _write_dataframe(path: Path, frame: pd.DataFrame, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    frame.to_csv(path, index=False)
