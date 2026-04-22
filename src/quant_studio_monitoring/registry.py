"""Discovery helpers for model bundles and monitoring datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from .config import ALLOWED_DATASET_SUFFIXES, WorkspaceConfig

GENERIC_MODEL_NAME = "Quant Studio Model"
MODEL_ARTIFACT_SUFFIXES = {".joblib", ".pkl", ".pickle", ".onnx"}
SUPPORTED_TARGET_MODES = {"binary", "continuous", "regression"}
MONITORING_METADATA_FIELDS = (
    "model_name",
    "model_version",
    "model_owner",
    "business_purpose",
    "portfolio_name",
    "segment_name",
    "default_segment_column",
    "approval_status",
    "approval_date",
    "monitoring_notes",
)
READINESS_ORDER = {"ready": 0, "ready_with_warnings": 1, "not_ready": 2}
REVIEW_REQUIRED_METADATA_FIELDS = {
    "model_name": "Model Name",
    "model_version": "Model Version",
    "model_owner": "Model Owner",
    "business_purpose": "Business Purpose",
    "approval_status": "Approval Status",
    "approval_date": "Approval Date",
}


@dataclass(slots=True)
class BundlePaths:
    root: Path
    model_path: Path
    run_config_path: Path
    generated_runner_path: Path | None
    manifest_path: Path | None
    reference_input_path: Path | None
    reference_predictions_path: Path | None
    monitoring_metadata_path: Path | None
    code_snapshot_path: Path | None


@dataclass(slots=True)
class BundleReadinessCheck:
    code: str
    item: str
    severity: str
    detail: str


@dataclass(slots=True)
class BundleColumnSpec:
    name: str
    source_name: str
    role: str
    dtype: str
    enabled: bool
    create_if_missing: bool
    required_for_run: bool


@dataclass(slots=True)
class MonitoringMetadata:
    model_name: str
    model_version: str
    model_owner: str
    business_purpose: str
    portfolio_name: str
    segment_name: str
    default_segment_column: str
    approval_status: str
    approval_date: str
    monitoring_notes: str

    def to_payload(self) -> dict[str, str]:
        return {field_name: getattr(self, field_name) for field_name in MONITORING_METADATA_FIELDS}


@dataclass(slots=True)
class ModelBundle:
    bundle_id: str
    display_name: str
    model_version: str
    model_owner: str
    business_purpose: str
    portfolio_name: str
    segment_name: str
    model_type: str
    target_mode: str
    model_threshold: float | None
    expected_input_columns: list[str]
    optional_input_columns: list[str]
    feature_columns: list[str]
    date_columns: list[str]
    identifier_columns: list[str]
    label_source_column: str | None
    label_output_column: str | None
    default_segment_column: str | None
    reference_row_count: int | None
    reference_score_column: str | None
    bundle_paths: BundlePaths
    monitoring_metadata: MonitoringMetadata
    column_specs: list[BundleColumnSpec]
    compatibility_checks: list[BundleReadinessCheck] = field(default_factory=list)
    export_version: str | None = None
    review_metadata_gaps: list[str] = field(default_factory=list)
    metadata_source: str = "inferred"
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    readiness_checks: list[BundleReadinessCheck] = field(default_factory=list)

    @property
    def readiness_status(self) -> str:
        if self.issues:
            return "not_ready"
        if self.warnings:
            return "ready_with_warnings"
        return "ready"

    @property
    def readiness_label(self) -> str:
        return self.readiness_status.replace("_", " ").title()

    @property
    def is_compliant(self) -> bool:
        return not self.issues

    @property
    def compatibility_status(self) -> str:
        severities = {check.severity for check in self.compatibility_checks}
        if "error" in severities:
            return "not_compatible"
        if "warning" in severities:
            return "compatible_with_warnings"
        return "compatible"

    @property
    def compatibility_label(self) -> str:
        return self.compatibility_status.replace("_", " ").title()

    @property
    def is_compatible(self) -> bool:
        return self.compatibility_status != "not_compatible"

    @property
    def review_status(self) -> str:
        return "review_complete" if not self.review_metadata_gaps else "review_incomplete"

    @property
    def review_label(self) -> str:
        return self.review_status.replace("_", " ").title()

    @property
    def is_review_complete(self) -> bool:
        return not self.review_metadata_gaps

    @property
    def all_input_columns(self) -> list[str]:
        return _unique(self.expected_input_columns + self.optional_input_columns)


@dataclass(slots=True)
class DatasetAsset:
    dataset_id: str
    name: str
    path: Path
    suffix: str
    modified_at: datetime


@dataclass(slots=True)
class InputContractResult:
    required_input_columns: list[str]
    missing_required_columns: list[str]
    optional_input_columns: list[str]
    missing_optional_columns: list[str]
    detected_columns: list[str]
    resolved_label_column: str | None
    labels_available: bool
    requested_segment_column: str | None
    segment_available: bool


@dataclass(slots=True)
class DatasetContractSummary:
    contract: InputContractResult
    row_count: int
    column_count: int
    overall_status: str
    score_only_run: bool
    hard_failures: list[str]
    warnings: list[str]
    column_checks: pd.DataFrame
    date_coverage: pd.DataFrame

    @property
    def summary_frame(self) -> pd.DataFrame:
        date_min = None
        date_max = None
        if not self.date_coverage.empty:
            date_min = self.date_coverage["min_date"].dropna().astype(str).min()
            date_max = self.date_coverage["max_date"].dropna().astype(str).max()
        execution_mode = "blocked"
        if self.overall_status != "not_ready":
            execution_mode = "score_only" if self.score_only_run else "full_monitoring"
        return pd.DataFrame(
            [
                {"metric": "overall_status", "value": self.overall_status},
                {"metric": "execution_mode", "value": execution_mode},
                {"metric": "row_count", "value": self.row_count},
                {"metric": "column_count", "value": self.column_count},
                {
                    "metric": "required_missing_count",
                    "value": len(self.contract.missing_required_columns),
                },
                {
                    "metric": "optional_missing_count",
                    "value": len(self.contract.missing_optional_columns),
                },
                {"metric": "labels_available", "value": self.contract.labels_available},
                {"metric": "segment_available", "value": self.contract.segment_available},
                {"metric": "date_min", "value": date_min or "n/a"},
                {"metric": "date_max", "value": date_max or "n/a"},
            ]
        )

    @property
    def findings_frame(self) -> pd.DataFrame:
        findings: list[dict[str, str]] = []
        for detail in self.hard_failures:
            findings.append({"severity": "error", "detail": detail})
        for detail in self.warnings:
            findings.append({"severity": "warning", "detail": detail})
        return pd.DataFrame(findings)


def discover_model_bundles(workspace: WorkspaceConfig) -> list[ModelBundle]:
    bundles: list[ModelBundle] = []
    seen_roots: set[Path] = set()
    if not workspace.models_root.exists():
        return bundles

    for config_path in workspace.models_root.rglob("run_config.json"):
        root = config_path.parent
        if root in seen_roots or "code_snapshot" in root.parts:
            continue
        seen_roots.add(root)
        bundles.append(_build_model_bundle(root))

    for artifact_path in workspace.models_root.rglob("*"):
        if not artifact_path.is_file() or artifact_path.suffix.lower() not in MODEL_ARTIFACT_SUFFIXES:
            continue
        if "code_snapshot" in artifact_path.parts:
            continue
        if artifact_path.parent in seen_roots:
            continue
        bundles.append(_build_standalone_model_bundle(workspace, artifact_path))

    return sorted(
        bundles,
        key=lambda bundle: (
            READINESS_ORDER[bundle.readiness_status],
            bundle.display_name.lower(),
            bundle.model_version.lower(),
        ),
    )


def discover_datasets(workspace: WorkspaceConfig) -> list[DatasetAsset]:
    datasets: list[DatasetAsset] = []
    if not workspace.incoming_data_root.exists():
        return datasets

    for path in workspace.incoming_data_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_DATASET_SUFFIXES:
            continue
        stat = path.stat()
        datasets.append(
            DatasetAsset(
                dataset_id=slugify(path.relative_to(workspace.incoming_data_root).as_posix()),
                name=path.name,
                path=path,
                suffix=path.suffix.lower(),
                modified_at=datetime.fromtimestamp(stat.st_mtime),
            )
        )

    return sorted(datasets, key=lambda dataset: dataset.modified_at, reverse=True)


def read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported dataset suffix '{path.suffix}' for {path}.")


def read_dataset_columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=0).columns.astype(str).tolist()
    return []


def validate_input_contract(
    bundle: ModelBundle,
    dataframe: pd.DataFrame,
    *,
    segment_column: str | None = None,
) -> InputContractResult:
    detected_columns = dataframe.columns.astype(str).tolist()
    missing_required_columns = [
        column for column in bundle.expected_input_columns if column not in detected_columns
    ]
    missing_optional_columns = [
        column for column in bundle.optional_input_columns if column not in detected_columns
    ]
    resolved_label_column = None
    for candidate in (bundle.label_source_column, bundle.label_output_column):
        if candidate and candidate in detected_columns:
            resolved_label_column = candidate
            break

    return InputContractResult(
        required_input_columns=bundle.expected_input_columns,
        missing_required_columns=missing_required_columns,
        optional_input_columns=bundle.optional_input_columns,
        missing_optional_columns=missing_optional_columns,
        detected_columns=detected_columns,
        resolved_label_column=resolved_label_column,
        labels_available=resolved_label_column is not None,
        requested_segment_column=segment_column,
        segment_available=bool(segment_column and segment_column in detected_columns),
    )


def summarize_dataset_contract(
    bundle: ModelBundle,
    dataframe: pd.DataFrame,
    *,
    segment_column: str | None = None,
) -> DatasetContractSummary:
    contract = validate_input_contract(bundle, dataframe, segment_column=segment_column)
    hard_failures: list[str] = []
    warnings: list[str] = []

    if dataframe.empty:
        hard_failures.append("The monitoring dataset contains zero rows.")
    if contract.missing_required_columns:
        hard_failures.append(
            "Missing required columns: " + ", ".join(contract.missing_required_columns)
        )

    missing_optional_non_label = [
        column
        for column in contract.missing_optional_columns
        if column and column != bundle.label_source_column
    ]
    if missing_optional_non_label:
        warnings.append(
            "Optional schema columns will be auto-filled for scoring: "
            + ", ".join(missing_optional_non_label)
        )
    if bundle.label_source_column and bundle.label_source_column in contract.missing_optional_columns:
        warnings.append(
            f"Label column '{bundle.label_source_column}' is not present. "
            "The run can continue in score-only mode and realized-performance tests will be N/A."
        )
    if segment_column and not contract.segment_available:
        warnings.append(
            f"Selected segment column '{segment_column}' is not present. Segment PSI will be skipped."
        )

    column_checks = _build_column_checks(bundle=bundle, dataframe=dataframe, contract=contract)
    date_coverage = _build_date_coverage(bundle=bundle, dataframe=dataframe)
    dtype_warnings = column_checks.loc[column_checks["status"] == "warning", "detail"].astype(str)
    warnings.extend(detail for detail in dtype_warnings if detail not in warnings)

    overall_status = "not_ready" if hard_failures else ("ready_with_warnings" if warnings else "ready")
    return DatasetContractSummary(
        contract=contract,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        overall_status=overall_status,
        score_only_run=not contract.labels_available,
        hard_failures=hard_failures,
        warnings=warnings,
        column_checks=column_checks,
        date_coverage=date_coverage,
    )


def load_monitoring_metadata(bundle: ModelBundle) -> MonitoringMetadata:
    return bundle.monitoring_metadata


def save_monitoring_metadata(bundle: ModelBundle, metadata: MonitoringMetadata) -> Path:
    target_path = bundle.bundle_paths.root / "monitoring_metadata.json"
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata.to_payload(), handle, indent=2)
    return target_path


def build_input_template_workbook_bytes(
    bundle: ModelBundle,
    *,
    include_example_row: bool = True,
) -> bytes:
    input_columns = [spec.source_name for spec in bundle.column_specs if not spec.create_if_missing]
    if include_example_row:
        template_frame = pd.DataFrame(
            [{column: _template_example_value(bundle, column) for column in input_columns}]
        )
    else:
        template_frame = pd.DataFrame(columns=input_columns)

    guide_frame = pd.DataFrame(
        [
            {
                "column_name": spec.name,
                "configured_name": spec.name,
                "source_name": spec.source_name,
                "role": spec.role,
                "expected_dtype": spec.dtype or "n/a",
                "required_for_run": spec.required_for_run,
                "notes": _column_note(bundle, spec),
            }
            for spec in bundle.column_specs
            if not spec.create_if_missing
        ]
    )
    notes_frame = pd.DataFrame(
        [
            {"item": "model_name", "value": bundle.display_name},
            {"item": "model_version", "value": bundle.model_version},
            {
                "item": "usage",
                "value": (
                    "Populate one row per observation. Required columns must be present. "
                    "Optional schema columns may be blank. If the label column is blank, the run "
                    "will continue in score-only mode."
                ),
            },
        ]
    )

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        template_frame.to_excel(writer, sheet_name="input_template", index=False)
        guide_frame.to_excel(writer, sheet_name="column_guide", index=False)
        notes_frame.to_excel(writer, sheet_name="usage_notes", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def build_bundle_fingerprint_frame(bundle: ModelBundle) -> pd.DataFrame:
    rows = [
        _path_fingerprint_row("model_artifact", bundle.bundle_paths.model_path),
        _path_fingerprint_row("run_config", bundle.bundle_paths.run_config_path),
        _path_fingerprint_row("generated_runner", bundle.bundle_paths.generated_runner_path),
        _path_fingerprint_row("artifact_manifest", bundle.bundle_paths.manifest_path),
        _path_fingerprint_row(
            "monitoring_metadata",
            bundle.bundle_paths.root / "monitoring_metadata.json",
        ),
        _path_fingerprint_row("reference_input", bundle.bundle_paths.reference_input_path),
        _path_fingerprint_row(
            "reference_predictions",
            bundle.bundle_paths.reference_predictions_path,
        ),
    ]
    if bundle.bundle_paths.code_snapshot_path is not None:
        rows.append(_directory_row("code_snapshot", bundle.bundle_paths.code_snapshot_path))
    return pd.DataFrame(rows)


def build_bundle_compatibility_frame(bundle: ModelBundle) -> pd.DataFrame:
    if not bundle.compatibility_checks:
        return pd.DataFrame(columns=["severity", "item", "detail"])
    return pd.DataFrame(
        [
            {
                "severity": check.severity,
                "item": check.item,
                "detail": check.detail,
            }
            for check in bundle.compatibility_checks
        ]
    )


def build_review_completeness_frame(bundle: ModelBundle) -> pd.DataFrame:
    rows = []
    gaps = set(bundle.review_metadata_gaps)
    for field_name, label in REVIEW_REQUIRED_METADATA_FIELDS.items():
        value = getattr(bundle.monitoring_metadata, field_name, "")
        rows.append(
            {
                "field": label,
                "value": value or "n/a",
                "status": "missing" if label in gaps else "present",
            }
        )
    return pd.DataFrame(rows)


def build_dataset_fingerprint_frame(dataset: DatasetAsset) -> pd.DataFrame:
    return pd.DataFrame([_path_fingerprint_row("monitoring_dataset", dataset.path)])


def resolve_prediction_column_name(columns: list[str], target_mode: str) -> str | None:
    if target_mode == "binary":
        preferred = [
            "predicted_probability_recommended",
            "predicted_probability",
            "predicted_probability_platt",
            "predicted_probability_isotonic",
        ]
        for candidate in preferred:
            if candidate in columns:
                return candidate
        fallback = [column for column in columns if column.startswith("predicted_probability")]
        return fallback[0] if fallback else None

    for candidate in ("predicted_value", "predicted_score"):
        if candidate in columns:
            return candidate
    fallback = [column for column in columns if column.startswith("predicted_")]
    return fallback[0] if fallback else None


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return normalized or "item"


def _build_model_bundle(root: Path) -> ModelBundle:
    model_path = root / "quant_model.joblib"
    run_config_path = root / "run_config.json"
    generated_runner_path = root / "generated_run.py"
    manifest_path = root / "artifact_manifest.json"
    reference_input_path = root / "input_snapshot.csv"
    reference_predictions_path = root / "predictions.csv"
    monitoring_metadata_path = root / "monitoring_metadata.json"
    code_snapshot_path = root / "code_snapshot"

    paths = BundlePaths(
        root=root,
        model_path=model_path,
        run_config_path=run_config_path,
        generated_runner_path=generated_runner_path if generated_runner_path.exists() else None,
        manifest_path=manifest_path if manifest_path.exists() else None,
        reference_input_path=reference_input_path if reference_input_path.exists() else None,
        reference_predictions_path=(
            reference_predictions_path if reference_predictions_path.exists() else None
        ),
        monitoring_metadata_path=(
            monitoring_metadata_path if monitoring_metadata_path.exists() else None
        ),
        code_snapshot_path=code_snapshot_path if code_snapshot_path.exists() else None,
    )

    issues: list[str] = []
    warnings: list[str] = []
    checks: list[BundleReadinessCheck] = []

    if not model_path.exists():
        _append_check(
            checks,
            severity="error",
            code="missing_model_artifact",
            item="quant_model.joblib",
            detail="The saved model artifact is missing. The bundle cannot score raw data.",
        )
        issues.append("Missing quant_model.joblib.")
    if not run_config_path.exists():
        _append_check(
            checks,
            severity="error",
            code="missing_run_config",
            item="run_config.json",
            detail="The saved run configuration is missing. The bundle contract cannot be reconstructed.",
        )
        issues.append("Missing run_config.json.")
    if paths.generated_runner_path is None:
        _append_check(
            checks,
            severity="error",
            code="missing_generated_runner",
            item="generated_run.py",
            detail="The generated rerun launcher is missing. Raw-data reruns are blocked.",
        )
        issues.append("Missing generated_run.py required for raw-data reruns.")

    config_payload, config_error = _read_json_safely(run_config_path)
    if config_error:
        _append_check(
            checks,
            severity="error",
            code="invalid_run_config",
            item="run_config.json",
            detail=f"The run configuration could not be parsed: {config_error}",
        )
        issues.append(f"run_config.json could not be parsed: {config_error}")
        config_payload = {}

    metadata_payload, metadata_error = _read_json_safely(monitoring_metadata_path)
    if metadata_error:
        _append_check(
            checks,
            severity="warning",
            code="invalid_monitoring_metadata",
            item="monitoring_metadata.json",
            detail=f"The monitoring metadata file could not be parsed: {metadata_error}",
        )
        warnings.append(f"monitoring_metadata.json could not be parsed: {metadata_error}")
        metadata_payload = {}
    elif not monitoring_metadata_path.exists():
        _append_check(
            checks,
            severity="warning",
            code="missing_monitoring_metadata",
            item="monitoring_metadata.json",
            detail="UI metadata is being inferred from run_config.json because no monitoring manifest was found.",
        )
        warnings.append("Missing monitoring_metadata.json. UI metadata is being inferred.")

    manifest_payload: dict[str, Any] = {}
    if paths.manifest_path is not None:
        manifest_payload, manifest_error = _read_json_safely(manifest_path)
        if manifest_error:
            _append_check(
                checks,
                severity="warning",
                code="invalid_artifact_manifest",
                item="artifact_manifest.json",
                detail=f"The artifact manifest could not be parsed: {manifest_error}",
            )
            warnings.append(f"artifact_manifest.json could not be parsed: {manifest_error}")
            manifest_payload = {}

    if paths.manifest_path is None:
        _append_check(
            checks,
            severity="warning",
            code="missing_artifact_manifest",
            item="artifact_manifest.json",
            detail="The bundle inventory cannot be verified because artifact_manifest.json is missing.",
        )
        warnings.append(
            "Missing artifact_manifest.json. Bundle inventory cannot be verified."
        )
    if paths.reference_input_path is None:
        _append_check(
            checks,
            severity="warning",
            code="missing_reference_input",
            item="input_snapshot.csv",
            detail="Reference row counts and missingness comparisons will be limited.",
        )
        warnings.append(
            "Missing input_snapshot.csv. Reference row-count and missingness comparisons will be limited."
        )
    if paths.reference_predictions_path is None:
        _append_check(
            checks,
            severity="warning",
            code="missing_reference_predictions",
            item="predictions.csv",
            detail="Reference score drift comparisons may be disabled.",
        )
        warnings.append(
            "Missing predictions.csv. Reference score drift comparisons may be disabled."
        )
    if paths.code_snapshot_path is None:
        _append_check(
            checks,
            severity="warning",
            code="missing_code_snapshot",
            item="code_snapshot",
            detail="Portable reruns may depend on the locally installed sister-project package because code_snapshot is missing.",
        )
        warnings.append(
            "Missing code_snapshot/. Portable reruns may depend on the local sister-project install."
        )

    documentation = config_payload.get("documentation", {})
    schema = config_payload.get("schema", {})
    model_config = config_payload.get("model", {})
    target_config = config_payload.get("target", {})
    diagnostics_config = config_payload.get("diagnostics", {})
    column_specs = _build_column_specs(schema.get("column_specs", []))
    if not column_specs and config_payload:
        _append_check(
            checks,
            severity="warning",
            code="missing_column_specs",
            item="schema.column_specs",
            detail="No enabled schema columns were found. Pre-run validation and template generation will be limited.",
        )
        warnings.append(
            "No enabled schema columns were found in run_config.json. Pre-run validation is limited."
        )

    expected_input_columns = [
        spec.source_name
        for spec in column_specs
        if spec.required_for_run and spec.role != "target_source"
    ]
    optional_input_columns = [
        spec.source_name
        for spec in column_specs
        if not spec.required_for_run and not spec.create_if_missing
    ]
    feature_columns = [spec.source_name for spec in column_specs if spec.role == "feature"]
    date_columns = [spec.source_name for spec in column_specs if spec.role == "date"]
    identifier_columns = [spec.source_name for spec in column_specs if spec.role == "identifier"]
    label_source_column = next(
        (spec.source_name for spec in column_specs if spec.role == "target_source"),
        None,
    )
    reference_row_count = _count_csv_rows(paths.reference_input_path)
    reference_columns = (
        read_dataset_columns(paths.reference_predictions_path)
        if paths.reference_predictions_path is not None
        else []
    )
    reference_score_column = resolve_prediction_column_name(
        reference_columns,
        str(target_config.get("mode", "")),
    )
    if reference_score_column is None and paths.reference_predictions_path is not None:
        _append_check(
            checks,
            severity="warning",
            code="missing_reference_score_column",
            item="predictions.csv",
            detail="A supported reference prediction column was not found. Score drift tests will be disabled.",
        )
        warnings.append(
            "No supported reference score column was found in predictions.csv. Drift tests will be limited."
        )

    monitoring_metadata = _build_monitoring_metadata(
        root=root,
        metadata_payload=metadata_payload,
        documentation=documentation,
        diagnostics_config=diagnostics_config,
    )
    metadata_source = "manifest" if metadata_payload else "inferred"
    export_version = _resolve_export_version(manifest_payload, config_payload)
    compatibility_checks = _build_bundle_compatibility_checks(
        bundle_root=root,
        paths=paths,
        target_mode=str(target_config.get("mode", "")),
        export_version=export_version,
        reference_score_column=reference_score_column,
        column_specs=column_specs,
    )
    review_metadata_gaps = _build_review_metadata_gaps(monitoring_metadata)

    return ModelBundle(
        bundle_id=slugify(root.name),
        display_name=monitoring_metadata.model_name,
        model_version=monitoring_metadata.model_version or root.name,
        model_owner=monitoring_metadata.model_owner,
        business_purpose=monitoring_metadata.business_purpose,
        portfolio_name=monitoring_metadata.portfolio_name,
        segment_name=monitoring_metadata.segment_name,
        model_type=str(model_config.get("model_type", "")),
        target_mode=str(target_config.get("mode", "")),
        model_threshold=_safe_float(model_config.get("threshold")),
        expected_input_columns=_unique(expected_input_columns),
        optional_input_columns=_unique(optional_input_columns),
        feature_columns=_unique(feature_columns),
        date_columns=_unique(date_columns),
        identifier_columns=_unique(identifier_columns),
        label_source_column=label_source_column,
        label_output_column=target_config.get("output_column"),
        default_segment_column=(
            monitoring_metadata.default_segment_column
            or diagnostics_config.get("default_segment_column")
        ),
        reference_row_count=reference_row_count,
        reference_score_column=reference_score_column,
        bundle_paths=paths,
        monitoring_metadata=monitoring_metadata,
        column_specs=column_specs,
        compatibility_checks=compatibility_checks,
        export_version=export_version,
        review_metadata_gaps=review_metadata_gaps,
        metadata_source=metadata_source,
        issues=issues,
        warnings=warnings,
        readiness_checks=checks,
    )


def _build_standalone_model_bundle(
    workspace: WorkspaceConfig,
    artifact_path: Path,
) -> ModelBundle:
    relative_path = artifact_path.relative_to(workspace.models_root)
    inferred_name = (
        artifact_path.stem
        if artifact_path.parent == workspace.models_root
        else artifact_path.parent.name
    )
    issue_detail = (
        "Standalone model artifact detected. Raw-data monitoring requires the full exported "
        "bundle directory, not only the model file."
    )
    checks = [
        BundleReadinessCheck(
            code="standalone_model_artifact",
            item=artifact_path.name,
            severity="error",
            detail=issue_detail,
        ),
        BundleReadinessCheck(
            code="missing_run_config",
            item="run_config.json",
            severity="error",
            detail="The saved run configuration is missing. The bundle contract cannot be reconstructed.",
        ),
        BundleReadinessCheck(
            code="missing_generated_runner",
            item="generated_run.py",
            severity="error",
            detail="The generated rerun launcher is missing. Raw-data reruns are blocked.",
        ),
    ]
    issues = [
        issue_detail,
        "Missing run_config.json.",
        "Missing generated_run.py required for raw-data reruns.",
    ]
    metadata = MonitoringMetadata(
        model_name=inferred_name,
        model_version="unknown",
        model_owner="",
        business_purpose="",
        portfolio_name="",
        segment_name="",
        default_segment_column="",
        approval_status="",
        approval_date="",
        monitoring_notes="",
    )

    compatibility_checks = [
        BundleReadinessCheck(
            code="standalone_bundle_shape",
            item=artifact_path.name,
            severity="error",
            detail="Standalone model files are not compatible. OM Studio expects the full exported bundle directory.",
        ),
        BundleReadinessCheck(
            code="unknown_export_version",
            item="export_version",
            severity="warning",
            detail="No export version could be determined for this asset.",
        ),
    ]
    review_metadata_gaps = list(REVIEW_REQUIRED_METADATA_FIELDS.values())

    return ModelBundle(
        bundle_id=slugify(relative_path.with_suffix("").as_posix()),
        display_name=inferred_name,
        model_version="unknown",
        model_owner="",
        business_purpose="",
        portfolio_name="",
        segment_name="",
        model_type=artifact_path.suffix.lower().lstrip("."),
        target_mode="",
        model_threshold=None,
        expected_input_columns=[],
        optional_input_columns=[],
        feature_columns=[],
        date_columns=[],
        identifier_columns=[],
        label_source_column=None,
        label_output_column=None,
        default_segment_column=None,
        reference_row_count=None,
        reference_score_column=None,
        bundle_paths=BundlePaths(
            root=artifact_path.parent,
            model_path=artifact_path,
            run_config_path=artifact_path.parent / "run_config.json",
            generated_runner_path=None,
            manifest_path=None,
            reference_input_path=None,
            reference_predictions_path=None,
            monitoring_metadata_path=artifact_path.parent / "monitoring_metadata.json"
            if (artifact_path.parent / "monitoring_metadata.json").exists()
            else None,
            code_snapshot_path=None,
        ),
        monitoring_metadata=metadata,
        column_specs=[],
        compatibility_checks=compatibility_checks,
        export_version=None,
        review_metadata_gaps=review_metadata_gaps,
        issues=issues,
        warnings=[],
        readiness_checks=checks,
    )


def _build_column_specs(payload: list[dict[str, Any]]) -> list[BundleColumnSpec]:
    specs: list[BundleColumnSpec] = []
    for item in payload:
        if not item.get("enabled", True):
            continue
        source_name = str(item.get("source_name") or item.get("name") or "").strip()
        name = str(item.get("name") or source_name).strip()
        if not source_name or not name:
            continue
        role = str(item.get("role", "feature"))
        create_if_missing = bool(item.get("create_if_missing", False))
        specs.append(
            BundleColumnSpec(
                name=name,
                source_name=source_name,
                role=role,
                dtype=str(item.get("dtype", "")),
                enabled=True,
                create_if_missing=create_if_missing,
                required_for_run=(role in {"feature", "date", "identifier"} and not create_if_missing),
            )
        )
    return specs


def _build_monitoring_metadata(
    *,
    root: Path,
    metadata_payload: dict[str, Any],
    documentation: dict[str, Any],
    diagnostics_config: dict[str, Any],
) -> MonitoringMetadata:
    model_name = str(
        metadata_payload.get("model_name")
        or documentation.get("model_name")
        or root.name
    ).strip()
    if model_name == GENERIC_MODEL_NAME:
        model_name = root.name

    return MonitoringMetadata(
        model_name=model_name,
        model_version=str(
            metadata_payload.get("model_version")
            or documentation.get("model_version")
            or root.name
        ).strip(),
        model_owner=str(
            metadata_payload.get("model_owner") or documentation.get("model_owner") or ""
        ).strip(),
        business_purpose=str(
            metadata_payload.get("business_purpose")
            or documentation.get("business_purpose")
            or ""
        ).strip(),
        portfolio_name=str(
            metadata_payload.get("portfolio_name")
            or documentation.get("portfolio_name")
            or ""
        ).strip(),
        segment_name=str(
            metadata_payload.get("segment_name") or documentation.get("segment_name") or ""
        ).strip(),
        default_segment_column=str(
            metadata_payload.get("default_segment_column")
            or diagnostics_config.get("default_segment_column")
            or ""
        ).strip(),
        approval_status=str(metadata_payload.get("approval_status") or "").strip(),
        approval_date=str(metadata_payload.get("approval_date") or "").strip(),
        monitoring_notes=str(metadata_payload.get("monitoring_notes") or "").strip(),
    )


def _build_bundle_compatibility_checks(
    *,
    bundle_root: Path,
    paths: BundlePaths,
    target_mode: str,
    export_version: str | None,
    reference_score_column: str | None,
    column_specs: list[BundleColumnSpec],
) -> list[BundleReadinessCheck]:
    checks: list[BundleReadinessCheck] = []
    if export_version:
        checks.append(
            BundleReadinessCheck(
                code="export_version_detected",
                item="export_version",
                severity="info",
                detail=f"Detected export version {export_version}.",
            )
        )
    else:
        checks.append(
            BundleReadinessCheck(
                code="unknown_export_version",
                item="export_version",
                severity="warning",
                detail="No export version could be determined from artifact_manifest.json or run_config.json.",
            )
        )

    normalized_target_mode = str(target_mode or "").strip().lower()
    if normalized_target_mode and normalized_target_mode not in SUPPORTED_TARGET_MODES:
        checks.append(
            BundleReadinessCheck(
                code="unsupported_target_mode",
                item="target.mode",
                severity="error",
                detail=(
                    f"Target mode '{target_mode}' is not supported for OM Studio monitoring reruns."
                ),
            )
        )

    if paths.manifest_path is None:
        checks.append(
            BundleReadinessCheck(
                code="compat_missing_artifact_manifest",
                item="artifact_manifest.json",
                severity="warning",
                detail="Compatibility cannot verify the full exported inventory because artifact_manifest.json is missing.",
            )
        )
    if paths.reference_input_path is None:
        checks.append(
            BundleReadinessCheck(
                code="compat_missing_reference_input",
                item="input_snapshot.csv",
                severity="warning",
                detail="Reference vs monitoring row-count and missingness deltas will be limited.",
            )
        )
    if paths.reference_predictions_path is None:
        checks.append(
            BundleReadinessCheck(
                code="compat_missing_reference_predictions",
                item="predictions.csv",
                severity="warning",
                detail="Reference score diagnostics and deltas will be limited because predictions.csv is missing.",
            )
        )
    elif reference_score_column is None:
        checks.append(
            BundleReadinessCheck(
                code="compat_missing_reference_score_column",
                item="predictions.csv",
                severity="warning",
                detail="A supported reference score column was not detected in predictions.csv.",
            )
        )
    if paths.generated_runner_path is None:
        checks.append(
            BundleReadinessCheck(
                code="compat_missing_generated_runner",
                item="generated_run.py",
                severity="error",
                detail="The generated rerun launcher is missing, so this bundle is not compatible with raw-data reruns.",
            )
        )
    if paths.code_snapshot_path is None:
        checks.append(
            BundleReadinessCheck(
                code="compat_missing_code_snapshot",
                item="code_snapshot",
                severity="warning",
                detail="Portable reruns may depend on the local sister-project install because code_snapshot is missing.",
            )
        )
    if not column_specs:
        checks.append(
            BundleReadinessCheck(
                code="compat_missing_column_specs",
                item="schema.column_specs",
                severity="warning",
                detail="No enabled schema columns were found. Validation and demo compatibility checks are limited.",
            )
        )
    if not bundle_root.exists():
        checks.append(
            BundleReadinessCheck(
                code="compat_missing_bundle_root",
                item="bundle_root",
                severity="error",
                detail="The bundle root directory does not exist.",
            )
        )
    return checks


def _resolve_export_version(
    manifest_payload: dict[str, Any],
    config_payload: dict[str, Any],
) -> str | None:
    candidates = (
        manifest_payload.get("export_version"),
        manifest_payload.get("artifact_version"),
        manifest_payload.get("schema_version"),
        manifest_payload.get("framework_version"),
        config_payload.get("export_version"),
        config_payload.get("schema_version"),
        config_payload.get("framework_version"),
        config_payload.get("version"),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        value = str(candidate).strip()
        if value:
            return value
    return None


def _build_review_metadata_gaps(metadata: MonitoringMetadata) -> list[str]:
    gaps: list[str] = []
    for field_name, label in REVIEW_REQUIRED_METADATA_FIELDS.items():
        value = getattr(metadata, field_name, "")
        if not str(value or "").strip():
            gaps.append(label)
    return gaps


def _build_column_checks(
    *,
    bundle: ModelBundle,
    dataframe: pd.DataFrame,
    contract: InputContractResult,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    detected_columns = set(contract.detected_columns)
    for spec in bundle.column_specs:
        present = spec.source_name in detected_columns
        parse_success_pct: float | None = None
        status = "pass"
        detail = "Column is present and aligns with the saved schema."

        if not present:
            status = "fail" if spec.required_for_run else "warning"
            if spec.role == "target_source":
                detail = "Optional label column is missing. The run will continue in score-only mode."
            elif spec.role == "ignore":
                detail = "Optional schema-only column is missing. The app will inject a placeholder value."
            else:
                detail = "Required raw input column is missing."
        else:
            series = dataframe[spec.source_name]
            parse_success_pct, parse_warning = _dtype_parse_profile(series, spec.dtype)
            if parse_warning:
                status = "warning"
                detail = parse_warning

        rows.append(
            {
                "column_name": spec.source_name,
                "configured_name": spec.name,
                "role": spec.role,
                "required_for_run": spec.required_for_run,
                "present": present,
                "expected_dtype": spec.dtype or "n/a",
                "parse_success_pct": parse_success_pct,
                "status": status,
                "detail": detail,
            }
        )

    return pd.DataFrame(rows)


def _build_date_coverage(bundle: ModelBundle, dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in bundle.column_specs:
        if spec.role != "date":
            continue
        if spec.source_name not in dataframe.columns:
            rows.append(
                {
                    "column_name": spec.source_name,
                    "configured_name": spec.name,
                    "present": False,
                    "parse_success_pct": None,
                    "non_null_count": 0,
                    "min_date": None,
                    "max_date": None,
                    "detail": "Date column is missing from the monitoring dataset.",
                }
            )
            continue

        parsed = pd.to_datetime(dataframe[spec.source_name], errors="coerce")
        non_null_source = dataframe[spec.source_name].notna().sum()
        parse_success_pct = (
            float(parsed.notna().sum() / non_null_source * 100.0)
            if non_null_source
            else None
        )
        rows.append(
            {
                "column_name": spec.source_name,
                "configured_name": spec.name,
                "present": True,
                "parse_success_pct": parse_success_pct,
                "non_null_count": int(parsed.notna().sum()),
                "min_date": parsed.min(),
                "max_date": parsed.max(),
                "detail": "Date coverage derived from successfully parsed values.",
            }
        )

    return pd.DataFrame(rows)


def _dtype_parse_profile(series: pd.Series, dtype: str) -> tuple[float | None, str | None]:
    normalized = dtype.lower().strip()
    if not normalized or normalized in {"string", "str", "text", "category", "categorical"}:
        return 100.0, None

    source_non_null = series.notna().sum()
    if not source_non_null:
        return None, "Column is present but all values are missing."

    if normalized in {"float", "float64", "double", "int", "int64", "integer"}:
        parsed = pd.to_numeric(series, errors="coerce")
        success_pct = float(parsed.notna().sum() / source_non_null * 100.0)
        if success_pct < 100.0:
            return success_pct, (
                f"Only {success_pct:.1f}% of non-null values could be parsed as {normalized}."
            )
        return success_pct, None

    if normalized in {"datetime", "datetime64", "date"}:
        parsed = pd.to_datetime(series, errors="coerce")
        success_pct = float(parsed.notna().sum() / source_non_null * 100.0)
        if success_pct < 100.0:
            return success_pct, (
                f"Only {success_pct:.1f}% of non-null values could be parsed as {normalized}."
            )
        return success_pct, None

    if normalized in {"bool", "boolean"}:
        truthy_falsy = {"1", "0", "true", "false", "t", "f", "yes", "no", "y", "n"}
        parsed = series.astype("string").str.strip().str.lower().isin(truthy_falsy)
        success_pct = float(parsed.sum() / source_non_null * 100.0)
        if success_pct < 100.0:
            return success_pct, (
                f"Expected boolean-compatible values, but only {success_pct:.1f}% of non-null values matched."
            )
        return success_pct, None

    return None, None


def _template_example_value(bundle: ModelBundle, column_name: str) -> Any:
    spec = next(
        (
            item
            for item in bundle.column_specs
            if item.source_name == column_name or item.name == column_name
        ),
        None,
    )
    if spec is None:
        return None

    lowered = column_name.lower()
    if spec.role == "date" or spec.dtype.lower() in {"datetime", "datetime64", "date"}:
        return "2026-01-31"
    if spec.role == "identifier":
        return "EXAMPLE_0001"
    if spec.role == "target_source":
        return None
    if "income" in lowered:
        return 85000.0
    if "debt_to_income" in lowered:
        return 0.32
    if "utilization" in lowered:
        return 0.45
    if "delinquency" in lowered:
        return 1
    if "region" in lowered:
        return "north"
    if "employment" in lowered:
        return "salaried"
    if spec.dtype.lower() in {"float", "float64", "double"}:
        return 100.0
    if spec.dtype.lower() in {"int", "int64", "integer"}:
        return 1
    return "example_value"


def _column_note(bundle: ModelBundle, spec: BundleColumnSpec) -> str:
    if spec.role == "target_source":
        return (
            "Optional label column. Leave blank or omit values for score-only runs. "
            "Populate it when you want realized-performance tests."
        )
    if spec.role == "ignore":
        return "Optional schema-only field. The monitoring app can auto-fill a placeholder if omitted."
    if spec.required_for_run:
        return "Required raw input column for the saved model bundle."
    return "Optional input column."


def _path_fingerprint_row(label: str, path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "artifact_name": label,
            "path": str(path) if path is not None else "",
            "exists": False,
            "size_bytes": None,
            "modified_at_utc": None,
            "sha256": None,
        }

    stat = path.stat()
    return {
        "artifact_name": label,
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        "sha256": _sha256_file(path),
    }


def _directory_row(label: str, path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "artifact_name": label,
        "path": str(path),
        "exists": True,
        "size_bytes": None,
        "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        "sha256": None,
    }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _append_check(
    checks: list[BundleReadinessCheck],
    *,
    severity: str,
    code: str,
    item: str,
    detail: str,
) -> None:
    checks.append(
        BundleReadinessCheck(
            code=code,
            item=item,
            severity=severity,
            detail=detail,
        )
    )


def _count_csv_rows(path: Path | None) -> int | None:
    if path is None or not path.exists() or path.suffix.lower() != ".csv":
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        row_count = sum(1 for _ in handle)
    return max(row_count - 1, 0)


def _read_json_safely(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except (OSError, json.JSONDecodeError) as exc:
        return {}, str(exc)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
