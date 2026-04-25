"""Formal intake contract for sister-project monitoring bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_BUNDLE_TYPES = {
    "quant_studio_model_bundle_for_monitoring",
    "monitoring_bundle",
    "om_studio_monitoring_bundle",
}
PREFERRED_BUNDLE_DIRECTORY_NAME = "model_bundle_for_monitoring"
LEGACY_BUNDLE_DIRECTORY_NAME = "monitoring_bundle"
REQUIRED_BUNDLE_FILES = ("quant_model.joblib", "run_config.json", "generated_run.py")
RECOMMENDED_BUNDLE_FILES = (
    "monitoring_metadata.json",
    "artifact_manifest.json",
    "predictions.csv",
)
OPTIONAL_BUNDLE_FILES = ("input_snapshot.csv", "code_snapshot")
SUPPORTED_TARGET_MODES = {"binary", "continuous", "regression"}
CURRENT_MONITORING_BUNDLE_VERSION = "1.0"
MONITORING_BUNDLE_VERSION = CURRENT_MONITORING_BUNDLE_VERSION
SUPPORTED_MONITORING_BUNDLE_VERSIONS = frozenset({CURRENT_MONITORING_BUNDLE_VERSION})
DEPRECATED_MONITORING_BUNDLE_VERSIONS = frozenset[str]()


@dataclass(frozen=True, slots=True)
class BundleContractFinding:
    code: str
    item: str
    severity: str
    detail: str


@dataclass(frozen=True, slots=True)
class BundleContractResult:
    preferred_shape: bool
    findings: list[BundleContractFinding]

    @property
    def status(self) -> str:
        severities = {finding.severity for finding in self.findings}
        if "error" in severities:
            return "not_compatible"
        if "warning" in severities:
            return "compatible_with_warnings"
        return "compatible"


def is_preferred_monitoring_bundle(root: Path, manifest_payload: dict[str, Any]) -> bool:
    bundle_type = normalized_bundle_type(manifest_payload)
    return root.name == PREFERRED_BUNDLE_DIRECTORY_NAME or bundle_type in {
        "quant_studio_model_bundle_for_monitoring",
        "model_bundle_for_monitoring",
    }


def is_supported_monitoring_bundle(root: Path, manifest_payload: dict[str, Any]) -> bool:
    bundle_type = normalized_bundle_type(manifest_payload)
    return root.name in {
        PREFERRED_BUNDLE_DIRECTORY_NAME,
        LEGACY_BUNDLE_DIRECTORY_NAME,
    } or bundle_type in SUPPORTED_BUNDLE_TYPES


def normalized_bundle_type(manifest_payload: dict[str, Any]) -> str:
    return str(
        manifest_payload.get("bundle_type")
        or manifest_payload.get("om_studio_bundle_type")
        or manifest_payload.get("type")
        or ""
    ).strip().lower()


def resolve_monitoring_bundle_version(
    *,
    manifest_payload: dict[str, Any],
    metadata_payload: dict[str, Any],
    config_payload: dict[str, Any] | None = None,
) -> str | None:
    """Return the first explicit monitoring contract version advertised by a bundle."""

    config_payload = config_payload or {}
    candidates = (
        metadata_payload.get("monitoring_contract_version"),
        metadata_payload.get("bundle_version"),
        metadata_payload.get("contract_version"),
        manifest_payload.get("monitoring_contract_version"),
        manifest_payload.get("bundle_version"),
        manifest_payload.get("contract_version"),
        manifest_payload.get("schema_version"),
        config_payload.get("monitoring_contract_version"),
        config_payload.get("contract_version"),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        value = str(candidate).strip()
        if value:
            return value
    return None


def classify_monitoring_bundle_version(version: str | None) -> str:
    if not version:
        return "missing"
    normalized = str(version).strip()
    if normalized in SUPPORTED_MONITORING_BUNDLE_VERSIONS:
        return "supported"
    if normalized in DEPRECATED_MONITORING_BUNDLE_VERSIONS:
        return "deprecated"
    parsed = _parse_version(normalized)
    current = _parse_version(CURRENT_MONITORING_BUNDLE_VERSION)
    if parsed is not None and current is not None and parsed > current:
        return "future"
    return "unsupported"


def validate_monitoring_bundle_contract(
    *,
    root: Path,
    manifest_payload: dict[str, Any],
    metadata_payload: dict[str, Any],
    config_payload: dict[str, Any] | None = None,
    target_mode: str,
    reference_score_column: str | None,
    column_count: int,
) -> BundleContractResult:
    """Returns reviewer-facing compatibility findings for a monitoring bundle."""

    findings: list[BundleContractFinding] = []
    preferred_shape = is_preferred_monitoring_bundle(root, manifest_payload)
    bundle_type = normalized_bundle_type(manifest_payload)
    bundle_version = resolve_monitoring_bundle_version(
        manifest_payload=manifest_payload,
        metadata_payload=metadata_payload,
        config_payload=config_payload,
    )
    bundle_version_status = classify_monitoring_bundle_version(bundle_version)

    if preferred_shape:
        findings.append(
            BundleContractFinding(
                code="preferred_model_bundle_for_monitoring_shape",
                item=PREFERRED_BUNDLE_DIRECTORY_NAME,
                severity="info",
                detail=(
                    "Bundle uses the sister-project preferred "
                    "model_bundle_for_monitoring intake shape."
                ),
            )
        )
    elif is_supported_monitoring_bundle(root, manifest_payload):
        findings.append(
            BundleContractFinding(
                code="legacy_monitoring_bundle_shape",
                item=root.name,
                severity="warning",
                detail=(
                    "Bundle is runnable, but the preferred sister-project output "
                    "directory is model_bundle_for_monitoring."
                ),
            )
        )
    else:
        findings.append(
            BundleContractFinding(
                code="unknown_bundle_shape",
                item=root.name,
                severity="warning",
                detail=(
                    "Bundle shape is inferred from required files. Prefer a "
                    "model_bundle_for_monitoring directory exported by Quant Studio."
                ),
            )
        )

    for file_name in REQUIRED_BUNDLE_FILES:
        if not (root / file_name).exists():
            findings.append(
                BundleContractFinding(
                    code=f"missing_required_{file_name.replace('.', '_')}",
                    item=file_name,
                    severity="error",
                    detail=f"Required bundle file {file_name} is missing.",
                )
            )

    for file_name in RECOMMENDED_BUNDLE_FILES:
        if not (root / file_name).exists():
            findings.append(
                BundleContractFinding(
                    code=f"missing_recommended_{file_name.replace('.', '_')}",
                    item=file_name,
                    severity="warning",
                    detail=(
                        f"Recommended bundle file {file_name} is missing. "
                        "Some reviewer diagnostics may be limited."
                    ),
                )
            )

    for file_name in OPTIONAL_BUNDLE_FILES:
        if not (root / file_name).exists():
            findings.append(
                BundleContractFinding(
                    code=f"missing_optional_{file_name.replace('.', '_')}",
                    item=file_name,
                    severity="info",
                    detail=f"Optional bundle file {file_name} was not exported.",
                )
            )

    if bundle_type and bundle_type not in SUPPORTED_BUNDLE_TYPES:
        findings.append(
            BundleContractFinding(
                code="unsupported_bundle_type",
                item="bundle_type",
                severity="warning",
                detail=f"Bundle type '{bundle_type}' is not a known OM Studio contract type.",
            )
        )

    metadata_bundle_type = str(metadata_payload.get("bundle_type", "")).strip().lower()
    if metadata_bundle_type and metadata_bundle_type not in SUPPORTED_BUNDLE_TYPES:
        findings.append(
            BundleContractFinding(
                code="unsupported_metadata_bundle_type",
                item="monitoring_metadata.bundle_type",
                severity="warning",
                detail=(
                    f"Monitoring metadata bundle type '{metadata_bundle_type}' "
                    "is not a known OM Studio contract type."
                ),
            )
        )

    if bundle_version_status == "missing":
        findings.append(
            BundleContractFinding(
                code="missing_monitoring_bundle_version",
                item="monitoring_contract_version",
                severity="warning",
                detail=(
                    "The bundle does not declare an OM Studio monitoring contract "
                    f"version. Current supported contract is {CURRENT_MONITORING_BUNDLE_VERSION}."
                ),
            )
        )
    elif bundle_version_status == "supported":
        findings.append(
            BundleContractFinding(
                code="supported_monitoring_bundle_version",
                item="monitoring_contract_version",
                severity="info",
                detail=f"Bundle declares supported monitoring contract version {bundle_version}.",
            )
        )
    elif bundle_version_status == "deprecated":
        findings.append(
            BundleContractFinding(
                code="deprecated_monitoring_bundle_version",
                item="monitoring_contract_version",
                severity="warning",
                detail=(
                    f"Bundle declares deprecated monitoring contract version {bundle_version}. "
                    f"Current supported contract is {CURRENT_MONITORING_BUNDLE_VERSION}."
                ),
            )
        )
    elif bundle_version_status == "future":
        findings.append(
            BundleContractFinding(
                code="future_monitoring_bundle_version",
                item="monitoring_metadata.bundle_version",
                severity="warning",
                detail=(
                    f"Bundle declares future monitoring contract version {bundle_version}. "
                    f"Current supported contract is {CURRENT_MONITORING_BUNDLE_VERSION}; "
                    "review compatibility before relying on all diagnostics."
                ),
            )
        )
    else:
        findings.append(
            BundleContractFinding(
                code="unsupported_monitoring_bundle_version",
                item="monitoring_contract_version",
                severity="error",
                detail=(
                    f"Bundle declares unsupported monitoring contract version {bundle_version}. "
                    f"Supported versions: {', '.join(sorted(SUPPORTED_MONITORING_BUNDLE_VERSIONS))}."
                ),
            )
        )

    normalized_target_mode = str(target_mode or "").strip().lower()
    if normalized_target_mode and normalized_target_mode not in SUPPORTED_TARGET_MODES:
        findings.append(
            BundleContractFinding(
                code="unsupported_target_mode",
                item="target.mode",
                severity="error",
                detail=(
                    f"Target mode '{target_mode}' is not supported for "
                    "OM Studio monitoring reruns."
                ),
            )
        )

    if (root / "predictions.csv").exists() and reference_score_column is None:
        findings.append(
            BundleContractFinding(
                code="missing_reference_score_column",
                item="predictions.csv",
                severity="warning",
                detail="A supported reference prediction column was not found.",
            )
        )

    if column_count <= 0:
        findings.append(
            BundleContractFinding(
                code="missing_column_contract",
                item="schema.column_specs",
                severity="warning",
                detail="No enabled schema columns were found in run_config.json.",
            )
        )

    return BundleContractResult(
        preferred_shape=preferred_shape,
        findings=findings,
    )


def _parse_version(value: str) -> tuple[int, ...] | None:
    parts = value.split(".")
    parsed: list[int] = []
    for part in parts:
        if not part.isdigit():
            return None
        parsed.append(int(part))
    return tuple(parsed)
