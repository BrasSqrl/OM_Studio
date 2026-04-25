# Monitoring MVP Roadmap

## Goal

Build a narrow monitoring application for already-approved quantitative models.

The application must:
- watch a local `models/` directory for compliant model bundles
- recognize each bundle and surface model metadata in the UI
- watch a local `incoming_data/` directory for monitoring datasets
- run raw uploaded data through the selected approved model bundle
- execute monitoring tests relevant to model risk review
- apply user-editable per-test thresholds saved per model
- show per-test pass/fail in the UI for the current session
- export an HTML report and Excel workbook plus supporting monitoring artifacts

## MVP Workflow

1. A user drops one or more approved model bundles into `models/`.
2. The app scans `models/` and lists every compliant bundle.
3. A user drops one or more monitoring datasets into `incoming_data/`.
4. The app scans `incoming_data/` and lists each dataset.
5. The user selects a model bundle and a dataset.
6. The user reviews or edits saved thresholds for that model.
7. The app validates the raw-data input contract before scoring.
8. The app runs the selected bundle on the uploaded raw dataset.
9. The app computes monitoring diagnostics and per-test pass/fail outcomes.
10. The app writes artifacts under `runs/`.
11. The current-session run appears in the UI with test-level status and artifact paths.

## Compliance Contract

For MVP, the app supports a model only when it is a compliant rerunnable bundle.

Minimum required files in a bundle directory:
- `quant_model.joblib`
- `run_config.json`
- `generated_run.py`

Reference files strongly expected for full monitoring diagnostics:
- `input_snapshot.csv`
- `predictions.csv`
- `artifact_manifest.json`

Optional metadata file:
- `monitoring_metadata.json`

The app must prefer explicit metadata from `monitoring_metadata.json`.
If that file is absent, the app must infer metadata from `run_config.json` and the bundle directory name.

## Raw-Data Scoring Rule

MVP supports raw-data monitoring only.

The monitoring dataset is expected to contain the raw input fields required by the approved model contract.
The app must not require the user to pre-transform the uploaded dataset.

Scoring behavior:
- the bundle's rerun path owns preprocessing and scoring
- the monitoring app orchestrates the run and reads the scored outputs back in
- if the dataset includes labels, realized-performance tests run
- if labels are absent, label-dependent tests return `N/A`

## Watched Directories

The application will use these workspace directories:
- `models/`
- `incoming_data/`
- `thresholds/`
- `runs/`

The UI must scan these directories on load and when the user requests a refresh.

## Metadata To Surface In UI

Each discovered model should surface at least:
- model name
- model version
- model owner
- business purpose
- portfolio or segment name when present
- model type
- target mode
- bundle path
- reference row count when available
- required input columns
- optional label column
- available reference artifacts

## Monitoring Tests In MVP

Always available when inputs exist:
- required-column schema validation
- row-count review
- missingness review
- score distribution drift
- score PSI by band
- score-distribution p-value test
- segment stability view and segment PSI when a segment column is available

Available when labels are present for binary targets:
- AUC
- KS
- Gini
- Brier score
- MAE
- calibration review
- Hosmer-Lemeshow p-value
- precision at model threshold
- recall at model threshold
- bad-rate-by-band review
- bad-rate-by-band MAE

Available when labels are present for continuous targets:
- MAE

## Threshold Rules

Thresholds must be:
- defined per model bundle
- editable in the UI
- saved to disk
- reused on future runs

Threshold evaluation must be per test only.
MVP does not need a single overall run pass/fail flag.

## Artifact Outputs

Each monitoring run must produce:
- `monitoring_report.html`
- `monitoring_workbook.xlsx`
- `monitoring_tests.json`
- `monitoring_test_results.csv`
- `current_scored_data.csv`
- `artifacts_manifest.json`

Helpful supporting artifacts should also be written when available:
- missingness summary
- score band summary
- segment summary
- copied threshold snapshot
- copied run metadata snapshot

## UI Scope

The Streamlit UI should contain:
- watched-directory status
- discovered model bundle list
- discovered dataset list
- selected model metadata panel
- per-model threshold editor
- run action for selected model and dataset
- current-session run history
- per-test pass/fail display
- artifact links or file paths

## Explicitly Out Of Scope

Do not build:
- model training
- challenger development workflow
- feature subset search
- scheduling or automation
- external databases
- authentication or user management
- multi-user workflow state
- persistent run-history registry beyond the current session
- pre-transformed dataset mode
- standalone `.joblib` support without a compliant rerunnable bundle

## Implementation Phases

### Phase 1: Contracts And Discovery
- define workspace directories
- scan for compliant model bundles
- extract bundle metadata
- scan for monitoring datasets
- persist per-model thresholds

### Phase 2: Monitoring Execution
- validate raw input contract
- invoke bundle rerun scoring on raw monitoring data
- load reference artifacts and current scored outputs
- compute monitoring tests and pass/fail outcomes

### Phase 3: Reporting
- export HTML monitoring report
- export Excel workbook
- export supporting JSON and CSV artifacts

### Phase 4: UI
- replace starter shell with monitoring workflow
- add model and dataset selectors
- add threshold editor
- add current-session run table and result views

### Phase 5: Hardening
- add regression tests for discovery, thresholds, and monitoring calculations
- document the compliant bundle contract
- verify environment setup and launch flow

### Phase 6: UI/UX Cleanup And Reviewability
- add a fixed run-readiness banner that clearly states `Ready to run`, `Blocked`, or `Score-only`
- add colored status chips for `Pass`, `Warning`, `Fail`, and `N/A` across validation and result views
- add one-click filters to show only missing columns, warnings, and failed tests
- collapse lower-signal detail by default so the first screen emphasizes readiness and run decisions
- add a compact run outcome card with artifact download actions after each run
- clean up the left sidebar so watched-directory context, refresh actions, and session status are clearer and take less space

## Next Narrow UI/UX Enhancement Roadmap

These items should be implemented after the current monitoring MVP because they improve reviewer usability without widening scope.

### Goal

Improve clarity, usability, and reviewer speed in the Streamlit interface while preserving the same narrow workflow:
- select a compliant bundle
- select a watched dataset
- validate the contract
- run monitoring
- review per-test pass/fail and download artifacts

### Workstream 1: Left Sidebar Cleanup

Objective:
- make the left panel more usable, helpful, and visually cleaner

Deliverables:
- reduce sidebar text density and remove repeated path noise
- group workspace information into a compact `Watched Directories` section
- add short human-readable status cues for bundle count, dataset count, and session run count
- keep the refresh control visible and obvious without dominating the sidebar
- move low-frequency detail out of the sidebar if it is already available in the main page

Acceptance:
- a user can understand the workspace state and refresh action at a glance without reading a large code block

### Workstream 2: Fixed Run-Readiness Banner

Objective:
- make overall execution readiness visible without requiring the user to scan multiple tabs

Deliverables:
- add a top-of-page readiness banner
- show exactly one state: `Ready to run`, `Blocked`, or `Score-only`
- include the primary reason when blocked or downgraded
- update the banner when the selected bundle, dataset, or contract state changes

Acceptance:
- a user can immediately tell whether the current selection is runnable and why

### Workstream 3: Visual Status Chips

Objective:
- make pass/fail states easier to interpret in dense validation and result tables

Deliverables:
- add consistent colored chips or badges for `Pass`, `Warning`, `Fail`, and `N/A`
- use the same status language across bundle intake, pre-run contract, and latest-run results
- preserve plain-text values for export compatibility while improving in-app readability

Acceptance:
- a reviewer can scan tables visually and identify issues faster than with plain text alone

### Workstream 4: Problem-Only Filters

Objective:
- reduce review time by letting users focus on actionable issues first

Deliverables:
- add toggles to show only missing required columns, warnings, or failed tests
- support filtering in the intake validation area, pre-run contract view, and latest-run results
- keep the unfiltered full-table view available

Acceptance:
- a user can isolate only blocking or failed items with one click in each major review surface

### Workstream 5: Default-Collapsed Secondary Detail

Objective:
- keep the primary workflow readable by hiding low-signal operational detail until needed

Deliverables:
- place fingerprints, verbose readiness findings, and lower-priority technical detail inside expanders
- keep headline readiness, contract status, threshold actions, and run actions fully visible
- preserve access to all current detail without deleting any governance-relevant information

Acceptance:
- the main page focuses on decisions and actions first, with technical detail available on demand

### Workstream 6: Run Outcome Card

Objective:
- make the result of the latest run easier to interpret and act on immediately

Deliverables:
- add a compact post-run summary card
- show model name, model version, dataset name, timestamp, run status, pass count, fail count, and `N/A` count
- include direct HTML report and workbook download actions on the card
- keep the detailed test results table below the card

Acceptance:
- after a run completes, a reviewer can understand the outcome and download artifacts without scrolling through the full results table

### Recommended Implementation Order

1. Left sidebar cleanup
2. Fixed run-readiness banner
3. Visual status chips
4. Problem-only filters
5. Default-collapsed secondary detail
6. Run outcome card

### Delivery Rule

These enhancements must remain within the original narrow scope.

They should not introduce:
- new monitoring tests
- persistent run history beyond the current session
- multi-user workflow state
- automation or scheduling
- external services or databases
- changes to the raw-data-only compliant-bundle monitoring model

### Phase 7: Guided Workflow And Editing Clarity
- add a compact workflow step tracker that shows `Select Bundle`, `Select Dataset`, `Validate`, `Run`, and `Download`
- add unsaved-change indicators and a revert action in the thresholds editor
- pin missing required columns to the top of validation views and add column-name search
- add short per-test help content so reviewers can understand what each test means, what a fail implies, and when `N/A` is expected
- strengthen empty-state guidance when no compliant bundle or dataset is available

## Next Workflow Guidance Roadmap

These items should improve usability without changing the app's narrow purpose.

### Workstream 1: Compact Step Tracker

Objective:
- make the end-to-end workflow visible as a short sequence instead of forcing users to infer the next action

Deliverables:
- add a compact step tracker near the top of the page
- show the five narrow workflow stages: `Select Bundle`, `Select Dataset`, `Validate`, `Run`, `Download`
- visually mark each step as complete, current, or blocked based on the current app state

Acceptance:
- a user can tell where they are in the monitoring workflow and what the next required action is

### Workstream 2: Threshold Editing Safety

Objective:
- make threshold edits safer and more transparent in the UI

Deliverables:
- show whether threshold edits are saved or currently modified
- add a `Revert To Saved` action that discards unsaved editor changes for the selected bundle
- keep the existing per-model persistence behavior unchanged

Acceptance:
- a user can see unsaved threshold edits immediately and can reset to the saved file without confusion

### Workstream 3: Validation Search And Priority Sorting

Objective:
- make contract review faster on wider schemas by putting the highest-risk columns first

Deliverables:
- pin missing required columns to the top of the bundle-intake validation and pre-run contract views
- add a column-name search box for validation tables
- preserve the full-table view after filtering and sorting

Acceptance:
- blocking column problems appear first, and users can quickly find a named field in the validation results

### Workstream 4: Test Help Drawer

Objective:
- give model reviewers short contextual explanations for each monitoring test without requiring external documentation

Deliverables:
- add a `What This Means` help drawer for each supported monitoring test
- include a brief explanation of what the test measures
- include what a fail generally implies
- include when `N/A` is expected

Acceptance:
- a reviewer can understand each test from inside the app without widening the scope into training or documentation workflows

### Workstream 5: Stronger Empty States

Objective:
- reduce first-run confusion when no bundle or no dataset is available yet

Deliverables:
- replace generic empty-state messages with explicit action guidance
- show exact watched folders and minimum bundle files when no compliant bundle is available
- show dataset intake guidance when no monitoring dataset is available

Acceptance:
- a first-time user can recover from an empty workspace without guessing what to drop where

### Recommended Implementation Order

1. Stronger empty states
2. Compact step tracker
3. Validation search and priority sorting
4. Threshold editing safety
5. Test help drawer

### Delivery Rule

These changes must stay within the existing narrow OM Studio workflow.

They must not introduce:
- directory configurability
- new monitoring tests
- persistent review state beyond the current session
- external documentation dependencies

### Phase 8: Operational Hardening And Reviewer Flow
- add a formal bundle compatibility check with explicit compatibility status and findings
- add end-to-end golden-path regression coverage using a compliant demo bundle fixture
- show reference vs monitoring values plus deltas for key diagnostics in the UI and exported artifacts
- add dataset sample preview and inferred dtypes to the validation workflow
- improve run failure diagnostics with failing step, captured output, and reviewer-visible failure detail
- add metadata completeness checks so bundles can be runnable but still flagged as review-incomplete
- add threshold reset and recommended-default actions while keeping per-model persistence
- add a single downloadable reviewer package that bundles the main monitoring artifacts
- add current-session run comparison for two selected runs
- add a built-in demo path so the app can always be smoke-tested without manual external setup

## Next Operational Hardening Roadmap

These items should strengthen reviewer usability and reduce avoidable run failures while preserving the app's narrow purpose.

### Workstream 1: Bundle Compatibility Contract

Objective:
- classify whether a discovered bundle is merely runnable or fully compatible with the richer monitoring workflow

Deliverables:
- explicit compatibility checks for export metadata, reference artifacts, rerun support, and supported target shape
- compatibility status and findings surfaced in the UI
- separation between execution readiness and compatibility warnings

Acceptance:
- a user can tell whether a bundle is `compatible`, `compatible_with_warnings`, or `not_compatible` before running it

### Workstream 2: Golden-Path Regression Coverage

Objective:
- lock down the narrow monitoring flow with a single realistic end-to-end fixture path

Deliverables:
- reusable compliant demo bundle fixture
- regression test covering discover, validate, score, report, and artifact package creation
- deterministic outputs suitable for local smoke tests

Acceptance:
- the core monitoring flow is covered by one end-to-end test rather than only isolated unit tests

### Workstream 3: Reference Vs Monitoring Delta View

Objective:
- make key changes between reference and monitoring populations visible without requiring manual comparison

Deliverables:
- one concise delta table with `reference`, `monitoring`, `delta`, and `status`
- inclusion of the delta table in the UI and HTML/Excel artifacts
- focus on the most decision-relevant diagnostics only

Acceptance:
- a reviewer can see how current data differs from the reference bundle in one place

### Workstream 4: Dataset Sample Preview

Objective:
- make obvious dataset issues visible before execution

Deliverables:
- dataset head preview
- inferred dtype summary
- placement inside the validation workflow rather than as a separate feature area

Acceptance:
- a user can catch simple upload mistakes without leaving the Validate tab

### Workstream 5: Failure Diagnostics

Objective:
- make execution failures actionable instead of opaque

Deliverables:
- failing stage capture
- command or runner context when relevant
- captured stdout/stderr excerpt
- exported failure diagnostics artifact

Acceptance:
- a failed run explains where it failed and what the user should inspect next

### Workstream 6: Reviewer Metadata Completeness

Objective:
- distinguish between a bundle that can run and a bundle that is review-ready

Deliverables:
- metadata completeness check for owner, version, purpose, approval state, and approval date
- `review_complete` vs `review_incomplete` state in the UI
- non-blocking warnings only

Acceptance:
- reviewers can identify bundles that are runnable but not fully documented

### Workstream 7: Threshold Defaults And Reset

Objective:
- make threshold editing safer and faster

Deliverables:
- `Reset To Saved`
- `Apply Recommended Defaults`
- preservation of per-model saved threshold files

Acceptance:
- a user can recover from bad edits quickly and can repopulate a threshold set with recommended defaults

### Workstream 8: Reviewer Package

Objective:
- reduce manual artifact gathering for reviewer handoff

Deliverables:
- zip package containing the main report, workbook, threshold snapshot, manifest, and test outputs
- package path and download action surfaced in the UI

Acceptance:
- one download contains the primary reviewer-facing monitoring artifacts

### Workstream 9: Current-Session Run Comparison

Objective:
- help users identify what changed between two runs in the same session

Deliverables:
- run selector pair
- comparison summary card
- changed-test table with status and metric differences

Acceptance:
- a user can compare two session runs without exporting or manually diffing files

### Workstream 10: Built-In Demo Path

Objective:
- make OM Studio self-demonstrable and easier to smoke-test

Deliverables:
- built-in demo asset generator for a compliant bundle and dataset
- safe non-destructive generation inside the watched workspace
- documentation of the demo path

Acceptance:
- a fresh workspace can be exercised end-to-end without any external bundle or dataset

### Recommended Implementation Order

1. Bundle compatibility contract
2. Golden-path regression coverage
3. Reference vs monitoring delta view
4. Dataset sample preview
5. Failure diagnostics
6. Reviewer metadata completeness
7. Threshold defaults and reset
8. Reviewer package
9. Current-session run comparison
10. Built-in demo path

### Delivery Rule

These changes must preserve the narrow OM Studio workflow.

They must not introduce:
- training or challenger workflows
- scheduled monitoring
- external services or storage
- persistent run comparison beyond the current session

### Phase 9: Monitoring Bundle Depth And Reviewer Artifacts
- prefer a first-class `monitoring_bundle/` intake shape when exported by the sister project while continuing to support existing compliant bundle directories
- add feature-level drift tests and supporting tables for numeric and categorical raw input fields
- add data-quality threshold tests for duplicate identifiers, identifier nulls, invalid dates, stale monitoring dates, unexpected categories, numeric range violations, and absolute row count
- support an optional outcome-file join so realized labels can be supplied separately from scoring data
- make per-model test enablement explicit in saved threshold snapshots and exported reviewer artifacts
- persist a lightweight run manifest index under `runs/` for reviewer traceability without adding a database or cross-session comparison workflow
- allow reviewer exception notes to be captured at run time and exported with test results
- add reference-baseline diagnostics for required reference artifacts, rows, score columns, labels, and date coverage
- add an executive summary sheet and report section focused on run status, test counts, and failed-test names
- add a lightweight UI regression checklist/screenshot script for consistent manual checks of the narrow workflow

## Next Monitoring Depth Roadmap

These items improve reviewer coverage and operational traceability while keeping OM Studio focused on existing approved model bundles and uploaded monitoring data.

### Workstream 1: First-Class Monitoring Bundle Intake

Objective:
- consume a sister-project `monitoring_bundle/` output directory as the preferred contract for OM Studio

Deliverables:
- detect bundles whose root is named `monitoring_bundle`
- surface preferred-vs-legacy bundle shape in compatibility findings
- keep existing compliant bundle directories runnable

Acceptance:
- a `models/<model>/monitoring_bundle/` directory with the required files appears as a runnable bundle and is marked as the preferred intake shape

### Workstream 2: Feature-Level Drift Tests

Objective:
- show which raw features drifted, not only whether the score distribution drifted

Deliverables:
- compute per-feature PSI for numeric and categorical bundle features
- compute numeric-feature KS p-values where applicable
- add threshold-backed tests for max feature PSI and minimum numeric feature KS p-value
- export the feature drift table to CSV, Excel, and HTML

Acceptance:
- every run with a reference input snapshot includes feature-level drift diagnostics and per-test pass/fail results

### Workstream 3: Data-Quality Threshold Tests

Objective:
- catch common monitoring-data issues before relying on model performance diagnostics

Deliverables:
- add threshold-backed tests for absolute row count, duplicate identifiers, identifier null rate, invalid dates, stale as-of dates, unexpected categories, and numeric range violations
- export a data-quality summary table
- keep defaults editable and persisted per model

Acceptance:
- data-quality tests appear in the same threshold editor and run results as existing monitoring tests

### Workstream 4: Optional Outcome-File Join

Objective:
- allow labels to be supplied in a separate watched dataset without requiring users to edit the scoring input file

Deliverables:
- add an optional outcome dataset selector in the Run tab
- infer sensible join keys from identifier and date columns
- merge the label column into the scoring input before contract validation and scoring
- export an outcome join summary table

Acceptance:
- a score-only input dataset can become a full monitoring run when a matching outcome file supplies the saved label column

### Workstream 5: Per-Model Test Enablement Profile

Objective:
- make enabled/disabled tests auditable as a per-model monitoring profile

Deliverables:
- preserve the existing per-model threshold persistence
- export an explicit test enablement profile table and artifact
- include disabled tests as `N/A` with clear detail

Acceptance:
- reviewers can see which tests were active for the model at run time without inspecting the UI state

### Workstream 6: Run Manifest Persistence

Objective:
- create durable run traceability without adding a database or cross-session comparison feature

Deliverables:
- write or update `runs/index.json` after each run
- include run id, timestamp, model, dataset, status, counts, and artifact paths
- show persisted run records in the History tab as read-only context

Acceptance:
- closing and reopening the app does not erase basic run traceability from disk

### Workstream 7: Reviewer Exception Notes

Objective:
- let reviewers document test-level exceptions or remediation context in the exported package

Deliverables:
- add a small notes editor in the Run tab keyed by test id
- carry notes into test-result exports, JSON, workbook, HTML, and reviewer package
- avoid introducing approval workflow state

Acceptance:
- a reviewer note entered for a test appears in the generated artifacts for that run

### Workstream 8: Reference-Baseline Diagnostics

Objective:
- make the quality of the reference baseline visible before and after a run

Deliverables:
- summarize reference input and prediction artifact availability
- report reference rows, score column, label availability, and date coverage
- surface the diagnostics in the Overview tab and exported artifacts

Acceptance:
- reviewers can identify weak or incomplete reference baselines without opening raw bundle files

### Workstream 9: Report Executive Summary

Objective:
- put the reviewer decision summary first in exported artifacts

Deliverables:
- add an executive summary table with status, counts, score mode, and failed-test names
- make it the first workbook sheet
- add an executive summary section near the top of the HTML report

Acceptance:
- a reviewer can open the report or workbook and immediately see the run outcome and failed tests

### Workstream 10: UI Regression Checklist

Objective:
- give a repeatable way to check the narrow UI workflow after design or logic changes

Deliverables:
- define the expected UI states to capture
- provide an optional screenshot script for a running local Streamlit app
- keep this as local tooling only, with no production dependency

Acceptance:
- contributors have a documented checklist and optional script for validating the main UI states

### Recommended Implementation Order

1. First-class monitoring bundle intake
2. Feature-level drift tests
3. Data-quality threshold tests
4. Outcome-file join workflow
5. Per-model test enablement profile
6. Run manifest persistence
7. Reviewer exception notes
8. Reference-baseline diagnostics
9. Report executive summary
10. UI regression checklist

### Delivery Rule

These enhancements must stay inside the original OM Studio scope.

They must not introduce:
- model training
- challenger workflows
- scheduled runs
- external services
- external databases
- authentication or multi-user review state

### Phase 10: Runtime Optimization And Debuggability
- split high-growth execution logic into smaller modules for metrics, support tables, telemetry, and artifact settings
- add run-stage telemetry so every run exports stage timings and debug events
- cache file-derived state using path, size, and modified-time keys to avoid repeated reads and hashes
- centralize data-quality and feature-drift profile construction so tests and support tables reuse the same calculations
- add configurable artifact profiles to reduce unnecessary report/support-output work during faster reruns
- keep subprocess scoring as the compatibility-first default while preserving an extension point for future direct in-process scoring
- add a local profiling script for repeatable demo and large-row runtime analysis

## Next Runtime Optimization Roadmap

These items make OM Studio easier to debug and faster to run without changing its narrow purpose.

### Workstream 1: Smaller Execution Modules

Objective:
- reduce the size and responsibility overlap in the pipeline module

Deliverables:
- move run telemetry into a dedicated module
- move feature drift and data-quality profiles into a dedicated metrics module
- move support-table assembly into a dedicated support module
- keep the public `execute_monitoring_run` API stable

Acceptance:
- pipeline orchestration reads as load -> validate -> score -> test -> report, with lower-level details delegated to focused modules

### Workstream 2: Stage-Level Telemetry

Objective:
- make slow or failing runs easy to diagnose from exported artifacts

Deliverables:
- capture timing events for dataset load, outcome join, validation, scoring, reference loading, test evaluation, support-table assembly, artifact writing, and run-index persistence
- write `run_events.jsonl` under each run directory
- include telemetry in the artifact manifest and reviewer package

Acceptance:
- every run folder contains a readable event log showing stage duration and status

### Workstream 3: File-Derived Caching

Objective:
- avoid repeated reads and hashes of unchanged local files

Deliverables:
- cache dataset header reads by path, size, and modified time
- cache CSV reference reads by path, size, and modified time
- cache file SHA256 values by path, size, and modified time
- return defensive DataFrame copies to avoid shared-state mutation

Acceptance:
- repeated UI refreshes and repeated diagnostics avoid redundant disk work when files are unchanged

### Workstream 4: Reusable Metric Profiles

Objective:
- compute raw-data profiles once and reuse them for tests, support tables, and reports

Deliverables:
- centralize data-quality summary construction
- centralize feature-drift summary construction
- keep existing threshold outputs unchanged

Acceptance:
- data-quality and feature-drift tests use the same tables exported to artifacts

### Workstream 5: Artifact Profiles

Objective:
- reduce runtime when users only need the primary reviewer artifacts

Deliverables:
- add an artifact profile setting with `full`, `reviewer`, and `minimal`
- keep HTML report and Excel workbook in all profiles
- skip non-essential support CSVs and reviewer zip where configured

Acceptance:
- users can reduce run-output work without losing the stitched report and workbook

### Workstream 6: Profiling Tooling

Objective:
- make future runtime changes measurable

Deliverables:
- add a local profiling script for demo monitoring runs
- support configurable monitoring row counts
- print stage outputs and cProfile top functions

Acceptance:
- contributors can run one command and see where runtime is being spent

### Recommended Implementation Order

1. Stage-level telemetry
2. File-derived caching
3. Reusable metric profiles
4. Smaller execution modules
5. Artifact profiles
6. Profiling tooling

### Delivery Rule

These changes must preserve:
- raw-data-only monitoring
- compliant bundle intake
- HTML and Excel reviewer outputs
- per-test pass/fail threshold semantics

### Phase 11: Reviewer Hardening And Narrow Workflow Refinement
- add explicit monitoring bundle contract versioning so users can distinguish supported, deprecated, missing, and unsupported bundle exports
- add a downloadable run-comparison export for two current-session runs
- add a structured reviewer exception workflow that remains run-scoped and does not become an approval system
- add threshold change audit logging for per-model benchmark edits
- add an artifact completeness checker that verifies expected run outputs were generated or intentionally skipped
- add a model bundle intake checklist for reviewer-facing bundle readiness
- add large dataset guardrails that warn or block based on row/column size limits
- add a test applicability matrix so reviewers can see why each test ran, failed, or was marked `N/A`
- polish reviewer reports so decision summaries, exceptions, intake status, applicability, and artifact completeness are easy to find
- add a one-click diagnostic export package for run debugging

## Next Reviewer Hardening Roadmap

These items improve reviewer confidence and troubleshooting speed while preserving the original narrow workflow: select a compliant exported model bundle, select watched monitoring data, validate, run, review per-test pass/fail, and download artifacts.

### Workstream 1: Bundle Contract Versioning

Objective:
- make the sister-project bundle contract explicit and future-safe

Deliverables:
- supported contract version policy in code
- compatibility findings for missing, supported, deprecated, future, or unsupported bundle versions
- version status surfaced in the UI and exported metadata

Acceptance:
- a user can tell whether an uploaded bundle uses a supported OM Studio intake contract before running it

### Workstream 2: Run Comparison Export

Objective:
- let reviewers preserve current-session run comparisons without manually copying tables

Deliverables:
- downloadable HTML comparison
- downloadable Excel workbook comparison
- summary and changed-test tables

Acceptance:
- two current-session runs can be selected and exported as reviewer-facing comparison artifacts

### Workstream 3: Reviewer Exception Workflow

Objective:
- let reviewers document test-level exceptions without adding approvals or multi-user state

Deliverables:
- run-scoped exception disposition and rationale fields
- exception details exported to JSON, CSV, Excel, HTML, and reviewer package
- exception summary in the UI and report

Acceptance:
- a failed or `N/A` test can carry reviewer context in the run artifacts

### Workstream 4: Threshold Change Audit

Objective:
- make per-model threshold edits traceable

Deliverables:
- JSONL audit log under `thresholds/audit/`
- changed value/enabled/operator fields captured per save
- UI view of recent threshold audit events

Acceptance:
- reviewers can see when saved thresholds changed and what changed

### Workstream 5: Artifact Completeness Checker

Objective:
- make generated, missing, and skipped run artifacts explicit

Deliverables:
- `artifact_completeness.csv`
- `artifact_completeness.json`
- completeness status in the manifest, report, workbook, UI, reviewer package, and diagnostic package

Acceptance:
- a run folder explains whether expected artifacts exist or were intentionally skipped by profile

### Workstream 6: Model Bundle Intake Checklist

Objective:
- convert compatibility findings into a simple reviewer checklist

Deliverables:
- required/recommended file checklist
- metadata, target, schema, reference, and version checklist rows
- UI and exported artifact coverage

Acceptance:
- a reviewer can inspect intake readiness without interpreting raw bundle files

### Workstream 7: Large Dataset Guardrails

Objective:
- reduce avoidable slow or memory-heavy runs

Deliverables:
- warning thresholds for large row/column counts
- hard-stop thresholds for oversized row/column counts
- guardrail table in validation UI

Acceptance:
- users are warned before heavy runs and blocked before configured oversized runs

### Workstream 8: Test Applicability Matrix

Objective:
- explain why each test ran, was disabled, or returned `N/A`

Deliverables:
- per-test applicability table tied to thresholds and actual run results
- workbook, report, UI, and reviewer package inclusion

Acceptance:
- reviewers can see the reason behind every active, disabled, or unavailable test

### Workstream 9: Reviewer Report Polish

Objective:
- make exported reports more decision-oriented

Deliverables:
- prominent failed-test and exception summary
- intake checklist, applicability, and artifact completeness sections
- clearer reviewer action context

Acceptance:
- the HTML and Excel reports put reviewer decisions and caveats before lower-level diagnostics

### Workstream 10: Diagnostic Export Package

Objective:
- make run debugging portable without requiring users to zip files manually

Deliverables:
- `diagnostic_export.zip`
- inclusion of manifests, telemetry, debug traces, failure diagnostics, contract, thresholds, provenance, completeness, and tests
- UI download action

Acceptance:
- a failed or questionable run can be shared for debugging with one download

### Recommended Implementation Order

1. Bundle contract versioning
2. Model bundle intake checklist
3. Large dataset guardrails
4. Test applicability matrix
5. Threshold change audit
6. Reviewer exception workflow
7. Artifact completeness checker
8. Diagnostic export package
9. Run comparison export
10. Reviewer report polish

### Delivery Rule

These changes must not introduce:
- model training or challenger workflows
- scheduled monitoring
- external services or databases
- authentication or multi-user workflow state
- persistent comparison workflow beyond current-session exports
