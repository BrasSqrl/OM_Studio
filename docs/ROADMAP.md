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
