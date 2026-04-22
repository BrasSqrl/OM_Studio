# OM Studio UI Composition Pattern

## Pattern Name

Focused Workspace Pattern

## Goal

Reduce cognitive load in the main workspace by making the page answer only three questions at a time:

1. What is selected?
2. What is the current decision state?
3. What should the user do next?

This pattern is intended for OM Studio's narrow workflow only:
- select a compliant bundle
- select a monitoring dataset
- validate the contract
- run monitoring
- review pass/fail and download artifacts

## Core Rules

### 1. One Control Deck

The top of the main workspace should contain a single control surface for:
- model bundle selection
- dataset selection
- optional segment selection
- current decision state
- next recommended action

The top area should not contain multiple competing summary blocks.

### 2. One Primary Message

At any moment, the page should surface exactly one primary message:
- blocked
- ready
- score-only
- completed and ready for download

If more detail is needed, it belongs below the primary message, not beside it.

### 3. Task Tabs, Not Summary Stacks

Primary navigation should be task-based:
- Overview
- Validate
- Thresholds
- Run
- History

Each tab should have one main purpose. Cross-tab duplication should be avoided.

### 4. One Dense Object Per Tab

Each tab may have one main dense object visible at a time:
- one summary table
- one editor
- one validation grid
- one results table

Secondary content should be hidden behind expanders or sub-tabs.

### 5. Details On Demand

Low-frequency governance detail belongs in expanders:
- fingerprints
- readiness findings
- help text
- artifact paths
- auxiliary summaries

These items remain accessible but should not compete with the primary workflow.

### 6. No Duplicate Summaries

The same information should not appear in multiple stacked summary surfaces.

Examples:
- readiness should not appear as a metric row, a banner, and a step tracker at the same time
- run outcome should not be summarized in multiple large cards on the same screen

## Mapping To OM Studio

### Control Deck

Top-of-page area:
- selectors
- current state
- next action

### Overview Tab

Purpose:
- understand the selected bundle and dataset at a glance

Contains:
- concise selection summary
- metadata editor
- input-template action
- expanders for bundle details

### Validate Tab

Purpose:
- fix blocking issues before execution

Contains:
- one primary validation table or sub-view
- search and problem filters
- expanders for secondary detail

### Thresholds Tab

Purpose:
- edit, save, or revert thresholds

Contains:
- one threshold editor
- one saved/modified state indicator
- collapsed help drawer

### Run Tab

Purpose:
- execute monitoring and review the current run

Contains:
- one run action area
- one outcome card
- one results view
- collapsed artifact/help detail

### History Tab

Purpose:
- review current-session runs only

Contains:
- one run-history table

## Rewrite Rule

When changing the Streamlit layout, prefer these decisions in order:

1. remove duplicate summary elements
2. reduce the number of simultaneously visible dense objects
3. move secondary information into expanders
4. preserve workflow clarity before adding visual treatment

If a proposed UI element does not improve the user's answer to:
- what is selected
- what state am I in
- what should I do next

then it should probably not be visible by default.
