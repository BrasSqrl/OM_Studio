# Architecture

## Goal

Build a separate monitoring application that feels like Quant Studio without turning the development application into a mixed-scope product.

## Recommended Shape

Use a fully separate application with its own:
- package name
- launcher
- Streamlit app
- configuration surface
- monitoring pipeline
- report outputs
- git repository

## Shared-by-Design Components

The following components should be intentionally aligned with the development app:
- color palette and typography
- chart theme
- section layout and card patterns
- report shell and exported HTML visual language
- artifact naming conventions where they make sense
- documentation style and audit-oriented structure

## Keep Separate

The following should remain monitoring-specific:
- model registry and approved-model intake
- monitoring execution modes
- drift and performance monitoring pipeline
- realized-outcome monitoring logic
- alert and threshold logic
- monitoring report taxonomy
- periodic or batch monitoring workflows

## Best Technical Pattern

Phase 1:
- keep this repo self-contained
- copy only minimal shared visual helpers needed to move quickly

Phase 2:
- extract a small shared internal package such as `quant_studio_shared`
- move only stable, presentation-level utilities into that package
- keep monitoring and development pipelines separate

## Product Taxonomy

Recommended naming:
- `Quant Studio Development`
- `Quant Studio Monitoring`

That makes it obvious these are sibling applications in one product family rather than one application with blurred scope.
