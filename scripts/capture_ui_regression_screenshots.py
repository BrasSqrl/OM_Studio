"""Capture baseline screenshots for a running OM Studio Streamlit app.

Usage:
    python scripts/capture_ui_regression_screenshots.py --base-url http://localhost:8501

The script keeps Playwright optional. If it is not installed, it still writes the
checklist JSON so the UI states can be reviewed manually.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_studio_monitoring.ui_regression import (
    UI_REGRESSION_SCENARIOS,
    write_ui_regression_checklist,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8501")
    parser.add_argument("--output-dir", default="ui_regression_artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checklist_path = write_ui_regression_checklist(output_dir / "ui_regression_checklist.json")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(f"Playwright is not installed. Wrote checklist to {checklist_path}.")
        return 0

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        for scenario in UI_REGRESSION_SCENARIOS:
            page.goto(args.base_url, wait_until="networkidle")
            page.screenshot(
                path=output_dir / f"{scenario.scenario_id}.png",
                full_page=True,
            )
        browser.close()

    print(f"Wrote UI regression checklist and screenshots to {output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
