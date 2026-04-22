"""Launches the Quant Studio Monitoring Streamlit app."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    app_path = project_root / "app" / "streamlit_app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.maxUploadSize",
        "51200",
        "--server.maxMessageSize",
        "51200",
    ]
    raise SystemExit(subprocess.run(command, cwd=project_root).returncode)


if __name__ == "__main__":
    main()
