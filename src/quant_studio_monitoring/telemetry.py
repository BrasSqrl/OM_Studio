"""Run-stage telemetry for monitoring execution."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter


@dataclass(slots=True)
class RunEvent:
    stage: str
    status: str
    started_at_utc: str
    duration_seconds: float
    detail: str = ""


@dataclass(slots=True)
class RunTelemetry:
    events: list[RunEvent] = field(default_factory=list)

    @contextmanager
    def stage(self, stage: str, *, detail: str = "") -> Iterator[None]:
        started_at = datetime.now(UTC)
        started = perf_counter()
        status = "completed"
        try:
            yield
        except Exception:
            status = "failed"
            raise
        finally:
            self.events.append(
                RunEvent(
                    stage=stage,
                    status=status,
                    started_at_utc=started_at.isoformat(),
                    duration_seconds=round(perf_counter() - started, 6),
                    detail=detail,
                )
            )

    def record(self, stage: str, *, status: str = "completed", detail: str = "") -> None:
        self.events.append(
            RunEvent(
                stage=stage,
                status=status,
                started_at_utc=datetime.now(UTC).isoformat(),
                duration_seconds=0.0,
                detail=detail,
            )
        )

    def write_jsonl(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(asdict(event), default=str) + "\n")
        return path
