from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Intervention:
    tick: int
    kind: str
    payload: dict[str, Any]
    reason: str = ""


def load_interventions(path: str | None) -> dict[int, list[Intervention]]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    by_tick: dict[int, list[Intervention]] = {}
    for item in data:
        intervention = Intervention(
            tick=int(item["tick"]),
            kind=str(item["kind"]),
            payload=dict(item.get("payload", {})),
            reason=str(item.get("reason", "")),
        )
        by_tick.setdefault(intervention.tick, []).append(intervention)
    return by_tick

