from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from .config import RunConfig


class RunLogger:
    def __init__(self, config: RunConfig):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = config.output_path / f"{stamp}_seed{config.seed}_{config.profile}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.story_path = self.run_dir / "story_events.jsonl"
        self._events: TextIO = self.events_path.open("w", encoding="utf-8")
        self._stories: TextIO = self.story_path.open("w", encoding="utf-8")
        self.write_json("config.json", config.to_dict())

    def write_json(self, name: str, payload: Any) -> None:
        path = self.run_dir / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def event(self, tick: int, kind: str, payload: dict[str, Any]) -> None:
        record = {"tick": tick, "kind": kind, **payload}
        self._events.write(json.dumps(record, sort_keys=True) + "\n")

    def story_event(self, record: dict[str, Any]) -> None:
        self._stories.write(json.dumps(record, sort_keys=True) + "\n")

    def flush(self) -> None:
        self._events.flush()
        self._stories.flush()

    def close(self) -> None:
        self.flush()
        self._events.close()
        self._stories.close()


def json_safe_number(value: float) -> float:
    return round(float(value), 6)
