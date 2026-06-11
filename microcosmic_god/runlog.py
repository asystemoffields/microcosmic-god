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
        # Structural-genealogy stream; opened lazily so runs without
        # structural steps don't grow an empty file.
        self._structures: TextIO | None = None
        self.write_json("config.json", config.to_dict())

    def write_json(self, name: str, payload: Any) -> None:
        path = self.run_dir / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def write_json_atomic(self, name: str, payload: Any) -> None:
        """Replace `name` atomically so a concurrent reader never sees a torn file."""
        path = self.run_dir / name
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp.replace(path)

    def event(self, tick: int, kind: str, payload: dict[str, Any]) -> None:
        record = {"tick": tick, "kind": kind, **payload}
        self._events.write(json.dumps(record, sort_keys=True) + "\n")

    def story_event(self, record: dict[str, Any]) -> None:
        self._stories.write(json.dumps(record, sort_keys=True) + "\n")

    def structure_event(self, record: dict[str, Any]) -> None:
        if self._structures is None:
            self._structures = (self.run_dir / "structure_events.jsonl").open("w", encoding="utf-8")
        self._structures.write(json.dumps(record, sort_keys=True) + "\n")

    def flush(self) -> None:
        if not self._events.closed:
            self._events.flush()
        if not self._stories.closed:
            self._stories.flush()
        if self._structures is not None and not self._structures.closed:
            self._structures.flush()

    def close(self) -> None:
        self.flush()
        if not self._events.closed:
            self._events.close()
        if not self._stories.closed:
            self._stories.close()
        if self._structures is not None and not self._structures.closed:
            self._structures.close()


def json_safe_number(value: float) -> float:
    return round(float(value), 6)
