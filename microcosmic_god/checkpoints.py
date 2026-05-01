from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .organisms import Organism


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, limit: int):
        self.checkpoint_dir = checkpoint_dir
        self.limit = limit
        self.saved = 0
        self.saved_reasons: dict[str, int] = {}
        self._saved_tool_affordances: set[str] = set()

    def save_brain(self, tick: int, organism: Organism, reason: str, context: dict[str, Any]) -> bool:
        if self.saved >= self.limit or organism.brain is None:
            return False
        self.saved += 1
        self.saved_reasons[reason] = self.saved_reasons.get(reason, 0) + 1
        filename = f"brain_t{tick:08d}_o{organism.id}_{reason.replace(' ', '_')}.json"
        path = self.checkpoint_dir / filename
        payload = {
            "tick": tick,
            "reason": reason,
            "organism": organism.to_summary(),
            "genome": organism.genome.to_dict(),
            "brain": organism.brain.to_dict(include_state=True),
            "brain_template": organism.brain_template.to_dict(include_state=False) if organism.brain_template else None,
            "inventory": dict(organism.inventory),
            "tool_skill": {k: round(v, 6) for k, v in organism.tool_skill.items()},
            "signal_values": [round(v, 6) for v in organism.signal_values],
            "context": context,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return True

    def save_first_tool(self, tick: int, organism: Organism, affordance: str, context: dict[str, Any]) -> bool:
        if affordance in self._saved_tool_affordances:
            return False
        self._saved_tool_affordances.add(affordance)
        return self.save_brain(tick, organism, f"first_{affordance}_tool_success", context)

    def to_summary(self) -> dict[str, Any]:
        return {
            "saved": self.saved,
            "limit": self.limit,
            "reasons": dict(self.saved_reasons),
            "first_tool_affordances": sorted(self._saved_tool_affordances),
        }

