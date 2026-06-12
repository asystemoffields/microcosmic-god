from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .individuals import Individual


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, limit: int):
        self.checkpoint_dir = checkpoint_dir
        self.limit = limit
        self.saved = 0
        self.saved_reasons: dict[str, int] = {}
        self.saved_buckets: dict[str, int] = {}
        self.bucket_limits = self._bucket_limits(limit)

    @staticmethod
    def _bucket_limits(limit: int) -> dict[str, int]:
        if limit <= 0:
            return {}
        return {
            "interval_champion": max(1, min(limit, math.ceil(limit * 0.12))),
            "spawn_champion": max(1, min(limit, math.ceil(limit * 0.25))),
            "tap_champion": max(1, min(limit, math.ceil(limit * 0.18))),
            "learner_champion": max(1, min(limit, math.ceil(limit * 0.12))),
            "line_founder": max(1, min(limit, math.ceil(limit * 0.18))),
            "notable_death": max(1, min(limit, math.ceil(limit * 0.16))),
            "general": limit,
        }

    def _bucket_has_room(self, bucket: str) -> bool:
        return self.saved_buckets.get(bucket, 0) < self.bucket_limits.get(bucket, self.limit)

    def save_controller(
        self,
        tick: int,
        individual: Individual,
        reason: str,
        context: dict[str, Any],
        bucket: str = "general",
        score: float | None = None,
    ) -> bool:
        if self.saved >= self.limit or individual.controller is None or not self._bucket_has_room(bucket):
            return False
        self.saved += 1
        self.saved_reasons[reason] = self.saved_reasons.get(reason, 0) + 1
        self.saved_buckets[bucket] = self.saved_buckets.get(bucket, 0) + 1
        filename = f"controller_t{tick:08d}_o{individual.id}_{reason.replace(' ', '_')}.json"
        path = self.checkpoint_dir / filename
        payload = {
            "tick": tick,
            "reason": reason,
            "bucket": bucket,
            "score": None if score is None else round(score, 6),
            "individual": individual.to_summary(),
            "params": individual.params.to_dict(),
            "controller": individual.controller.to_dict(include_state=True),
            "controller_template": individual.controller_template.to_dict(include_state=False) if individual.controller_template else None,
            "cognition": individual.cognitive_snapshot(),
            "signal_values": [round(v, 6) for v in individual.signal_values],
            "context": context,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return True

    def to_summary(self) -> dict[str, Any]:
        return {
            "saved": self.saved,
            "limit": self.limit,
            "reasons": dict(self.saved_reasons),
            "buckets": dict(self.saved_buckets),
            "bucket_limits": dict(self.bucket_limits),
        }
