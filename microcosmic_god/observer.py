from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any

from .runlog import RunLogger


class EventObserver:
    """Promotes rare or consequential events into a compact story stream.

    The observer is intentionally descriptive. It does not feed back into
    organism fitness, action choice, world physics, or evolution operators.
    """

    def __init__(
        self,
        logger: RunLogger,
        *,
        recent_limit: int = 8,
        story_limit: int = 2_000,
        threshold: float = 1.0,
    ):
        self.logger = logger
        self.recent_limit = recent_limit
        self.story_limit = story_limit
        self.threshold = threshold
        self.candidate_counts: Counter[str] = Counter()
        self.promoted_counts: Counter[str] = Counter()
        self.rarity_counts: Counter[str] = Counter()
        self.recent_by_subject: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=recent_limit))
        self.promoted = 0

    def observe(
        self,
        tick: int,
        kind: str,
        payload: dict[str, Any],
        *,
        subjects: list[str] | None = None,
        score: float = 0.0,
        rarity_key: str | None = None,
        force: bool = False,
    ) -> bool:
        self.candidate_counts[kind] += 1
        subjects = subjects or []
        compact = self._compact_event(tick, kind, payload, score)
        for subject in subjects:
            self.recent_by_subject[subject].append(compact)

        rarity_bonus = 0.0
        if rarity_key:
            self.rarity_counts[rarity_key] += 1
            if self.rarity_counts[rarity_key] == 1:
                rarity_bonus = 0.85
            elif self.rarity_counts[rarity_key] <= 3:
                rarity_bonus = 0.30

        interestingness = score + rarity_bonus
        if not force and (interestingness < self.threshold or self.promoted >= self.story_limit):
            return False

        record = {
            "tick": tick,
            "kind": kind,
            "interestingness": round(interestingness, 6),
            "rarity_key": rarity_key,
            "subjects": subjects,
            "payload": self._json_safe(payload),
            "recent_context": {
                subject: list(self.recent_by_subject.get(subject, ()))[:-1]
                for subject in subjects
                if self.recent_by_subject.get(subject)
            },
        }
        self.logger.story_event(record)
        self.promoted += 1
        self.promoted_counts[kind] += 1
        return True

    def force_promote(
        self,
        tick: int,
        kind: str,
        payload: dict[str, Any],
        *,
        subjects: list[str] | None = None,
        score: float = 1.0,
        rarity_key: str | None = None,
    ) -> bool:
        return self.observe(tick, kind, payload, subjects=subjects, score=score, rarity_key=rarity_key, force=True)

    def to_summary(self) -> dict[str, Any]:
        return {
            "candidates": dict(self.candidate_counts),
            "promoted": dict(self.promoted_counts),
            "promoted_total": self.promoted,
            "story_limit": self.story_limit,
            "threshold": self.threshold,
            "tracked_subjects": len(self.recent_by_subject),
        }

    def _compact_event(self, tick: int, kind: str, payload: dict[str, Any], score: float) -> dict[str, Any]:
        compact_payload = {
            key: value
            for key, value in payload.items()
            if key in {
                "organism_id",
                "source_id",
                "child_id",
                "parent_ids",
                "place",
                "affordance",
                "target",
                "gain",
                "clarity",
                "fidelity",
                "token",
                "released",
                "mode",
                "cause",
                "method_quality",
                "problem_kind",
                "lesson_kind",
                "sequence",
                "skill_gain",
                "score",
            }
        }
        return {"tick": tick, "kind": kind, "score": round(score, 6), **self._json_safe(compact_payload)}

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, float):
            return round(value, 6)
        return value
