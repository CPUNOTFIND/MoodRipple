"""Concurrency-safe JSON persistence for MoodRipple runtime state."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def clamp(value: float, lower: float = -100, upper: float = 100) -> float:
    return max(lower, min(upper, value))


class StateStore:
    """Keep all mutable plugin state in one atomically replaced JSON document."""

    def __init__(self, path: Path, initial_mood: int) -> None:
        self.path = path
        self.initial_mood = int(clamp(initial_mood))
        self._lock = asyncio.Lock()
        self._state: dict[str, Any] = self._empty_state()

    def _empty_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "mood": self.initial_mood,
            "labels": [],
            "last_decay_date": "",
            "events": [],
            "event_schedule": {},
            "daily_stats": {},
            "topic_queue": [],
            "proactive_records": [],
            "milestones": [],
            "users": {},
            "groups": {},
            "journals": [],
        }

    async def load(self) -> None:
        async with self._lock:
            self._state = await asyncio.to_thread(self._read)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_state()
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("state root must be an object")
        except (OSError, ValueError, json.JSONDecodeError):
            return self._empty_state()
        state = self._empty_state()
        state.update(loaded)
        state["mood"] = int(clamp(float(state.get("mood", self.initial_mood))))
        for key in ("events", "labels", "journals", "topic_queue", "proactive_records", "milestones"):
            if not isinstance(state.get(key), list):
                state[key] = []
        for key in ("users", "groups", "event_schedule", "daily_stats"):
            if not isinstance(state.get(key), dict):
                state[key] = {}
        return state

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return copy.deepcopy(self._state)

    async def mutate(self, change: Callable[[dict[str, Any]], Any]) -> Any:
        """Apply a change and commit it before releasing the state lock."""
        async with self._lock:
            result = change(self._state)
            await asyncio.to_thread(self._write, self._state)
            return result

    def _write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)
