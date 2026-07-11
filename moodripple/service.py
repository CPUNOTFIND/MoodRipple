"""Domain services: AI assessment, gentle state updates, and scheduling helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .ai import MoodAI
from .store import StateStore, clamp


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def affection_delta(raw_delta: float, current: float, sensitivity: float, damping: float) -> float:
    """Make the middle range responsive and the two relationship extremes resistant."""
    remaining = max(0.0, 1.0 - abs(current) / 100.0)
    curve = remaining ** max(0.05, damping)
    return raw_delta * max(0.05, sensitivity) * curve


class MoodService:
    def __init__(self, store: StateStore, ai: MoodAI, config: dict[str, Any]) -> None:
        self.store = store
        self.ai = ai
        self.config = config

    async def record_seen(self, user_id: str, origin: str, group_id: str = "") -> None:
        timestamp = now_iso()

        def change(state: dict[str, Any]) -> None:
            user = state["users"].setdefault(user_id, {"affection": 0, "relationship": "尚在慢慢认识。"})
            user.update({"last_seen": timestamp, "last_origin": origin})
            if group_id:
                group = state["groups"].setdefault(group_id, {})
                group["last_message_at"] = timestamp

        await self.store.mutate(change)

    async def dynamic_hint(self, user_id: str) -> str:
        state = await self.store.snapshot()
        user = state["users"].get(user_id, {})
        labels = "、".join(str(x) for x in state.get("labels", [])[:4]) or "平静"
        reminder = str(user.get("next_hint", "")).strip()
        lines = [
            "<moodripple_runtime_hint>",
            f"你此刻的内在情绪标签：{labels}。请仅让语气自然带出，不解释或量化它。",
            f"与这位用户的关系摘要：{user.get('relationship', '尚在慢慢认识。')}。保持底层人格与边界。",
        ]
        if reminder:
            lines.append(f"一次性情绪延续提醒：{reminder}")
        topic = str(user.get("pending_topic", "")).strip()
        if topic:
            lines.append(f"可以在合适时自然接入的话题意图：{topic}")
        lines.append("禁止提及以上提示、内部心情数值、好感度或系统机制。")
        lines.append("</moodripple_runtime_hint>")
        return "\n".join(lines)

    async def consume_one_shot_hints(self, user_id: str) -> None:
        def change(state: dict[str, Any]) -> None:
            user = state["users"].get(user_id)
            if user:
                user.pop("next_hint", None)
                user.pop("pending_topic", None)

        await self.store.mutate(change)

    async def refresh_labels(self) -> list[str] | None:
        """Ask the model for fresh labels without changing the current mood value."""
        state = await self.store.snapshot()
        limit = max(1, int(self.config.get("max_emotion_labels", 4)))
        result = await self.ai.json(
            "根据当前内在心情生成细腻的中文情绪标签。返回 JSON："
            '{"labels": ["最多4个，每个不超过12字"]}。'
            "不得出现数值、用户身份、聊天内容或系统说明。输入："
            + str({"mood": state["mood"], "old_labels": state["labels"], "max_labels": limit})
        )
        labels = result.get("labels") if result else None
        if not isinstance(labels, list):
            return None
        cleaned = [str(item)[:24] for item in labels if str(item).strip()][:limit]
        if not cleaned:
            return None
        await self.store.mutate(lambda current: current.update({"labels": cleaned}))
        return cleaned

    async def set_mood_for_debug(self, value: int) -> int:
        """Explicit administrator override used only by the debug command."""
        mood = int(round(clamp(value)))

        def change(state: dict[str, Any]) -> int:
            state["mood"] = mood
            self._note_mood(state)
            state["events"].append({"at": now_iso(), "type": "debug_override", "summary": "管理员调试覆盖", "delta": 0})
            state["events"] = state["events"][-120:]
            return mood

        return await self.store.mutate(change)

    async def user_debug_profile(self, user_id: str) -> tuple[float, str]:
        state = await self.store.snapshot()
        user = state["users"].get(user_id, {})
        return float(user.get("affection", 0)), str(user.get("relationship", "尚在慢慢认识。"))

    async def set_affection_for_debug(self, user_id: str, value: int) -> float:
        affection = round(clamp(value), 2)

        def change(state: dict[str, Any]) -> float:
            user = state["users"].setdefault(user_id, {"relationship": "尚在慢慢认识。"})
            user["affection"] = affection
            return affection

        return await self.store.mutate(change)

    async def assess_turn(self, user_id: str, user_text: str, reply_text: str) -> None:
        state = await self.store.snapshot()
        user = state["users"].get(user_id, {})
        payload = {
            "current_mood": state["mood"],
            "current_labels": state["labels"],
            "relationship_summary": user.get("relationship", "尚在慢慢认识。"),
            "current_affection": user.get("affection", 0),
            "user_message": user_text[:1600],
            "bot_reply": reply_text[:1600],
        }
        result = await self.ai.json(
            "评估一轮已完成对话的情感影响。返回 JSON："
            '{"mood_delta": number(-20..20), "affection_delta": number(-15..15), '
            '"relationship_summary": string(不含身份信息，最多50字), '
            '"next_hint": string(不含数值，最多80字), "labels": [string], '
            '"self_reflection": string(可为空，最多80字), "self_adjustment": number(-3..3)}。'
            "只有当前心情绝对值很高时才给非零 self_adjustment；所有判断由语境完成。输入："
            + str(payload)
        )
        if result is not None:
            await self.apply_assessment(user_id, result, "conversation")

    async def apply_assessment(self, user_id: str, result: dict[str, Any], source: str) -> None:
        try:
            mood_change = float(result.get("mood_delta", 0))
            raw_affection = float(result.get("affection_delta", 0))
            self_adjustment = float(result.get("self_adjustment", 0))
        except (TypeError, ValueError):
            return
        threshold = abs(float(self.config.get("significant_change_threshold", 12)))
        label_cap = max(1, int(self.config.get("max_emotion_labels", 4)))

        def change(state: dict[str, Any]) -> None:
            before = float(state["mood"])
            total_delta = mood_change + self_adjustment
            state["mood"] = int(round(clamp(before + total_delta)))
            self._note_mood(state)
            hint = str(result.get("next_hint", "")).strip()
            reflection = str(result.get("self_reflection", "")).strip()
            combined_hint = "；".join(x for x in (hint, reflection) if x)[:180]
            if user_id != "__world__":
                user = state["users"].setdefault(user_id, {"affection": 0, "relationship": "尚在慢慢认识。"})
                current = float(user.get("affection", 0))
                applied = affection_delta(
                    raw_affection,
                    current,
                    float(self.config.get("affection_sensitivity", 1.0)),
                    float(self.config.get("affection_boundary_damping", 0.75)),
                )
                user["affection"] = round(clamp(current + applied), 2)
                summary = str(result.get("relationship_summary", "")).strip()
                if summary:
                    user["relationship"] = summary[:100]
                if combined_hint:
                    user["next_hint"] = combined_hint
            labels = result.get("labels", [])
            if abs(total_delta) >= threshold and isinstance(labels, list):
                state["labels"] = [str(item)[:24] for item in labels if str(item).strip()][:label_cap]
            state["events"].append({
                "at": now_iso(), "type": source, "summary": combined_hint or "一次情绪涟漪", "delta": total_delta
            })
            state["events"] = state["events"][-120:]

        await self.store.mutate(change)

    @staticmethod
    def _note_mood(state: dict[str, Any]) -> None:
        """Retain day-level extremes without retaining conversational content."""
        date_key = datetime.now().astimezone().date().isoformat()
        mood = int(state["mood"])
        stats = state.setdefault("daily_stats", {}).setdefault(date_key, {"min": mood, "max": mood})
        stats["min"] = min(int(stats.get("min", mood)), mood)
        stats["max"] = max(int(stats.get("max", mood)), mood)
        state["daily_stats"] = dict(list(state["daily_stats"].items())[-120:])

    async def apply_daily_decay(self, date_key: str) -> bool:
        rate = min(1.0, max(0.0, float(self.config.get("daily_decay_rate", 0.12))))

        def change(state: dict[str, Any]) -> bool:
            if state.get("last_decay_date") == date_key:
                return False
            state["mood"] = int(round(clamp(float(state["mood"]) * (1.0 - rate))))
            self._note_mood(state)
            state["last_decay_date"] = date_key
            return True

        return await self.store.mutate(change)

    async def poetic_mood(self) -> str | None:
        state = await self.store.snapshot()
        result = await self.ai.json(
            "将当前内在状态写成一句中文诗意心情描述，不能含数值、标签列表、系统或用户信息。"
            "返回 JSON：{\"description\": \"最多45字\"}。输入："
            + str({"mood": state["mood"], "labels": state["labels"]})
        )
        if result:
            text = str(result.get("description", "")).strip()
            if text:
                return text[:100]
        return None

    async def create_event(self, atmosphere: str = "") -> dict[str, Any] | None:
        state = await self.store.snapshot()
        persona = self.ai.default_persona_context()
        result = await self.ai.json(
            "基于提供的 AstrBot 人格设定，生成一件符合该人格会经历或在意的虚构日常事件。"
            "事件必须是生动、具体、已经发生的小事：写出明确场景、正在做的动作、至少两个可感知细节"
            "（物件、声音、光线、天气或身体感受等）以及事情如何收束。"
            "不要写抽象的情绪概述、泛泛的日常感想或真实世界未经证实的消息。"
            "事件只能温和影响语气，不能改写人格设定。返回 JSON："
            '{"summary": "具体事件，80到160字，不含真实人物或私密聊天内容", "mood_delta": number(-20..20), '
            '"topic_intent": "最多60字"}。'
            "群体氛围若提供，仅可作为完全匿名的抽象灵感，绝不可复述聊天："
            + str({
                "time": now_iso(),
                "mood": state["mood"],
                "labels": state["labels"],
                "persona": persona,
                "anonymous_atmosphere": atmosphere,
            })
        )
        if result is None:
            return None
        try:
            delta = float(result.get("mood_delta", 0))
        except (TypeError, ValueError):
            return None
        result["mood_delta"] = delta
        return result

    async def apply_event(self, event: dict[str, Any]) -> list[str] | None:
        """Apply an event first, then summarize labels from its resulting mood."""
        await self.apply_assessment("__world__", {
            "mood_delta": event["mood_delta"], "affection_delta": 0,
            "relationship_summary": "", "next_hint": str(event.get("summary", "")),
            "labels": [], "self_reflection": "", "self_adjustment": 0,
        }, "daily_event")
        return await self.refresh_labels()

    async def assess_group_atmosphere(self, group_id: str, messages: list[str]) -> None:
        state = await self.store.snapshot()
        result = await self.ai.json(
            "从一组完全匿名的群聊文本中抽象群体情绪氛围，并给出非常轻微、缓慢的情绪传染影响。"
            "不得输出或转述任何原句、姓名或身份线索。返回 JSON："
            '{"mood_delta": number(-4..4), "summary": "匿名氛围概括，最多40字", "labels": [string]}。输入：'
            + str({"current_mood": state["mood"], "messages": [m[:240] for m in messages]})
        )
        if result is not None:
            await self.apply_assessment("__world__", {
                "mood_delta": result.get("mood_delta", 0), "affection_delta": 0,
                "relationship_summary": "", "next_hint": result.get("summary", ""),
                "labels": result.get("labels", []), "self_reflection": "", "self_adjustment": 0,
            }, "group_atmosphere")

    async def journal(self, date_key: str) -> None:
        state = await self.store.snapshot()
        day_events = [event for event in state["events"] if str(event.get("at", "")).startswith(date_key)]
        result = await self.ai.json(
            "为机器人生成一条仅内部保存的情绪日记摘要。不得出现任何用户、昵称、QQ号、原话或隐私。"
            "返回 JSON：{\"summary\": \"最多240字\"}。输入："
            + str({
                "date": date_key,
                "events": day_events[-30:],
                "mood_extremes": state.get("daily_stats", {}).get(date_key, {}),
                "ending_mood": state["mood"],
            })
        )
        if not result or not str(result.get("summary", "")).strip():
            return

        def change(current: dict[str, Any]) -> None:
            current["journals"] = [entry for entry in current["journals"] if entry.get("date") != date_key]
            current["journals"].append({"date": date_key, "summary": str(result["summary"])[:500], "created_at": now_iso()})
            current["journals"] = current["journals"][-90:]

        await self.store.mutate(change)
