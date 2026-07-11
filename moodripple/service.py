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
        flirtiness = min(3, max(0, int(self.config.get("event_flirtiness", 1))))
        recent_events = [
            {"at": item.get("at", ""), "summary": item.get("summary", "")}
            for item in reversed(state["events"])
            if item.get("type") == "daily_event"
        ][:5]
        chat_references = await self.ai.sample_anonymous_chat_references(state["users"])
        result = await self.ai.json(
            "你是一个独立个体的近况写作者。角色是有自身生活感与社交习惯的个体，不要突出或提及机器人、模型、算法。"
            "输入中的 bot_persona 是 AstrBot 当前实际使用的人格设定，包含人格名称与系统提示词；"
            "它是本次创作的最高角色约束，事件的视角、用词、关注点和情绪反应必须服从它。"
            "根据当前时间、心情、氛围和参考信息，为角色创造一件刚刚发生的具体小事。"
            "它必须是主动找用户聊天的强话题引子，像一句说到一半的话，诱使对方自然追问或分享。"
            "优先发生在具体的线上社交场景，例如某个对话框、未发送的语音草稿、游戏房间、链接、图片或消息通知；"
            "现实场景也可使用，但必须能自然成为一次线上聊天的开场。"
            "使用第一人称、符合人格的口吻，写成带细节的微日记。必须交代正在何处、碰到了什么具体对象、做了什么动作、"
            "发生了什么变化或留下什么未完结果；若去掉这些细节后可套用到任何一天，就说明事件不合格。"
            "禁止信息海、数据流、算法波动等虚无缥缈的拟物化描写，也不要只说‘忽然有点怎样’。"
            "情绪可以是开心、寂寞、吃醋、悸动、疲惫、顽皮或想使坏等，但必须合理源自当前心情。"
            "可按配置有优雅留白的暧昧氛围，不得出现露骨性描写、性行为、裸露、性胁迫，"
            "也不得假定用户年龄、关系或同意。"
            "recent_events 按从新到旧排列，第一条必须是最主要的时间线参考；不得暴露用户信息、原话或身份。"
            "chat_references 是从随机抽取的一到两段对话脱敏得到的备选灵感，高好感用户被抽中的概率更高；"
            "只能借鉴其抽象互动倾向，绝不可复述其中任何内容。"
            "topic_intent 必须说明如何基于该事件自然开启开放话题。只输出 JSON，不要 markdown："
            '{"description": "第一人称内心事件，80到160字", "delta": number(-15..15), '
            '"topic_intent": "最多60字"}。输入：'
            + str({
                "time": now_iso(),
                "current_mood": state["mood"],
                "tags": state["labels"],
                "bot_persona": persona,
                "flirtiness": flirtiness,
                "recent_events": recent_events,
                "anonymous_atmosphere": atmosphere,
                "chat_references": chat_references,
            })
        )
        if not result:
            return None
        try:
            delta = int(result.get("delta", 0))
        except (TypeError, ValueError):
            return None
        description = str(result.get("description", "")).strip()
        if not description:
            return None
        return {"summary": description[:400], "delta": max(-15, min(15, delta)), "topic_intent": str(result.get("topic_intent", ""))[:120]}

    async def apply_event(self, event: dict[str, Any]) -> list[str] | None:
        """Apply the AI-generated event delta, then refresh labels from the result."""
        await self.apply_assessment("__world__", {
            "mood_delta": event.get("delta", 0), "affection_delta": 0,
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
