"""AstrBot entry point for MoodRipple."""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, register

from .moodripple.ai import MoodAI
from .moodripple.service import MoodService, now_iso
from .moodripple.store import StateStore


@register("moodripple", "MoodRipple contributors", "全局心情、关系记忆与克制主动回复", "1.0.11")
class MoodRipplePlugin(Star):
    """A non-invasive emotional layer; it never replaces the configured persona."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config: dict[str, Any] = dict(config)
        self.store = StateStore(Path("data/plugin_data/moodripple/state.json"), self.config.get("initial_mood", 0))
        self.service = MoodService(self.store, MoodAI(context, self.config), self.config)
        self._tasks: set[asyncio.Task[Any]] = set()
        self._journals_in_flight: set[str] = set()
        self._group_buffers: dict[str, list[str]] = {}
        self._status_sync_unavailable_logged = False
        self._scheduler_task: asyncio.Task[Any] | None = None

    def _spawn(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def initialize(self) -> None:
        """AstrBot template-compatible asynchronous plugin initialization hook."""
        await self.store.load()
        self._scheduler_task = asyncio.create_task(self._scheduler())
        logger.info("MoodRipple loaded; state persistence and scheduler are ready")

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def observe_message(self, event: AstrMessageEvent) -> None:
        """Store only routing/activity data; group text is short-lived and anonymous."""
        user_id = str(event.get_sender_id())
        group_id = str(getattr(event.message_obj, "group_id", "") or "")
        await self.service.record_seen(user_id, event.unified_msg_origin, group_id)
        if group_id:
            await self._collect_group_message(group_id, event.message_str)

    @filter.on_llm_request()
    async def inject_mood_hint(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        user_id = str(event.get_sender_id())
        try:
            from astrbot.core.agent.message import TextPart

            req.extra_user_content_parts.append(TextPart(text=await self.service.dynamic_hint(user_id)).mark_as_temp())
            await self.service.consume_one_shot_hints(user_id)
        except Exception as exc:
            logger.warning("MoodRipple context injection skipped: %s", exc)

    @filter.on_llm_response()
    async def assess_after_reply(self, event: AstrMessageEvent, response: LLMResponse) -> None:
        user_id = str(event.get_sender_id())
        group_id = str(getattr(event.message_obj, "group_id", "") or "")
        if group_id:
            await self._mark_group_active(group_id)
        text = str(getattr(response, "completion_text", "") or "")
        if text:
            self._spawn(self.service.assess_turn(user_id, event.message_str, text))

    @filter.command("mood", alias={"心情"})
    async def mood(self, event: AstrMessageEvent):
        """返回一句不含数值的诗意心情描述。"""
        description = await self.service.poetic_mood()
        yield event.plain_result(description or "心绪像一盏尚未点亮的灯，安静等着风经过。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("moodjournal", alias={"情绪日记"})
    async def mood_journal(self, event: AstrMessageEvent):
        """管理员查看最近一条仅内部保存的情绪日记。"""
        journals = (await self.store.snapshot()).get("journals", [])
        if not journals:
            yield event.plain_result("还没有生成情绪日记。")
            return
        latest = journals[-1]
        yield event.plain_result(f"{latest.get('date', '')} 的情绪日记：\n{latest.get('summary', '')}")

    @filter.command_group("mooddebug")
    def mood_debug(self):
        """仅管理员可用的 MoodRipple 调试指令组。"""
        pass

    @filter.permission_type(filter.PermissionType.ADMIN)
    @mood_debug.command("state")
    async def debug_state(self, event: AstrMessageEvent):
        """查看当前内部心情值、标签和已记录用户数量。"""
        state = await self.store.snapshot()
        labels = "、".join(str(item) for item in state.get("labels", [])) or "（无）"
        yield event.plain_result(f"MoodRipple 调试状态\n心情：{state['mood']}\n标签：{labels}\n用户记录：{len(state['users'])}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @mood_debug.command("labels")
    async def debug_labels(self, event: AstrMessageEvent):
        """立即通过 AI 重新生成当前心情标签。"""
        labels = await self.service.refresh_labels()
        if labels:
            yield event.plain_result("已刷新心情词条：" + "、".join(labels))
        else:
            yield event.plain_result("词条生成失败：请检查内部模型配置后重试。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @mood_debug.command("event")
    async def debug_event(self, event: AstrMessageEvent):
        """立刻生成、应用并在事后总结一条随机情绪事件。"""
        generated = await self.service.create_event(await self._anonymous_atmosphere())
        if not generated:
            yield event.plain_result("事件生成失败：请检查内部模型配置后重试。")
            return
        labels = await self.service.apply_event(generated)
        label_text = "、".join(labels or []) or "（词条总结失败）"
        yield event.plain_result(f"已生成事件：{generated.get('summary', '')}\n事件后词条：{label_text}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @mood_debug.command("set")
    async def debug_set_mood(self, event: AstrMessageEvent, value: int):
        """设置调试心情值，数值会自动限制到 -100 至 100。"""
        mood = await self.service.set_mood_for_debug(value)
        yield event.plain_result(f"调试心情已设为：{mood}。可再执行 /mooddebug labels 刷新词条。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @mood_debug.command("affection")
    async def debug_affection(self, event: AstrMessageEvent, user_id: str):
        """查询指定 QQ 号的内部好感度。"""
        affection, _ = await self.service.user_debug_profile(user_id)
        yield event.plain_result(f"用户 {user_id} 的好感度：{affection}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @mood_debug.command("setaffection")
    async def debug_set_affection(self, event: AstrMessageEvent, user_id: str, value: int):
        """设置指定 QQ 号的调试好感度，数值自动限制到 -100 至 100。"""
        affection = await self.service.set_affection_for_debug(user_id, value)
        yield event.plain_result(f"用户 {user_id} 的好感度已设为：{affection}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @mood_debug.command("relation")
    async def debug_relation(self, event: AstrMessageEvent, user_id: str):
        """查询指定 QQ 号的关系描述。"""
        _, relationship = await self.service.user_debug_profile(user_id)
        yield event.plain_result(f"用户 {user_id} 的关系描述：{relationship}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @mood_debug.command("proactive")
    async def debug_proactive(self, event: AstrMessageEvent, user_id: str):
        """立刻向指定 QQ 号发起一条调试主动消息。"""
        outcome = await self._proactive_for_user(
            user_id,
            {"summary": "管理员正在测试一次自然的主动问候。", "topic_intent": "轻松地开启一段对话"},
            force=True,
        )
        yield event.plain_result(outcome)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @mood_debug.command("journal")
    async def debug_journal(self, event: AstrMessageEvent):
        """立即重新总结今天的内部情绪日记。"""
        date_key = datetime.now().astimezone().date().isoformat()
        await self.service.journal(date_key)
        journals = (await self.store.snapshot()).get("journals", [])
        if any(item.get("date") == date_key for item in journals):
            yield event.plain_result(f"已重新总结 {date_key} 的情绪日记。")
        else:
            yield event.plain_result("日记生成失败：请检查内部模型配置后重试。")

    async def _collect_group_message(self, group_id: str, text: str) -> None:
        if not self.config.get("group_atmosphere_enabled", True) or not text.strip():
            return
        whitelist = {str(x) for x in self.config.get("group_whitelist", [])}
        if whitelist and group_id not in whitelist:
            return
        state = await self.store.snapshot()
        group = state.get("groups", {}).get(group_id, {})
        last_active = self._parse_time(group.get("last_bot_active_at", ""))
        active_for = timedelta(minutes=max(1, int(self.config.get("group_active_minutes", 20))))
        if last_active is None or datetime.now().astimezone() - last_active > active_for:
            return
        batch_size = max(2, int(self.config.get("group_batch_size", 8)))

        buffer = self._group_buffers.setdefault(group_id, [])
        buffer.append(text[:500])
        if len(buffer) >= batch_size:
            ready = buffer[:batch_size]
            del buffer[:batch_size]
            self._spawn(self.service.assess_group_atmosphere(group_id, ready))

    async def _mark_group_active(self, group_id: str) -> None:
        await self.store.mutate(lambda state: state["groups"].setdefault(group_id, {}).update({"last_bot_active_at": now_iso()}))

    async def _scheduler(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("MoodRipple scheduler tick failed: %s", exc)
            await asyncio.sleep(45)

    async def _tick(self) -> None:
        now = datetime.now().astimezone()
        today = now.date().isoformat()
        await self.service.apply_daily_decay(today)
        yesterday = (now.date() - timedelta(days=1)).isoformat()
        journals = (await self.store.snapshot()).get("journals", [])
        journal_exists = any(entry.get("date") == yesterday for entry in journals)
        if now.hour >= int(self.config.get("journal_hour", 0)) and not journal_exists and yesterday not in self._journals_in_flight:
            self._journals_in_flight.add(yesterday)
            self._spawn(self._create_journal(yesterday))
        for event_id in await self._due_event_ids(now):
            event = await self.service.create_event(await self._anonymous_atmosphere())
            if event:
                await self.service.apply_event(event)
                self._spawn(self._sync_visual_status())
                await self._proactive_after_event(event)
            await self._mark_event_done(today, event_id)

    async def _due_event_ids(self, now: datetime) -> list[str]:
        date_key = now.date().isoformat()
        state = await self.store.snapshot()
        planned = state.get("event_schedule", {}).get(date_key)
        if planned is None:
            planned = self._make_daily_schedule(now)
            await self.store.mutate(lambda current: current["event_schedule"].update({date_key: planned}))
        due: list[str] = []
        for item in planned:
            scheduled_at = self._parse_time(item.get("at", ""))
            if not item.get("done") and scheduled_at and now >= scheduled_at:
                due.append(str(item["id"]))
        return due

    async def _create_journal(self, date_key: str) -> None:
        try:
            await self.service.journal(date_key)
        finally:
            self._journals_in_flight.discard(date_key)

    def _make_daily_schedule(self, now: datetime) -> list[dict[str, Any]]:
        windows = [str(x) for x in self.config.get("event_time_windows", [])]
        random.shuffle(windows)
        count = min(max(0, int(self.config.get("max_daily_events", 2))), len(windows))
        scheduled: list[dict[str, Any]] = []
        for index, window in enumerate(windows[:count]):
            try:
                start, end = (self._clock_to_datetime(now, value) for value in window.split("-", 1))
                if end <= start:
                    continue
                at = start + (end - start) * random.random()
                scheduled.append({"id": f"{now.date()}-{index}", "at": at.isoformat(timespec="seconds"), "done": False})
            except ValueError:
                logger.warning("MoodRipple ignored invalid event time window: %s", window)
        return scheduled

    def _clock_to_datetime(self, now: datetime, value: str) -> datetime:
        hour, minute = (int(part) for part in value.strip().split(":", 1))
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def _parse_time(self, value: str) -> datetime | None:
        try:
            return datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None

    async def _mark_event_done(self, date_key: str, event_id: str) -> None:
        def change(state: dict[str, Any]) -> None:
            for item in state.get("event_schedule", {}).get(date_key, []):
                if item.get("id") == event_id:
                    item["done"] = True

        await self.store.mutate(change)

    async def _anonymous_atmosphere(self) -> str:
        if random.random() > float(self.config.get("chat_atmosphere_ratio", 0.35)):
            return ""
        state = await self.store.snapshot()
        summaries = [str(event.get("summary", "")) for event in state.get("events", []) if event.get("type") == "group_atmosphere"]
        return summaries[-1] if summaries else ""

    async def _proactive_after_event(self, event: dict[str, Any]) -> None:
        if random.random() > float(self.config.get("proactive_probability", 0.35)):
            return
        candidates = [str(item) for item in self.config.get("proactive_user_ids", [])]
        if not candidates:
            return
        user_id = random.choice(candidates)
        await self._proactive_for_user(user_id, event)

    async def _proactive_for_user(self, user_id: str, event: dict[str, Any], force: bool = False) -> str:
        state = await self.store.snapshot()
        user = state.get("users", {}).get(user_id)
        if not user or not user.get("last_origin"):
            return "无法主动发送：该 QQ 号尚未与机器人建立可用会话。"
        now = datetime.now().astimezone()
        active_window = timedelta(minutes=max(1, int(self.config.get("session_awareness_minutes", 30))))
        last_seen = self._parse_time(user.get("last_seen", ""))
        if not force and last_seen and now - last_seen <= active_window:
            await self.store.mutate(lambda current: current["users"][user_id].update({"pending_topic": str(event.get("topic_intent", ""))[:100]}))
            return "目标处于近期活跃会话，已缓存话题等待自然引入。"
        cooldown = timedelta(minutes=max(1, int(self.config.get("proactive_cooldown_minutes", 720))))
        last_sent = self._parse_time(user.get("last_proactive_at", ""))
        if not force and last_sent and now - last_sent < cooldown:
            return "主动消息仍在冷却期内，本次未发送。"
        message = await self._proactive_message(event, user)
        if not message:
            return "主动消息生成失败：请检查内部模型配置后重试。"
        try:
            await self.context.send_message(str(user["last_origin"]), MessageChain().message(message))
            await self.store.mutate(lambda current: current["users"][user_id].update({"last_proactive_at": now_iso()}))
            return f"已向 {user_id} 发起主动消息。"
        except Exception as exc:
            logger.warning("MoodRipple proactive message was not sent: %s", exc)
            return f"主动消息发送失败：{exc}"

    async def _proactive_message(self, event: dict[str, Any], user: dict[str, Any]) -> str | None:
        state = await self.store.snapshot()
        result = await self.service.ai.json(
            "生成一条克制、自然、不施压的中文主动消息。消息必须以当前事件为唯一核心，"
            "明确提到事件中的一个独特细节，或由该细节自然抛出 topic_intent；禁止发送脱离事件的泛用问候。"
            "关系与心情只可影响措辞，不可取代事件内容。根据情境选择关怀、分享、轻微求助或话题延续之一。"
            "不得透露内部数值、好感度、用户资料、系统或评估机制。返回 JSON："
            '{"event_anchor": "消息中使用的事件独特细节", "message": "最多90字"}。输入：'
            + str({"event": event.get("summary", ""), "topic": event.get("topic_intent", ""), "mood": state["mood"], "relationship": user.get("relationship", ""), "affection": user.get("affection", 0)})
        )
        text = str(result.get("message", "")).strip() if result else ""
        return text[:180] or None

    async def _sync_visual_status(self) -> None:
        """Best-effort bridge for adapters that explicitly expose a bot-status API."""
        if not self.config.get("enable_qq_status_sync", False):
            return
        description = await self.service.poetic_mood()
        if not description:
            return
        manager = getattr(self.context, "platform_manager", None)
        platforms = getattr(manager, "get_insts", lambda: [])()
        for platform in platforms:
            update_status = getattr(platform, "set_bot_status", None)
            if not callable(update_status):
                continue
            try:
                result = update_status(signature=description)
                if asyncio.iscoroutine(result):
                    await result
                return
            except TypeError:
                try:
                    result = update_status(description)
                    if asyncio.iscoroutine(result):
                        await result
                    return
                except Exception as exc:
                    logger.warning("MoodRipple status sync failed: %s", exc)
            except Exception as exc:
                logger.warning("MoodRipple status sync failed: %s", exc)
        if not self._status_sync_unavailable_logged:
            logger.warning("MoodRipple status sync enabled, but no loaded QQ adapter exposes set_bot_status")
            self._status_sync_unavailable_logged = True

    async def terminate(self) -> None:
        if self._scheduler_task:
            self._scheduler_task.cancel()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*(task for task in [self._scheduler_task, *self._tasks] if task), return_exceptions=True)
