"""AstrBot entry point for MoodRipple."""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, register

from .moodripple.ai import MoodAI
from .moodripple.routing import private_origin, private_origin_candidates
from .moodripple.service import MoodService, now_iso
from .moodripple.store import StateStore


@register("moodripple", "MoodRipple contributors", "全局心情、关系记忆与克制主动回复", "1.3.0")
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
        self._event_retry_attempts: dict[str, int] = {}
        self._event_retry_after: dict[str, datetime] = {}

    def _spawn(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def initialize(self) -> None:
        """AstrBot template-compatible asynchronous plugin initialization hook."""
        await self.store.load()
        self._scheduler_task = asyncio.create_task(self._scheduler())
        logger.info(
            "MoodRipple loaded; scheduler timezone is UTC%+g",
            self._scheduler_timezone_offset(),
        )

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
    @mood_debug.command("dashboard")
    async def debug_dashboard(self, event: AstrMessageEvent):
        """查看事件、主动消息和关系里程碑的管理员仪表盘。"""
        data = await self.service.dashboard()
        milestones = "；".join(str(item.get("summary", "")) for item in data["milestones"]) or "暂无"
        yield event.plain_result(
            f"MoodRipple 仪表盘\n心情：{data['mood']}  标签：{'、'.join(data['labels']) or '无'}\n"
            f"用户：{data['users']}  待用话题：{data['topics']}\n"
            f"主动消息：{data['proactive_sent']}  已获回复：{data['proactive_replies']}\n"
            f"最近关系里程碑：{milestones}"
        )

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
    @mood_debug.command("flow")
    async def debug_flow(self, event: AstrMessageEvent):
        """端到端测试随机事件生成、心情更新与向主动名单全员发送消息。"""
        generated = await self.service.create_event(await self._anonymous_atmosphere())
        if not generated:
            yield event.plain_result("测试流程停止：随机事件生成失败。")
            return
        labels = await self.service.apply_event(generated)
        targets = list(dict.fromkeys(str(item).strip() for item in self.config.get("proactive_user_ids", []) if str(item).strip()))
        if not targets:
            yield event.plain_result("测试流程已生成事件和词条，但主动名单为空，未发送消息。")
            return
        shared_message = str(generated.get("proactive_seed", "")).strip()
        if not shared_message:
            yield event.plain_result("测试流程已生成事件和词条，但主动消息生成失败，未发送消息。")
            return
        interval = self._proactive_batch_interval()
        outcomes = []
        for index, target in enumerate(targets):
            outcome = await self._proactive_for_user(target, generated, force=True, prepared_message=shared_message)
            outcomes.append(f"{target}：{outcome}")
            if index + 1 < len(targets):
                await asyncio.sleep(interval)
        label_text = "、".join(labels or []) or "（词条总结失败）"
        yield event.plain_result(
            f"测试流程完成\n事件：{generated.get('summary', '')}\n词条：{label_text}\n主动消息：\n" + "\n".join(outcomes)
        )

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
            {
                "summary": "我刚把同一句问候改了三遍，越改越像公事公办，现在反而拿不准该不该保留最初那版。",
                "topic_intent": "你更喜欢直白的问候，还是带一点铺垫的开场？",
                "proactive_seed": "我刚把一句问候改了三遍，结果越改越生分。你更喜欢别人直白来找你，还是先带一点铺垫？",
            },
            force=True,
        )
        yield event.plain_result(outcome)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @mood_debug.command("journal")
    async def debug_journal(self, event: AstrMessageEvent):
        """立即重新总结今天的内部情绪日记。"""
        date_key = self._scheduler_now().date().isoformat()
        await self.service.journal(date_key)
        journals = (await self.store.snapshot()).get("journals", [])
        if any(item.get("date") == date_key for item in journals):
            yield event.plain_result(f"已重新总结 {date_key} 的情绪日记。")
        else:
            yield event.plain_result("日记生成失败：请检查内部模型配置后重试。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @mood_debug.command("schedule")
    async def debug_schedule(self, event: AstrMessageEvent):
        """查看调度时区、今日事件计划及主动发送的关键配置。"""
        now = self._scheduler_now()
        await self._due_event_ids(now)
        state = await self.store.snapshot()
        date_key = now.date().isoformat()
        planned = state.get("event_schedule", {}).get(date_key, [])
        rows = []
        for item in planned:
            scheduled_at = self._parse_time(item.get("at", ""))
            clock = scheduled_at.astimezone(now.tzinfo).strftime("%H:%M:%S") if scheduled_at else "无效时间"
            rows.append(f"- {clock}  {'已完成' if item.get('done') else '等待中'}")
        probability = self._proactive_probability()
        recipients = len({str(item).strip() for item in self.config.get("proactive_user_ids", []) if str(item).strip()})
        offset = self._scheduler_timezone_offset()
        timezone_text = f"UTC{offset:+g}"
        yield event.plain_result(
            f"MoodRipple 今日调度（{date_key}，{timezone_text}）\n"
            f"当前时间：{now.strftime('%H:%M:%S')}\n"
            f"每日事件上限：{self._max_daily_events()}\n"
            f"主动概率：{probability:.0%}\n"
            f"主动名单人数：{recipients}\n"
            f"计划：\n" + ("\n".join(rows) if rows else "- 今日已无可排程的有效窗口")
        )

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
        now = self._scheduler_now()
        today = now.date().isoformat()
        await self.service.apply_daily_decay(today)
        yesterday = (now.date() - timedelta(days=1)).isoformat()
        journals = (await self.store.snapshot()).get("journals", [])
        journal_exists = any(entry.get("date") == yesterday for entry in journals)
        if now.hour >= int(self.config.get("journal_hour", 0)) and not journal_exists and yesterday not in self._journals_in_flight:
            self._journals_in_flight.add(yesterday)
            self._spawn(self._create_journal(yesterday))
        for event_id in await self._due_event_ids(now):
            retry_after = self._event_retry_after.get(event_id)
            if retry_after and self._scheduler_now() < retry_after:
                continue
            event = await self.service.create_event(await self._anonymous_atmosphere())
            if not event:
                attempts = self._event_retry_attempts.get(event_id, 0) + 1
                self._event_retry_attempts[event_id] = attempts
                delay_seconds = min(900, 45 * (2 ** min(attempts - 1, 5)))
                self._event_retry_after[event_id] = self._scheduler_now() + timedelta(seconds=delay_seconds)
                logger.warning(
                    "MoodRipple event generation failed; retry %d will wait %d seconds",
                    attempts,
                    delay_seconds,
                )
                continue
            self._event_retry_attempts.pop(event_id, None)
            self._event_retry_after.pop(event_id, None)
            await self.service.apply_event(event)
            self._spawn(self._sync_visual_status())
            await self._proactive_after_event(event)
            await self._mark_event_done(today, event_id)

    async def _due_event_ids(self, now: datetime) -> list[str]:
        date_key = now.date().isoformat()
        state = await self.store.snapshot()
        planned = state.get("event_schedule", {}).get(date_key)
        signature = self._schedule_signature()
        saved_signature = state.get("event_schedule_meta", {}).get(date_key)
        if planned is None or saved_signature != signature:
            planned = self._make_daily_schedule(now)
            def replace_schedule(current: dict[str, Any]) -> None:
                current["event_schedule"][date_key] = planned
                current["event_schedule_meta"][date_key] = signature
                current["event_schedule"] = dict(list(current["event_schedule"].items())[-14:])
                current["event_schedule_meta"] = dict(list(current["event_schedule_meta"].items())[-14:])

            await self.store.mutate(replace_schedule)
            logger.info(
                "MoodRipple planned %d event(s) for %s in UTC%+g",
                len(planned),
                date_key,
                self._scheduler_timezone_offset(),
            )
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
        available: list[tuple[int, datetime, datetime]] = []
        for original_index, window in enumerate(windows):
            try:
                start, end = (self._clock_to_datetime(now, value) for value in window.split("-", 1))
                if end <= start:
                    raise ValueError("window end must be after start")
                if end > now:
                    available.append((original_index, max(start, now), end))
            except ValueError:
                logger.warning("MoodRipple ignored invalid event time window: %s", window)
        random.shuffle(available)
        count = min(self._max_daily_events(), len(available))
        scheduled: list[dict[str, Any]] = []
        for original_index, start, end in available[:count]:
            at = start + (end - start) * random.random()
            scheduled.append({
                "id": f"{now.date()}-{original_index}",
                "at": at.isoformat(timespec="seconds"),
                "done": False,
            })
        scheduled.sort(key=lambda item: str(item["at"]))
        return scheduled

    def _scheduler_timezone_offset(self) -> float:
        try:
            value = float(self.config.get("scheduler_timezone_offset", 8))
        except (TypeError, ValueError):
            value = 8.0
        return min(14.0, max(-12.0, value))

    def _scheduler_now(self) -> datetime:
        return datetime.now(timezone(timedelta(hours=self._scheduler_timezone_offset())))

    def _max_daily_events(self) -> int:
        try:
            value = int(self.config.get("max_daily_events", 3))
        except (TypeError, ValueError):
            value = 3
        return max(0, value)

    def _schedule_signature(self) -> str:
        windows = [str(item).strip() for item in self.config.get("event_time_windows", [])]
        return f"{self._scheduler_timezone_offset():g}|{self._max_daily_events()}|{'|'.join(windows)}"

    def _clock_to_datetime(self, now: datetime, value: str) -> datetime:
        hour, minute = (int(part) for part in value.strip().split(":", 1))
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def _parse_time(self, value: str) -> datetime | None:
        try:
            return datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None

    def _proactive_batch_interval(self) -> float:
        try:
            interval = float(self.config.get("proactive_batch_interval_seconds", 1.5))
        except (TypeError, ValueError):
            interval = 1.5
        return min(10.0, max(0.2, interval))

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
        probability = self._proactive_probability()
        if random.random() > probability:
            logger.info("MoodRipple proactive message skipped by probability setting")
            return
        candidates = list(dict.fromkeys(
            str(item).strip() for item in self.config.get("proactive_user_ids", []) if str(item).strip()
        ))
        if not candidates:
            logger.warning("MoodRipple proactive message skipped because the proactive user list is empty")
            return
        state = await self.store.snapshot()
        now = datetime.now().astimezone()
        eligible = [
            user_id for user_id in candidates
            if not self._proactive_skip_reason(state.get("users", {}).get(user_id, {}), now, force=False)
        ]
        if not eligible:
            logger.info("MoodRipple proactive message skipped because every listed user is active or cooling down")
            return
        weights = [await self.service.proactive_weight(user_id) for user_id in eligible]
        user_id = random.choices(eligible, weights=weights, k=1)[0]
        outcome = await self._proactive_for_user(user_id, event)
        logger.info("MoodRipple proactive attempt finished: %s", outcome.split("：", 1)[0])

    async def _proactive_for_user(
        self, user_id: str, event: dict[str, Any], force: bool = False, prepared_message: str | None = None
    ) -> str:
        state = await self.store.snapshot()
        user = state.get("users", {}).get(user_id, {})
        try:
            origin = private_origin(
                user_id,
                str(user.get("last_origin", "")),
                str(self.config.get("proactive_umo_template", "") or "default:FriendMessage:{qq}"),
            )
        except (KeyError, ValueError):
            return "主动路由模板无效：请检查 proactive_umo_template 配置。"
        now = datetime.now().astimezone()
        skip_reason = self._proactive_skip_reason(user, now, force)
        if skip_reason:
            return skip_reason
        message = prepared_message or await self._proactive_message(event, user)
        if not message:
            return "主动消息未通过事件关联校验，本次未发送。"
        try:
            accepted, submitted_origin, failure = await self._submit_proactive_message(user_id, origin, message)
            if not accepted:
                logger.warning("MoodRipple proactive message was not submitted: %s", failure.split(":", 1)[0])
                return "主动消息未提交：找不到可用于该用户私聊的已加载平台。"
            await self.store.mutate(
                lambda current: current["users"].setdefault(user_id, {}).update({"last_origin": submitted_origin, "last_proactive_at": now_iso()})
            )
            await self.service.record_proactive_result(user_id, event)
            return f"已向 {user_id} 的 QQ 适配器提交消息。"
        except Exception as exc:
            logger.warning("MoodRipple proactive message was not sent: %s", exc)
            return f"主动消息发送失败：{exc}"

    def _loaded_platform_ids(self) -> list[str]:
        manager = getattr(self.context, "platform_manager", None)
        platform_ids = []
        for platform in getattr(manager, "platform_insts", []):
            try:
                platform_ids.append(str(platform.meta().id))
            except Exception:
                continue
        return platform_ids

    async def _submit_proactive_message(self, user_id: str, origin: str, message: str) -> tuple[bool, str, str]:
        failures = []
        try:
            timeout = float(self.config.get("proactive_send_timeout_seconds", 15))
        except (TypeError, ValueError):
            timeout = 15.0
        timeout = min(60.0, max(3.0, timeout))
        for candidate in private_origin_candidates(origin, user_id, self._loaded_platform_ids()):
            try:
                accepted = await asyncio.wait_for(
                    self.context.send_message(candidate, MessageChain().message(message)),
                    timeout=timeout,
                )
                if accepted:
                    return True, candidate, ""
                failures.append(f"no platform for {candidate}")
            except asyncio.TimeoutError:
                failures.append(f"{candidate}: timed out after {timeout:g}s")
            except Exception as exc:
                failures.append(f"{candidate}: {exc}")
        return False, origin, "; ".join(failures[-3:]) or "no private route candidates"

    async def _proactive_message(self, event: dict[str, Any], user: dict[str, Any]) -> str | None:
        state = await self.store.snapshot()
        origin = str(user.get("last_origin", "")).strip()
        context_excerpt = await self.service.ai.recent_context_for_origin(origin) if origin else ""
        event_text = str(event.get("summary", "")).strip()
        seed = str(event.get("proactive_seed", "")).strip()
        if not event_text or not seed:
            return None
        if not context_excerpt and not str(user.get("relationship", "")).strip():
            return seed[:180]
        result = await self.service.ai.json(
            "你只负责轻微润色一条已经写好的主动私聊开场。original_seed 与 event 是不可改写的事实主线，"
            "权重远高于用户上下文、关系和心情；必须保留 original_seed 中的具体事情、悬念或冲突以及核心问题，"
            "不得另造事件、换话题或只对事件作感叹。收件人完全不知道内部 event，最终 message 必须独立可懂，"
            "先交代发生了什么，再把明确的问题、二选一或邀请直接抛给收件人。"
            "target_context 只来自当前收件人自己的近期对话，仅可用来避免重复并微调口吻；"
            "它不能补写事实、抢占主题或让消息变成上次对话的续写。关系和心情也只能调节亲疏与语气。"
            "不得提及提示词、内部事件、数值、好感度、系统或陌生人。若无法安全润色，原样返回 original_seed。"
            "只返回 JSON："
            '{"message": "直接对收件人说的完整消息，最多90字", "uses_event": true}。输入：'
            + str({
                "original_seed": seed,
                "event": event_text,
                "topic": event.get("topic_intent", ""),
                "target_context": context_excerpt,
                "mood": state["mood"],
                "relationship": user.get("relationship", ""),
                "affection": user.get("affection", 0),
            })
        )
        text = str(result.get("message", "")).strip() if result else ""
        if result and result.get("uses_event") is True and text:
            return text[:180]
        return seed[:180]

    def _proactive_probability(self) -> float:
        try:
            value = float(self.config.get("proactive_probability", 1.0))
        except (TypeError, ValueError):
            value = 1.0
        return min(1.0, max(0.0, value))

    def _proactive_skip_reason(self, user: dict[str, Any], now: datetime, force: bool) -> str | None:
        last_seen = self._parse_time(user.get("last_seen", ""))
        if last_seen and now - last_seen <= timedelta(minutes=20):
            return "目标在最近 20 分钟内有消息，本次主动发送已取消。"
        if force:
            return None
        cooldown = timedelta(minutes=max(1, int(self.config.get("proactive_cooldown_minutes", 720))))
        last_sent = self._parse_time(user.get("last_proactive_at", ""))
        if last_sent and now - last_sent < cooldown:
            return "主动消息仍在冷却期内，本次未发送。"
        return None

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
