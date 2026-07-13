import unittest
import tempfile
import json
from pathlib import Path

from moodripple.ai import MoodAI
from moodripple.routing import is_direct_friend_origin, private_origin, private_origin_candidates
from moodripple.service import MoodService, affection_delta, select_proactive_targets
from moodripple.store import StateStore


class FakeAI:
    def __init__(self):
        self.prompts = []

    async def json(self, prompt):
        self.prompts.append(prompt)
        if "独立个体的近况写作者" in prompt:
            return {
                "description": "我把同一句个人签名改成了俏皮和安静两版，来回预览后反而更难决定该留下哪一版。",
                "delta": 12,
                "topic_intent": "如果是你，会保留哪一个版本？",
                "proactive_seed": "我刚把个人签名改成了俏皮和安静两版，来回预览后反而选不出了。换你来选，会留哪一版？",
                "self_centered": True,
                "recipient_assumption": False,
            }
        return {"labels": ["清醒", "期待"]}

    def default_persona_context(self):
        return {"name": "测试人格", "prompt": "用温柔但直接的方式观察生活。"}

    async def sample_anonymous_chat_references(self, users):
        return ["匿名互动倾向"]


class AffectionCurveTests(unittest.TestCase):
    def test_relationship_change_slows_near_boundary(self):
        middle = affection_delta(10, 0, 1.0, 0.75)
        edge = affection_delta(10, 95, 1.0, 0.75)
        self.assertGreater(middle, edge)
        self.assertGreater(edge, 0)


class ProactiveProbabilityTests(unittest.TestCase):
    def test_probability_is_rolled_independently_for_every_eligible_user(self):
        rolls = iter([0.1, 0.8, 0.49, 0.5])
        selected = select_proactive_targets(
            ["10001", "10002", "10003", "10004"],
            0.5,
            lambda: next(rolls),
        )
        self.assertEqual(selected, ["10001", "10003"])

    def test_probability_boundaries_apply_to_the_whole_eligible_list(self):
        users = ["10001", "10002", "10003"]
        self.assertEqual(select_proactive_targets(users, 0, lambda: 0.1), [])
        self.assertEqual(select_proactive_targets(users, 1, lambda: 0.99), users)


class RoutingTests(unittest.TestCase):
    def test_group_origin_is_not_reused_for_private_message(self):
        self.assertEqual(
            private_origin("123", "default:GroupMessage:456", "default:FriendMessage:{qq}"),
            "default:FriendMessage:123",
        )
        self.assertFalse(is_direct_friend_origin("default:GroupMessage:456", "123"))

    def test_private_origin_is_preserved_and_loaded_platforms_are_fallbacks(self):
        origin = private_origin("123", "onebot:FriendMessage:123", "default:FriendMessage:{qq}")
        self.assertEqual(origin, "onebot:FriendMessage:123")
        self.assertEqual(
            private_origin_candidates(origin, "123", ["onebot", "backup", "onebot"]),
            ["onebot:FriendMessage:123", "backup:FriendMessage:123"],
        )


class ReferenceAI(MoodAI):
    def __init__(self, context):
        super().__init__(context, {})
        self.prompts = []

    async def json(self, prompt):
        self.prompts.append(prompt)
        return {"summaries": ["匿名的轻松互动倾向"]}


class JsonResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_accepts_a_fenced_object_from_provider(self):
        class Provider:
            async def text_chat(self, **kwargs):
                return type("Response", (), {"completion_text": "```json\n{\"labels\": [\"期待\"]}\n```"})()

        context = type("Context", (), {"get_using_provider": lambda self: Provider()})()
        result = await MoodAI(context, {}).json("测试")
        self.assertEqual(result, {"labels": ["期待"]})


class ConversationReferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_conversation_reference_is_anonymized_before_event_creation(self):
        history = json.dumps([{"role": "user", "content": "一段不应被复述的私人文字"}])
        conversation = type("Conversation", (), {"history": history})()

        class Manager:
            async def get_curr_conversation_id(self, origin):
                return "conversation-id"

            async def get_conversation(self, origin, conversation_id):
                return conversation

        context = type("Context", (), {"conversation_manager": Manager()})()
        ai = ReferenceAI(context)
        references = await ai.sample_anonymous_chat_references({"1": {"last_origin": "qq:private:1", "affection": 80}})
        self.assertEqual(references, ["匿名的轻松互动倾向"])
        self.assertIn("严禁复述", ai.prompts[0])


class DebugServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_debug_override_clamps_mood_and_label_refresh_updates_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json", 0)
            await store.load()
            ai = FakeAI()
            service = MoodService(store, ai, {"max_emotion_labels": 4})
            self.assertEqual(await service.set_mood_for_debug(500), 100)
            self.assertEqual(await service.refresh_labels(), ["清醒", "期待"])
            state = await store.snapshot()
            self.assertEqual(state["mood"], 100)
            self.assertEqual(state["labels"], ["清醒", "期待"])

    async def test_event_refreshes_labels_after_its_mood_change(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json", 0)
            await store.load()
            ai = FakeAI()
            service = MoodService(store, ai, {"max_emotion_labels": 4, "significant_change_threshold": 12})
            labels = await service.apply_event({"summary": "一次测试事件", "delta": 20})
            self.assertEqual(labels, ["清醒", "期待"])
            self.assertIn("'mood': 20", ai.prompts[-1])
            self.assertEqual((await service.dashboard())["topics"], 0)

    async def test_debug_affection_is_clamped_and_relationship_is_available(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json", 0)
            await store.load()
            service = MoodService(store, FakeAI(), {})
            self.assertEqual(await service.set_affection_for_debug("123", -500), -100)
            affection, relationship = await service.user_debug_profile("123")
            self.assertEqual(affection, -100)
            self.assertTrue(relationship)

    async def test_dashboard_tracks_event_topics_and_proactive_replies(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json", 0)
            await store.load()
            service = MoodService(store, FakeAI(), {})
            event = {"summary": "一件可聊的事", "topic_intent": "问问对方会怎么想"}
            await service.queue_event_topic(event)
            await service.record_proactive_result("123", event)
            await service.record_seen("123", "qq:private:123")
            dashboard = await service.dashboard()
            self.assertEqual(dashboard["topics"], 1)
            self.assertEqual(dashboard["proactive_sent"], 1)
            self.assertEqual(dashboard["proactive_replies"], 1)
            self.assertGreater(await service.proactive_weight("123"), await service.proactive_weight("456"))

    async def test_event_prompt_contains_default_persona_and_requires_concrete_event(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json", 0)
            await store.load()
            ai = FakeAI()
            service = MoodService(store, ai, {})
            await store.mutate(lambda state: state["events"].append({"at": "2026-07-11T20:00:00+08:00", "type": "daily_event", "summary": "上一件事件"}))
            generated = await service.create_event()
            self.assertIn("个人签名", generated["summary"])
            self.assertIn("换你来选", generated["proactive_seed"])
            self.assertIn("测试人格", ai.prompts[-1])
            self.assertIn("独立个体", ai.prompts[-1])
            self.assertIn("bot_persona", ai.prompts[-1])
            self.assertIn("话题引子", ai.prompts[-1])
            self.assertIn("线上社交场景", ai.prompts[-1])
            self.assertIn("明确触发物", ai.prompts[-1])
            self.assertIn("意外的转折", ai.prompts[-1])
            self.assertIn("proactive_seed", ai.prompts[-1])
            self.assertIn("唯一行动核心", ai.prompts[-1])
            self.assertIn("禁止虚构任何第三方", ai.prompts[-1])
            self.assertIn("尚未选定收件人", ai.prompts[-1])
            self.assertIn("self_centered", ai.prompts[-1])
            self.assertIn("stranger_reference_samples", ai.prompts[-1])
            self.assertIn("上一件事件", ai.prompts[-1])
            self.assertIn("current_mood", ai.prompts[-1])

    async def test_event_is_rejected_when_model_does_not_confirm_self_centered_boundaries(self):
        class BoundaryBreakingAI(FakeAI):
            async def json(self, prompt):
                result = await super().json(prompt)
                if "独立个体的近况写作者" in prompt:
                    result["self_centered"] = False
                return result

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json", 0)
            await store.load()
            service = MoodService(store, BoundaryBreakingAI(), {})
            self.assertIsNone(await service.create_event())
