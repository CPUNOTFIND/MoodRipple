import unittest
import tempfile
from pathlib import Path

from moodripple.service import MoodService, affection_delta
from moodripple.store import StateStore


class FakeAI:
    def __init__(self):
        self.prompts = []

    async def json(self, prompt):
        self.prompts.append(prompt)
        return {"labels": ["清醒", "期待"]}

    def default_persona_context(self):
        return {"name": "测试人格", "prompt": "用温柔但直接的方式观察生活。"}


class AffectionCurveTests(unittest.TestCase):
    def test_relationship_change_slows_near_boundary(self):
        middle = affection_delta(10, 0, 1.0, 0.75)
        edge = affection_delta(10, 95, 1.0, 0.75)
        self.assertGreater(middle, edge)
        self.assertGreater(edge, 0)


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
            labels = await service.apply_event({"summary": "一次测试事件", "mood_delta": 20})
            self.assertEqual(labels, ["清醒", "期待"])
            self.assertIn("'mood': 20", ai.prompts[-1])

    async def test_debug_affection_is_clamped_and_relationship_is_available(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json", 0)
            await store.load()
            service = MoodService(store, FakeAI(), {})
            self.assertEqual(await service.set_affection_for_debug("123", -500), -100)
            affection, relationship = await service.user_debug_profile("123")
            self.assertEqual(affection, -100)
            self.assertTrue(relationship)

    async def test_event_prompt_contains_default_persona_and_requires_concrete_event(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json", 0)
            await store.load()
            ai = FakeAI()
            service = MoodService(store, ai, {})
            await service.create_event()
            self.assertIn("测试人格", ai.prompts[-1])
            self.assertIn("具体", ai.prompts[-1])
