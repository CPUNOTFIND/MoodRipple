import tempfile
import unittest
from pathlib import Path

from moodripple.store import StateStore, clamp


class StateStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_mutation_is_persisted_and_clamped_on_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            store = StateStore(path, 0)
            await store.load()
            await store.mutate(lambda state: state.update({"mood": 999, "labels": ["期待"]}))
            reloaded = StateStore(path, 0)
            await reloaded.load()
            state = await reloaded.snapshot()
            self.assertEqual(state["mood"], 100)
            self.assertEqual(state["labels"], ["期待"])

    async def test_invalid_file_recovers_to_default_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text("not json", encoding="utf-8")
            store = StateStore(path, -8)
            await store.load()
            self.assertEqual((await store.snapshot())["mood"], -8)


class ClampTests(unittest.TestCase):
    def test_clamp_bounds(self):
        self.assertEqual(clamp(-500), -100)
        self.assertEqual(clamp(500), 100)
