"""Self-contained checks for whole-phase consolidation in schema v10."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from event_memory.phase_end_collapse import PhaseEndCollapseManager, PhaseEndCollapseStore


def node(identifier, children, *, title=None, summary=None, closed=False, end=None):
    child_key = "event_ids" if identifier.startswith("PH") else "phase_ids"
    value = {
        "id": identifier,
        "title": title or identifier,
        "summary": summary or identifier,
        child_key: list(children),
        "status": "closed" if closed else "open",
        "start": 1.0,
        "end": float(end if end is not None else len(children) * 2),
    }
    if closed:
        value.update(closed_at=float(end if end is not None else len(children) * 2),
                     close_reason="test", close_trigger_id=identifier)
    return value


class PhaseEndCollapseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.frames = []
        for i in range(1, 7):
            path = Path(self.temp.name) / f"f{i}.jpg"
            path.write_bytes(f"image-{i}".encode())
            self.frames.append({"id": f"f{i}", "time": float(i), "path": str(path),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    def store(self):
        store = PhaseEndCollapseStore()
        store.write(
            now=4.0,
            frames=self.frames[:4],
            events=[
                {"id": "E1", "caption": "第一小事件", "frame_ids": ["f1", "f2"], "start": 1.0, "end": 2.0},
                {"id": "E2", "caption": "第二小事件", "frame_ids": ["f3", "f4"], "start": 3.0, "end": 4.0},
            ],
            phases=[node("PH1", ["E1", "E2"], end=4.0)],
            episodes=[node("P1", ["PH1"], end=4.0)],
        )
        return store

    @staticmethod
    def decision(members=None, support=None):
        return {"phase_id": "PH1", "event_ids": members or ["E1", "E2"],
                "caption": "按观察顺序完成第一段和第二段动作。",
                "phase_summary": "完成这一阶段。",
                "support_frame_ids": support or ["f1", "f4"]}

    def close(self, store, now=5.0):
        phase = deepcopy(store.snapshot()["phases"]["PH1"])
        phase.update(status="closed", closed_at=now, close_reason="test", close_trigger_id="PH1")
        store.write(now=now, phases=[phase])

    def test_open_phase_accumulates_then_closed_phase_replaces_all_events(self):
        store = self.store()
        store.write(
            now=6.0, frames=self.frames[4:],
            events=[{"id": "E3", "caption": "第三小事件", "frame_ids": ["f5", "f6"], "start": 5.0, "end": 6.0}],
            phases=[node("PH1", ["E1", "E2", "E3"], end=6.0)],
            episodes=[node("P1", ["PH1"], end=6.0)],
        )
        self.assertEqual(store.snapshot()["phases"]["PH1"]["event_ids"], ["E1", "E2", "E3"])
        self.assertEqual(store.reviewable_phases(), [])
        self.close(store, 7.0)
        replacement = store.collapse_phase(self.decision(["E1", "E2", "E3"], ["f1", "f6"]), 7.0)
        state = store.snapshot()
        self.assertEqual(replacement, "E4")
        self.assertEqual(set(state["events"]), {"E4"})
        self.assertEqual(state["events"]["E4"]["frame_ids"], ["f1", "f2", "f3", "f4", "f5", "f6"])
        self.assertEqual(state["phases"]["PH1"]["event_ids"], ["E4"])
        self.assertEqual(state["completed_phase_ids"], ["PH1"])
        self.assertNotIn("第一小事件", json.dumps(state, ensure_ascii=False))
        with self.assertRaises(ValueError):
            store.write(now=8.0, events=[{"id": "E1", "caption": "不能恢复", "frame_ids": ["f1"], "start": 1.0, "end": 1.0}])

    def test_invalid_partial_or_unsupported_decision_rolls_back(self):
        store = self.store()
        self.close(store)
        before = store.snapshot()
        for changes in (
            {"event_ids": ["E1"]},
            {"event_ids": ["E2", "E1"]},
            {"support_frame_ids": ["unknown"]},
            {"phase_summary": "长" * 81},
            {"source_events": []},
        ):
            decision = self.decision()
            decision.update(changes)
            with self.assertRaises(ValueError):
                store.collapse_phase(decision, 5.0)
            self.assertEqual(store.snapshot(), before)

    def test_singleton_still_requires_one_close_transaction(self):
        store = PhaseEndCollapseStore()
        store.write(
            now=2.0, frames=self.frames[:2],
            events=[{"id": "E1", "caption": "一段观察", "frame_ids": ["f1", "f2"], "start": 1.0, "end": 2.0}],
            phases=[node("PH1", ["E1"], closed=True, end=2.0)],
            episodes=[node("P1", ["PH1"], end=2.0)],
        )
        self.assertEqual(store.reviewable_phases(), ["PH1"])
        self.assertEqual(store.collapse_phase(self.decision(["E1"], ["f1"]), 2.0), "E2")
        self.assertEqual(store.reviewable_phases(), [])

    def test_manager_sees_every_small_event_only_after_phase_close(self):
        store = self.store()
        seen = []

        class Backend:
            calls = []
            def generate(_, system, prompt, evidence, **kwargs):
                payload = json.loads(prompt)
                seen.append((payload, [frame["id"] for frame in evidence]))
                return json.dumps(PhaseEndCollapseTests.decision())

        manager = PhaseEndCollapseManager(Backend(), image_limit=2)
        self.assertEqual(manager.collapse_closed(store, 4.0), [])
        self.assertEqual(seen, [])
        self.close(store, 5.0)
        result = manager.collapse_closed(store, 5.0)
        self.assertEqual([item["event_id"] for item in result], ["E3"])
        self.assertEqual([event["id"] for event in seen[0][0]["events"]], ["E1", "E2"])
        self.assertEqual(seen[0][1], ["f1", "f4"])
        self.assertEqual(set(store.snapshot()["events"]), {"E3"})

    def test_manager_rejects_support_that_was_not_displayed(self):
        store = self.store()
        self.close(store)

        class Backend:
            calls = []
            def generate(_, *args, **kwargs):
                return json.dumps(PhaseEndCollapseTests.decision(support=["f2"]))

        before = store.snapshot()
        with self.assertRaisesRegex(ValueError, "actually displayed"):
            PhaseEndCollapseManager(Backend(), image_limit=2).collapse_closed(store, 5.0)
        self.assertEqual(store.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
