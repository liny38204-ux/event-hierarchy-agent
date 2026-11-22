import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from event_memory.buffered_phase import BufferedPhaseStore
from event_memory.text_retrieval import CaptionRetriever


class TableEmbedder:
    """Small deterministic embedder: test scoring without a model download."""
    def __init__(self, vectors, query_vectors):
        self.vectors = vectors
        self.query_vectors = query_vectors

    def encode_documents(self, texts):
        return [self.vectors[text] for text in texts]

    def encode_query(self, query):
        return self.query_vectors[query]


def node(identifier, title, summary, child_key, children, events_or_phases):
    return {
        "id": identifier,
        "title": title,
        "summary": summary,
        child_key: children,
        "status": "open",
        "start": min(events_or_phases[item]["start"] for item in children),
        "end": max(events_or_phases[item]["end"] for item in children),
    }


class TextRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def memory(self, captions=("whole alpha description\nline beta detail", "other gamma")):
        store = BufferedPhaseStore()
        frames, events = [], []
        for index, caption in enumerate(captions, start=1):
            path = Path(self.tmp.name) / f"f{index}.jpg"
            path.write_bytes(f"image-{index}".encode())
            frame = {"id": f"f{index}", "time": float(index), "path": str(path),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            frames.append(frame)
            events.append({"id": f"E{index}", "caption": caption,
                           "frame_ids": [frame["id"]], "start": float(index),
                           "end": float(index)})
        store.write(now=float(len(events)), frames=frames, events=events)
        if len(events) >= 2:
            state = store.snapshot()
            phase = node("PH1", "局部阶段", "阶段短摘要", "event_ids", ["E1"], state["events"])
            episode = node("P1", "整体过程", "大事件短摘要", "phase_ids", ["PH1"], {"PH1": phase})
            store.write(now=float(len(events)), phases=[phase], episodes=[episode])
        return store.snapshot()

    def retriever(self, memory, vectors, query_vectors):
        return CaptionRetriever(memory, TableEmbedder(vectors, query_vectors))

    def test_caption_score_can_win_over_every_line(self):
        memory = self.memory()
        vectors = {
            "whole alpha description\nline beta detail": [1, 0],
            "whole alpha description": [0, 1],
            "line beta detail": [0, -1],
            "other gamma": [-1, 0],
        }
        hit = self.retriever(memory, vectors, {"q": [1, 0]}).search("q")[0]
        self.assertEqual(hit["event_id"], "E1")
        self.assertEqual(hit["matched"], {"kind": "caption", "line_index": None,
                                           "text": "whole alpha description\nline beta detail", "score": 1.0})
        self.assertEqual(hit["caption_score"], 1.0)
        self.assertEqual(hit["line_score"], 0.0)

    def test_best_line_can_win_over_whole_caption(self):
        memory = self.memory()
        vectors = {
            "whole alpha description\nline beta detail": [0, 1],
            "whole alpha description": [0, -1],
            "line beta detail": [1, 0],
            "other gamma": [-1, 0],
        }
        hit = self.retriever(memory, vectors, {"q": [1, 0]}).search("q")[0]
        self.assertEqual(hit["event_id"], "E1")
        self.assertEqual(hit["matched"], {"kind": "line", "line_index": 1,
                                           "text": "line beta detail", "score": 1.0})
        self.assertEqual(hit["score"], hit["line_score"])

    def test_bare_headers_are_not_embedded_and_line_hits_report_their_window(self):
        caption = "<1s - 1s>\n摸牌。\n<2s - 2s>\n放牌。"
        # Deliberately no vector for either time header.
        vectors = {caption: [0, 1], "摸牌。": [-1, 0], "放牌。": [1, 0]}
        hit = self.retriever(self.memory((caption,)), vectors, {"q": [1, 0]}).search("q")[0]
        self.assertEqual(hit["matched"]["text"], "放牌。")
        self.assertEqual(hit["matched"]["line_index"], 3)
        self.assertEqual(hit["matched"]["time_range"], [2, 2])
        self.assertEqual(hit["caption"], caption)

    def test_negative_similarities_are_not_replaced_by_zero(self):
        memory = self.memory(("bad whole\nbad line",))
        vectors = {"bad whole\nbad line": [-1, 0], "bad whole": [-1, 0], "bad line": [-1, 0]}
        hit = self.retriever(memory, vectors, {"q": [1, 0]}).search("q")[0]
        self.assertEqual(hit["caption_score"], -1.0)
        self.assertEqual(hit["line_score"], -1.0)
        self.assertEqual(hit["score"], -1.0)
        self.assertEqual(hit["matched"]["kind"], "caption")

    def test_event_is_returned_once_with_original_multiline_caption_and_parent_chain(self):
        memory = self.memory()
        vectors = {
            "whole alpha description\nline beta detail": [0, 1],
            "whole alpha description": [1, 0],
            "line beta detail": [1, 0],
            "other gamma": [-1, 0],
        }
        hits = self.retriever(memory, vectors, {"q": [1, 0]}).search("q", top_k=2)
        self.assertEqual([hit["event_id"] for hit in hits], ["E1", "E2"])
        first = hits[0]
        self.assertEqual(first["caption"], "whole alpha description\nline beta detail")
        self.assertEqual(first["phase"], {"id": "PH1", "title": "局部阶段", "summary": "阶段短摘要"})
        self.assertEqual(first["episode"], {"id": "P1", "title": "整体过程", "summary": "大事件短摘要"})
        root = hits[1]
        self.assertIsNone(root["phase"])
        self.assertIsNone(root["episode"])

    def test_top_k_bounds_and_snapshot_is_immune_to_caller_mutation(self):
        memory = self.memory()
        vectors = {
            "whole alpha description\nline beta detail": [1, 0],
            "whole alpha description": [1, 0],
            "line beta detail": [1, 0],
            "other gamma": [0, 1],
        }
        retriever = self.retriever(memory, vectors, {"q": [1, 0]})
        memory["events"]["E1"]["caption"] = "caller changed this"
        self.assertEqual(len(retriever.search("q", top_k=99)), 2)
        self.assertEqual(retriever.search("q")[0]["caption"], "whole alpha description\nline beta detail")
        with self.assertRaises(ValueError):
            retriever.search("q", top_k=0)
        with self.assertRaises(ValueError):
            retriever.search("q", top_k=True)

    def test_rejects_bad_vectors_and_future_summary_for_early_query_time(self):
        memory = self.memory()
        vectors = {
            "whole alpha description\nline beta detail": [1, 0],
            "whole alpha description": [1, 0],
            "line beta detail": [1, 0],
            "other gamma": [0, 1],
        }
        retriever = self.retriever(memory, vectors, {"q": [1, 0]})
        with self.assertRaises(ValueError):
            retriever.search("q", query_time=1.0)
        with self.assertRaises(ValueError):
            self.retriever(memory, {**vectors, "other gamma": [1, 2, 3]}, {"q": [1, 0]})
        with self.assertRaises(ValueError):
            self.retriever(memory, vectors, {"q": [0, 0]}).search("q")


if __name__ == "__main__":
    unittest.main()
