from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from event_memory import evidence_search


WINDOW_ANSWER = '{"action":"ANSWER","evidence_sufficient":true,"option":"B","thought":"The book cover is red."}'
WINDOW_SEARCH = '{"action":"SEARCH","evidence_sufficient":false,"thought":"The frame shows a shelf; the object color is obscured."}'


def frames():
    return [{"id": f"f{i:06d}", "time": i / 2, "sample_time": i / 2,
             "sha256": str(i) * 64, "path": f"/{i}.jpg"} for i in range(1, 5)]


def question():
    return {"id": 17, "question": "Which object is red?", "options": ["cup", "book", "lamp"],
            "video": "source.mp4", "realtime": 3.0}


class EvidenceSearchTests(unittest.TestCase):
    def setUp(self):
        self.reader = Mock()
        self.reader.backend = Mock()

    def test_merged_only_runner_accepts_short_window_without_a_control_call(self):
        self.reader.backend.generate.return_value = WINDOW_ANSWER
        result, direct = evidence_search.run_question(
            self.reader, question(), {}, frames(), include_direct_control=False)
        self.assertEqual((result["prediction"], result["route"], direct), ("B", "short_window", None))
        self.reader.backend.generate.assert_called_once()
        self.reader.build.assert_not_called()
        self.assertEqual((result["action"], result["thought"]), ("ANSWER", "The book cover is red."))

    def test_window_uses_four_independent_images_and_returns_a_letter(self):
        backend = Mock()
        backend.generate.side_effect = ["A", WINDOW_ANSWER]
        gate, direct = evidence_search.run_window(question(), frames(), backend)
        self.assertEqual((direct["prediction"], gate["prediction"], gate["route"]), ("A", "B", "short_window"))
        self.assertEqual([call.kwargs["stage"] for call in backend.generate.call_args_list],
                         ["evidence-direct:17", "evidence-window:17"])
        self.assertEqual(backend.generate.call_args_list[0].kwargs["evidence"], frames())

    def test_window_search_and_invalid_response_do_not_guess(self):
        backend = Mock()
        backend.generate.side_effect = ["A", WINDOW_SEARCH]
        gate, direct = evidence_search.run_window(question(), frames(), backend)
        self.assertEqual((direct["prediction"], gate["status"], gate["route"]), ("A", "needs_history", "history"))
        backend.generate.side_effect = ["A", "answer: B", '{"action":"ANSWER","option":"B","thought":" "}']
        gate, _ = evidence_search.run_window(question(), frames(), backend)
        self.assertEqual((gate["status"], gate["prediction"]), ("invalid_response", None))

    def test_bare_letter_retries_on_the_same_images_and_preserves_both_outputs(self):
        backend = Mock()
        backend.generate.side_effect = ["B", WINDOW_ANSWER]
        gate, _ = evidence_search.run_window(question(), frames(), backend, include_direct_control=False)
        self.assertEqual((gate["route"], gate["thought"], gate["invalid_first_raw"], gate["raw"]),
                         ("short_window", "The book cover is red.", "B", WINDOW_ANSWER))
        self.assertEqual([c.kwargs["evidence"] for c in backend.generate.call_args_list], [frames(), frames()])

    def test_answer_with_the_printed_option_text_remains_a_direct_answer(self):
        backend = Mock()
        backend.generate.return_value = '{"action":"ANSWER","evidence_sufficient":true,"option":"B. book","thought":"Red book."}'
        gate, _ = evidence_search.run_window(question(), frames(), backend, include_direct_control=False)
        self.assertEqual((gate["prediction"], gate["thought"], gate["route"]), ("B", "Red book.", "short_window"))
        backend.generate.assert_called_once()

    def test_invalid_or_conflicting_decisions_cannot_supply_a_note_to_history(self):
        invalid = ['{"action":"SEARCH","evidence_sufficient":true,"option":"A","thought":"ambiguous"}',
                   '{"action":"ANSWER","evidence_sufficient":true,"option":"D","thought":"outside the options"}',
                   '{"action":"ANSWER","evidence_sufficient":true,"option":"A. book","thought":"letter and text conflict"}',
                   '{"action":"ANSWER","evidence_sufficient":true,"option":"B","thought":5}',
                   '{"action":"ANSWER","option":"B","thought":"missing boolean"}',
                   '{"action":"ANSWER","evidence_sufficient":"false","option":"B","thought":"wrong type"}',
                   '{"action":"ANSWER","evidence_sufficient":1,"option":"B","thought":"wrong type"}', '["ANSWER"]']
        for raw in invalid:
            with self.subTest(raw=raw):
                backend = Mock()
                backend.generate.return_value = raw
                gate, _ = evidence_search.run_window(question(), frames(), backend, include_direct_control=False)
                self.assertEqual((gate["status"], gate["action"], gate["thought"], gate["prediction"]),
                                 ("invalid_response", None, None, None))

    def test_false_with_a_candidate_routes_to_history_for_either_action(self):
        for action in ("ANSWER", "SEARCH"):
            with self.subTest(action=action):
                self.reader.backend.generate.return_value = (
                    '{"action":"' + action + '","evidence_sufficient":false,"option":"B","thought":"Need earlier evidence."}')
                with patch("event_memory.agent_retrieval.run_agent_history", return_value={"prediction": "C"}) as history:
                    result, _ = evidence_search.run_question(self.reader, question(), {}, frames(),
                                                             include_direct_control=False)
                gate = result["short_window"]
                self.assertIs(gate["evidence_sufficient"], False)
                self.assertEqual((gate["model_action"], gate["action"], gate["candidate_option"],
                                  gate["prediction"], result["prediction"]), (action, "SEARCH", "B", None, "C"))
                history.assert_called_once()

    def test_normal_agent_receives_the_search_note_from_the_window(self):
        self.reader.backend.generate.return_value = WINDOW_SEARCH
        with patch("event_memory.agent_retrieval.run_agent_history", return_value={"prediction": "C"}) as history:
            result, _ = evidence_search.run_question(self.reader, question(), {}, frames(),
                                                     include_direct_control=False)
        self.assertEqual(history.call_args.kwargs["policy"], "search_agent")
        self.assertEqual(history.call_args.kwargs["window_thought"], result["short_window"]["thought"])
        self.assertIn("object color is obscured", history.call_args.kwargs["window_thought"])

    def test_model_or_memory_infrastructure_errors_are_not_reported_as_abstention(self):
        self.reader.backend.generate.side_effect = RuntimeError("GPU lost")
        with self.assertRaisesRegex(RuntimeError, "GPU lost"):
            evidence_search.run_window(question(), frames(), self.reader.backend)

    def test_retired_policies_are_rejected_before_window_inference(self):
        for policy in ("fixed", "baseline"):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "unknown retrieval policy"):
                evidence_search.run_question(self.reader, question(), {}, frames(), policy=policy)
        self.reader.backend.generate.assert_not_called()

    def test_search_candidates_prioritize_distinct_phases_then_fill_remaining_slots(self):
        hits = [{"event_id": event, "phase": {"id": phase}}
                for event, phase in [("E1", "PH1"), ("E2", "PH1"), ("E3", "PH2"), ("E4", "PH1")]]
        self.assertEqual(evidence_search._history_candidates(hits), [hits[0], hits[2], hits[1]])

    def test_invalid_history_response_can_force_but_invalid_forced_letter_fails(self):
        history = {"status": "invalid_response", "prediction": None, "attempts": [],
                   "page_evidence": [], "context": ""}
        self.reader.backend.choose_option.return_value = "A"
        result = evidence_search.finalize_search(question(), frames(), history, self.reader.backend)
        self.assertEqual((result["status"], result["prediction"]), ("answered", "A"))
        self.reader.backend.choose_option.return_value = "not an option"
        with self.assertRaisesRegex(ValueError, "option letter"):
            evidence_search.finalize_search(question(), frames(), history, self.reader.backend)

    def test_run_question_uses_direct_answer_only_for_classified_memory_unavailability(self):
        self.reader.backend.generate.side_effect = ["B", WINDOW_SEARCH]
        with patch("event_memory.agent_retrieval.run_agent_history",
                          side_effect=evidence_search.RecoverableMemoryUnavailable("prefix format exhausted")):
            result, direct = evidence_search.run_question(self.reader, question(), {}, frames())
        self.assertEqual((direct["prediction"], result["status"], result["prediction"], result["final_mode"]),
                         ("B", "answered", "B", "direct_when_memory_unavailable"))

    def test_run_question_forces_recent_window_when_direct_output_is_invalid(self):
        self.reader.backend.generate.side_effect = ["an option is B", WINDOW_SEARCH]
        self.reader.backend.choose_option.return_value = "C"
        with patch("event_memory.agent_retrieval.run_agent_history",
                          side_effect=evidence_search.RecoverableMemoryUnavailable("known missing prefix")):
            result, direct = evidence_search.run_question(self.reader, question(), {}, frames())
        self.assertIsNone(direct["prediction"])
        self.assertEqual((result["status"], result["prediction"], result["final_mode"]),
                         ("answered", "C", "forced_short_window_when_memory_unavailable"))
        self.assertEqual(self.reader.backend.choose_option.call_args.kwargs["evidence"], frames())

    def test_run_question_propagates_integrity_and_model_errors(self):
        self.reader.backend.generate.side_effect = ["B", WINDOW_SEARCH]
        with patch("event_memory.agent_retrieval.run_agent_history", side_effect=ValueError("bad source hash")):
            with self.assertRaisesRegex(ValueError, "bad source hash"):
                evidence_search.run_question(self.reader, question(), {}, frames())


if __name__ == "__main__":
    unittest.main()
