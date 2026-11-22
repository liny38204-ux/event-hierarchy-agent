from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from event_memory import agent_retrieval as agent


def question():
    return {"id": 42, "question": "Where is the vase?", "options": ["floor", "table", "sink"],
            "video": "video.mp4", "realtime": 10.0}


def frames():
    return [{"id": "f000001", "time": 1.0, "sample_time": 1.0,
             "sha256": "a" * 64, "path": "/frame.jpg"}]


class Backend:
    def __init__(self, replies, forced="A"):
        self.replies = iter(replies)
        self.forced = forced
        self.calls = []

    def generate(self, system, text, *, evidence, stage, max_tokens):
        self.calls.append({"stage": stage, "evidence": evidence, "text": text, "system": system})
        return next(self.replies)

    def choose_option(self, system, text, *, evidence, options, stage):
        self.calls.append({"stage": stage, "evidence": evidence, "text": text})
        return self.forced


class Tools:
    latest = None

    def __init__(self, reader, _question, _job):
        type(self).latest = self
        self.backend = reader.backend
        self.origin = "manager_merged_events"
        self.refs = {}
        self.pages = {}
        self.hits = {}
        self.searches = []
        self.inspections = []
        self.expansions = []
        self.expanded = {}

    def search(self, query, scope):
        self.searches.append((query, scope))
        candidates = []
        for phase, event, windows in (("PH1", "E1", ["<0s - 1s>"]),
                                      ("PH2", "E2", ["<2s - 3s>"])):
            self.hits[phase] = {"event_id": event, "caption": f"{windows[0]}\nvase"}
            ref = f"C{len(self.refs) + 1}:{phase}"
            self.refs[ref] = {"kind": "caption", "phase": phase, "text": query}
            candidates.append({
                "phase": {"id": phase,
                          "title": "first room" if phase == "PH1" else "second room",
                          "summary": "vase appears" if phase == "PH1" else "vase moves"},
                "episode": {"id": "P1", "title": "tour", "summary": "indoor tour"},
                "caption_ref": ref, "windows": windows, "matched_sentence": "vase",
            })
        return {"query": query, "scope": scope, "candidates": candidates}

    def inspect(self, req, number, stage):
        self.inspections.append((req, number, stage))
        phase = req["phase"]
        event = self.hits[phase]["event_id"]
        fid = "f000002" if event == "E2" else "f000001"
        ref = f"V{number}:{fid}"
        self.refs[ref] = {"kind": "visual", "phase": phase, "frame_id": fid, "time": 2.0}
        page = {"frame_ids": [fid], "sha256": event, "path": f"/{event}.png"}
        self.pages[phase] = page
        return {"phase": phase, "window": req.get("window", "matched"), "question": req["question"],
                "visual_refs": {ref: 2.0}, "observation": "The vase is on the floor.",
                "unclear": "", "page": page, "cells": {fid: 2.0}}

    def expand_phase(self, req):
        phase = req["phase"]
        self.expansions.append(phase)
        if phase not in self.hits:
            raise agent.ActionError("Use a phase ID from a search result.")
        if phase in self.expanded:
            return self.expanded[phase]
        blocks = []
        for index, (window, text) in enumerate((("<0s - 1s>", "The vase is on a table."),
                                                 ("<2s - 3s>", "The vase is now on the floor.")), 1):
            ref = f"X{index}:{phase}"
            self.refs.setdefault(ref, {"kind": "caption", "phase": phase,
                                       "window": window, "text": text})
            blocks.append({"window": window, "text": text, "caption_ref": ref})
        self.expanded[phase] = {"phase": phase, "blocks": blocks}
        return self.expanded[phase]

    def focus_text(self, req, number):
        phase = req["phase"]
        windows = req["windows"]
        ref = f"F{number}:{phase}"
        text = "\n".join(windows)
        self.refs[ref] = {"kind": "caption", "phase": phase, "windows": windows, "text": text}
        return {"phase": phase, "focused_windows": windows, "caption_ref": ref, "caption": text}


class Reader:
    def __init__(self, backend):
        self.backend = backend


class TableEmbedder:
    """Deterministic vectors for exercising the real caption index."""
    def __init__(self, documents, queries):
        self.documents = documents
        self.queries = queries

    def encode_documents(self, texts):
        return [self.documents[text] for text in texts]

    def encode_query(self, text):
        return self.queries[text]


def action(**kwargs):
    kwargs.setdefault("evidence_sufficient", kwargs.get("action") == "ANSWER")
    return __import__("json").dumps(kwargs, ensure_ascii=False)


class AgentControllerTests(unittest.TestCase):
    def execute(self, policy, replies, forced="A", **kwargs):
        backend = Backend(replies, forced=forced)
        reader = Reader(backend)
        with patch.object(agent, "HistoryTools", Tools):
            result = agent.run_agent_history(reader, question(), {}, recent_frames=frames(),
                                             policy=policy, **kwargs)
        return result, backend

    def test_inspect_agent_can_select_second_candidate_and_answer_with_visual_ref(self):
        result, backend = self.execute("inspect_agent", [
            action(action="INSPECT", phase="PH2", window="<2s - 3s>", question="What supports the vase?", thought="Candidate two may show the vase location."),
            action(action="ANSWER", option="A", ref="V2:f000002", thought="The cited frame shows the vase on the floor."),
        ])
        self.assertEqual((result["status"], result["prediction"], result["evidence_ref"]),
                         ("answered", "A", "V2:f000002"))
        self.assertEqual([step["action"] for step in result["actions"]], ["SEARCH", "INSPECT", "ANSWER"])
        self.assertIsNone(result["actions"][0]["thought"])
        self.assertEqual(result["actions"][1]["thought"], result["actions"][1]["reason"])
        self.assertEqual(backend.calls[0]["stage"], "inspect_agent:42:controller:2")

    def test_search_agent_can_search_again_inside_returned_phase(self):
        result, _ = self.execute("search_agent", [
            action(action="SEARCH", query="orange flowers", scope="all", thought="I need the event containing the object."),
            action(action="SEARCH", query="vase base", scope="PH2", thought="The returned phase may resolve the vase location."),
            action(action="ANSWER", option="A", ref="C2:PH2", thought="The scoped caption supports option A."),
        ])
        self.assertEqual(result["prediction"], "A")
        self.assertEqual([(item["request"]["query"], item["request"]["scope"])
                          for item in result["actions"] if item["action"] == "SEARCH"],
                         [("orange flowers", "all"), ("vase base", "PH2")])

    def test_explicit_three_tool_budget_requires_an_answer_after_three_tools(self):
        result, _ = self.execute("inspect_agent", [
            action(action="INSPECT", phase="PH1", window="<0s - 1s>", question="Find the vase.", thought="The location remains unverified."),
            action(action="INSPECT", phase="PH1", window="<0s - 1s>", question="Confirm the floor.", thought="I need to verify the floor relation."),
            action(action="ANSWER", option="A", ref="V3:f000001", thought="The inspected frame supports option A."),
        ], max_history_tools=3)
        self.assertEqual((result["tool_calls"], result["final_mode"], result["prediction"]),
                         (3, "forced_budget_answer", "A"))

    def test_default_twenty_tool_budget_keeps_nineteen_inspections_after_initial_search(self):
        inspections = [
            action(action="INSPECT", phase="PH1", window="<0s - 1s>",
                   question=f"Check detail {index}.", thought=f"Detail {index} remains unverified.")
            for index in range(1, 20)
        ]
        result, backend = self.execute("inspect_agent", [
            *inspections,
            action(action="ANSWER", option="A", ref="V20:f000001",
                   thought="The final requested check supports option A."),
        ])
        self.assertEqual((result["tool_calls"], result["prediction"], result["final_mode"]),
                         (20, "A", "forced_budget_answer"))
        self.assertEqual(len([item for item in result["actions"] if item["action"] == "INSPECT"]), 19)
        controller_text = [call["text"] for call in backend.calls if "controller" in call["stage"]]
        self.assertIn("Remaining tool calls: 19.", controller_text[0])
        self.assertIn("Remaining tool calls: 0.", controller_text[-1])

    def test_default_budget_rejects_twenty_first_tool_before_calling_it(self):
        inspections = [
            action(action="INSPECT", phase="PH1", window="<0s - 1s>",
                   question=f"Check detail {index}.", thought=f"Detail {index} remains unverified.")
            for index in range(1, 20)
        ]
        result, backend = self.execute("inspect_agent", [
            *inspections,
            action(action="INSPECT", phase="PH1", window="<0s - 1s>",
                   question="This would be tool 21.", thought="One more detail remains unverified."),
            action(action="ANSWER", option="A", ref="V20:f000001",
                   thought="The twentieth tool supports option A."),
        ])
        self.assertEqual((result["tool_calls"], result["prediction"], result["final_mode"]),
                         (20, "A", "forced_budget_answer"))
        self.assertEqual(len([item for item in result["actions"] if item["action"] == "INSPECT"]), 19)
        self.assertEqual(result["actions"][-2]["action"], "INVALID")
        self.assertIn("No tool calls remain", result["actions"][-2]["error"])
        self.assertEqual(len(Tools.latest.inspections), 19)

    def test_larger_budget_preserves_early_answer_exit(self):
        result, backend = self.execute("inspect_agent", [
            action(action="ANSWER", option="A", ref="C1:PH1",
                   thought="The initial retrieved caption supports option A."),
        ])
        self.assertEqual((result["tool_calls"], result["prediction"], result["final_mode"]),
                         (1, "A", "agent_answer"))
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(Tools.latest.inspections, [])

    def test_window_thought_is_handed_to_planner_but_not_promoted_to_evidence(self):
        window_note = "The vase may be near the floor, but its location is unclear."
        result, backend = self.execute("inspect_agent", [
            action(action="ANSWER", option="A", ref="C1:PH1",
                   thought="The retrieved caption is the evidence supporting option A."),
        ], window_thought=window_note)
        self.assertEqual(result["cited_evidence"], Tools.latest.refs["C1:PH1"])
        self.assertNotIn("window", result["evidence_refs"])
        self.assertIn("unverified and not citeable evidence", backend.calls[0]["text"])
        self.assertIn(window_note, backend.calls[0]["text"])

    def test_missing_model_thought_is_a_protocol_error(self):
        result, _ = self.execute("inspect_agent", [
            action(action="ANSWER", option="A", ref="C1:PH1"),
            action(action="ANSWER", option="A", ref="C1:PH1"),
            action(action="ANSWER", option="A", ref="C1:PH1",
                   thought="The retrieved caption supports option A."),
        ])
        self.assertEqual((result["status"], result["prediction"], result["protocol_errors"]),
                         ("answered", "A", 2))
        self.assertIn("Missing text field: thought", result["actions"][1]["error"])
        self.assertIsNone(result["actions"][1]["thought"])
        self.assertEqual([item["action"] for item in result["recovery_actions"]],
                         ["EXPAND_PHASE", "EXPAND_PHASE"])

    def test_answer_accepts_matching_labeled_option_and_rejects_conflict(self):
        result, _ = self.execute("inspect_agent", [
            action(action="ANSWER", option="A. table", ref="C1:PH1",
                   thought="The retrieved caption supports option A."),
            action(action="ANSWER", option="A. floor", ref="C1:PH1",
                   thought="The retrieved caption supports option A."),
        ])
        self.assertEqual((result["prediction"], result["protocol_errors"]), ("A", 1))
        self.assertIn("valid option", result["actions"][1]["error"])

    def test_repeated_inspections_are_independent_and_see_prior_results(self):
        result, backend = self.execute("inspect_agent", [
            action(action="INSPECT", phase="PH1", window="<0s - 1s>",
                   question="Where is the vase?", thought="The exact vase location remains unverified."),
            action(action="INSPECT", phase="PH2", window="<2s - 3s>",
                   question="Is it still on the floor?", thought="I need to check whether the vase remains on the floor."),
            action(action="ANSWER", option="A", ref="V3:f000002",
                   thought="Both visual observations support option A."),
        ], max_history_tools=3)
        self.assertEqual((result["tool_calls"], result["prediction"]), (3, "A"))
        self.assertEqual(
            [(req["phase"], req["question"]) for req, _number, _stage in Tools.latest.inspections],
            [("PH1", "Where is the vase?"), ("PH2", "Is it still on the floor?")],
        )
        controller_text = [call["text"] for call in backend.calls if "controller" in call["stage"]]
        self.assertIn("The vase is on the floor.", controller_text[1])

    def test_expand_phase_returns_chronological_blocks_and_only_one_copy_stays_in_context(self):
        result, backend = self.execute("search_agent", [
            action(action="SEARCH", query="vase", scope="all", thought="Locate the phase containing the vase."),
            action(action="EXPAND_PHASE", phase="PH1", thought="I need the phase timeline before deciding."),
            action(action="EXPAND_PHASE", phase="PH1", thought="I need to revisit the same phase timeline."),
            action(action="ANSWER", option="A", ref="X2:PH1",
                   thought="The second phase block says the vase is on the floor."),
        ], max_history_tools=3)
        self.assertEqual((result["prediction"], result["tool_calls"]), ("A", 3))
        self.assertEqual([item["action"] for item in result["actions"]],
                         ["SEARCH", "EXPAND_PHASE", "EXPAND_PHASE", "ANSWER"])
        self.assertEqual(Tools.latest.expansions, ["PH1", "PH1"])
        controller_text = [call["text"] for call in backend.calls if "controller" in call["stage"]]
        self.assertEqual(controller_text[-1].count("The vase is now on the floor."), 1)
        self.assertEqual(result["cited_evidence"], Tools.latest.refs["X2:PH1"])

    def test_invalid_budget_is_rejected_before_history_tools_or_model_calls(self):
        backend = Backend([])
        reader = Reader(backend)
        with patch.object(agent, "HistoryTools") as history_tools:
            for value in (0, -1, 1.5, True, "20"):
                with self.assertRaisesRegex(ValueError, "max_history_tools"):
                    agent.run_agent_history(reader, question(), {}, recent_frames=frames(),
                                            policy="inspect_agent", max_history_tools=value)
        history_tools.assert_not_called()
        self.assertEqual(backend.calls, [])

    def test_protocol_errors_recover_with_tools_instead_of_early_forced_answer(self):
        result, backend = self.execute("inspect_agent", [
            action(action="SEARCH", query="vase", scope="all", thought="I want to search although this policy disallows it."),
            action(action="ANSWER", option="A", ref="missing", thought="An unavailable reference supports option A."),
            action(action="ANSWER", option="A", ref="C1:PH1",
                   thought="The retrieved caption supports option A."),
        ])
        self.assertEqual((result["status"], result["prediction"], result["final_mode"]),
                         ("answered", "A", "agent_answer"))
        self.assertEqual(result["protocol_errors"], 2)
        self.assertEqual([item["action"] for item in result["recovery_actions"]],
                         ["EXPAND_PHASE", "EXPAND_PHASE"])
        self.assertFalse(any("must-answer:42" in call["stage"] for call in backend.calls))

    def test_insufficient_answer_is_blocked_then_recovery_explores_before_answering(self):
        result, _ = self.execute("inspect_agent", [
            action(action="ANSWER", option="A", ref="C1:PH1",
                   thought="This is only a tentative choice.", evidence_sufficient=False),
            action(action="ANSWER", option="A", ref="C1:PH1",
                   thought="The expanded timeline now supports option A.", evidence_sufficient=True),
        ])
        self.assertEqual((result["prediction"], result["tool_calls"]), ("A", 2))
        self.assertEqual([item["action"] for item in result["actions"]],
                         ["SEARCH", "BLOCKED_ANSWER", "EXPAND_PHASE", "ANSWER"])
        self.assertEqual(result["blocked_candidates"][0]["option"], "A")
        self.assertEqual(result["recovery_actions"][0]["tool"], "EXPAND_PHASE")

    def test_false_exploration_executes_even_with_option_fields_and_string_bool_is_recovered(self):
        result, _ = self.execute("inspect_agent", [
            action(action="INSPECT", phase="PH1", window="<0s - 1s>", question="Check the vase.",
                   thought="The location needs visual confirmation.", evidence_sufficient=False,
                   option="A", ref="C1:PH1"),
            action(action="ANSWER", option="A", ref="C1:PH1",
                   thought="Wrong type must not terminate.", evidence_sufficient="false"),
            action(action="ANSWER", option="A", ref="C1:PH1",
                   thought="The evidence supports A.", evidence_sufficient=True),
        ])
        inspect = next(item for item in result["actions"] if item["action"] == "INSPECT")
        self.assertEqual(inspect["ignored_answer_fields"], {"option": "A", "ref": "C1:PH1"})
        self.assertEqual(len(Tools.latest.inspections), 1)
        self.assertEqual(result["prediction"], "A")
        self.assertIn("JSON boolean", next(item for item in result["actions"]
                                             if item["action"] == "INVALID")["error"])


class HistoryToolTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend([
            action(observation="The vase touches the floor.", frame_ids=["f1"], unclear=""),
            action(observation="The vase remains on the floor.", frame_ids=["f2"], unclear=""),
        ])
        self.reader = Mock()
        self.reader.backend = self.backend
        self.reader.build.return_value = ({"memory": "fixture"}, Path("/pages"), "manager_merged_events")
        self.reader._get_embedder.return_value = object()
        self.tools = None

    def test_first_prompt_and_focused_followup_use_the_real_text_action(self):
        headers = ["<0s - 1s>", "<2s - 3s>", "<4s - 5s>"]
        texts = ["The vase was on a table.", "An unrelated observation.", "The vase now stands on the floor."]
        hit = {"event_id": "E1", "caption": "\n".join(f"{h}\n{t}" for h, t in zip(headers, texts)),
               "score": 1., "phase": {"id": "PH1", "title": "room tour", "summary": "the vase changes place"},
               "episode": None, "best_action_line": {"text": texts[1], "time_range": [2, 3]}}
        self.reader.build.return_value = ({"store": {"events": {"E1": {}}}}, Path("/pages"), "test")
        self.backend.replies = iter([
            action(action="SEARCH", query="vase", scope="all", thought="Locate the vase."),
            action(action="FOCUS_TEXT", phase="PH1", windows=[headers[2], headers[0]],
                   thought="Compare the first and last observations."),
            action(action="ANSWER", option="A", ref="F2:PH1", thought="The later observation places the vase on the floor."),
        ])
        with patch.object(agent, "validate_merged_memory"), patch.object(agent, "CaptionRetriever") as index:
            index.return_value.search.return_value = [hit]
            result = agent.run_agent_history(self.reader, question(), {}, recent_frames=frames(), policy="search_agent")
        self.assertIn('"action":"FOCUS_TEXT"', self.backend.calls[0]["system"])
        self.assertEqual([item["action"] for item in result["actions"]], ["SEARCH", "FOCUS_TEXT", "ANSWER"])
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(result["cited_evidence"]["text"], f"{headers[0]}\n{texts[0]}\n{headers[2]}\n{texts[2]}")
        self.assertIn(question()["question"], self.backend.calls[-1]["text"])
        self.assertNotIn(texts[1], self.backend.calls[-1]["text"])
        self.assertTrue(all(not call["evidence"] for call in self.backend.calls))

    def test_phase_text_tools_keep_raw_event_private(self):
        caption = "<0s - 1s>\nfirst\n<2s - 3s>\nsecond"
        with patch.object(agent, "validate_merged_memory"):
            tools = agent.HistoryTools(self.reader, question(), {})
        tools.hits["PH1"] = {"event_id": "E1", "caption": caption}

        focused = tools.focus_text({"phase": "PH1", "windows": ["<2s - 3s>"]}, 2)
        full = tools.inspect({"phase": "PH1", "view": "caption"}, 3, "caption:full")

        self.assertEqual((focused["phase"], focused["caption_ref"], focused["caption"]),
                         ("PH1", "F2:PH1", "<2s - 3s>\nsecond"))
        self.assertEqual((full["phase"], full["caption_ref"], full["caption"]),
                         ("PH1", "T3:PH1", caption))
        self.assertEqual(tools.refs["F2:PH1"]["phase"], "PH1")
        self.assertEqual(tools.refs["T3:PH1"]["phase"], "PH1")
        self.assertNotIn("E1", __import__("json").dumps(
            {"focused": focused, "full": full, "refs": tools.refs}, sort_keys=True))

    def test_expand_phase_exposes_all_timed_blocks_once_and_reuses_the_same_refs(self):
        caption = "<0s - 1s>\nfirst\n<2s - 3s>\nsecond"
        with patch.object(agent, "validate_merged_memory"):
            tools = agent.HistoryTools(self.reader, question(), {})
        tools.hits["PH1"] = {"event_id": "E1", "caption": caption}

        expanded = tools.expand_phase({"phase": "PH1"})
        repeated = tools.expand_phase({"phase": "PH1"})

        self.assertEqual(expanded, {
            "phase": "PH1",
            "blocks": [
                {"window": "<0s - 1s>", "text": "first", "caption_ref": "X1:PH1"},
                {"window": "<2s - 3s>", "text": "second", "caption_ref": "X2:PH1"},
            ],
        })
        self.assertEqual(repeated, expanded)
        self.assertEqual(tools.refs["X2:PH1"],
                         {"kind": "caption", "phase": "PH1", "window": "<2s - 3s>", "text": "second"})
        self.assertNotIn("E1", __import__("json").dumps({"expanded": expanded, "refs": tools.refs}, sort_keys=True))

    def test_expand_phase_accepts_untimed_legacy_caption_and_rejects_unknown_phase(self):
        with patch.object(agent, "validate_merged_memory"):
            tools = agent.HistoryTools(self.reader, question(), {})
        tools.hits["PH1"] = {"event_id": "E1", "caption": "A vase is on the table."}

        self.assertEqual(tools.expand_phase({"phase": "PH1"}), {
            "phase": "PH1",
            "blocks": [{"window": None, "text": "A vase is on the table.", "caption_ref": "X1:PH1"}],
        })
        with self.assertRaisesRegex(agent.ActionError, "phase ID"):
            tools.expand_phase({"phase": "E1"})

    def test_focus_rejects_unknown_or_unlisted_times_without_creating_a_ref(self):
        with patch.object(agent, "validate_merged_memory"):
            tools = agent.HistoryTools(self.reader, question(), {})
        tools.hits["PH1"] = {"event_id": "E1", "caption": "<0s - 1s>\nfirst"}
        for phase, windows in [("PH2", ["<0s - 1s>"]), ("PH1", []), ("PH1", "matched"),
                               ("PH1", ["<0s - 1s>", "<6s - 7s>"]), ("E1", ["<0s - 1s>"])]:
            with self.subTest(phase=phase, windows=windows), self.assertRaises(agent.ActionError):
                tools.focus_text({"phase": phase, "windows": windows}, 2)
        self.assertEqual(tools.refs, {})

    def _tools(self):
        hit = {"event_id": "E1", "caption": "<0s - 1s>\nfirst\n<2s - 3s>\nsecond"}
        with patch.object(agent, "validate_merged_memory"), patch.object(agent, "select_merged_page") as select:
            self.tools = agent.HistoryTools(self.reader, question(), {})
            self.tools.hits["PH1"] = hit
            select.side_effect = [
                {"frame_ids": ["f1"], "sha256": "page-one", "path": "/one.png", "cells": [{"frame_id": "f1", "time_s": 1.0}]},
                {"frame_ids": ["f1"], "sha256": "page-one", "path": "/one.png", "cells": [{"frame_id": "f1", "time_s": 1.0}]},
                {"frame_ids": ["f2"], "sha256": "page-two", "path": "/two.png", "cells": [{"frame_id": "f2", "time_s": 3.0}]},
            ]
            first = self.tools.inspect({"phase": "PH1", "window": "<0s - 1s>", "question": "Where?"}, 1, "inspect:1")
            repeated = self.tools.inspect({"phase": "PH1", "window": "<0s - 1s>", "question": "Where?"}, 2, "inspect:2")
            second = self.tools.inspect({"phase": "PH1", "window": "<2s - 3s>", "question": "Where?"}, 3, "inspect:3")
        return first, repeated, second

    def test_same_page_and_question_reuses_observation_but_another_window_gets_a_new_page(self):
        first, repeated, second = self._tools()
        self.assertFalse(first["reused"])
        self.assertTrue(repeated["reused"])
        self.assertEqual((second["page_id"], second["visual_refs"]), ("page-two", {"V3:f2": 3.0}))
        self.assertEqual([call["stage"] for call in self.backend.calls], ["inspect:1", "inspect:3"])

    def test_invalid_event_or_window_is_rejected_before_a_visual_call(self):
        self._tools()
        with self.assertRaisesRegex(agent.ActionError, "phase ID"):
            self.tools.inspect({"phase": "E1", "window": "<0s - 1s>", "question": "Where?"}, 4, "bad")
        with self.assertRaisesRegex(agent.ActionError, "window"):
            self.tools.inspect({"phase": "PH1", "window": "<9s - 10s>", "question": "Where?"}, 4, "bad")

    def test_memory_validation_failure_propagates_before_any_tool_or_model_call(self):
        with patch.object(agent, "validate_merged_memory", side_effect=ValueError("future frame")):
            with self.assertRaisesRegex(ValueError, "future frame"):
                agent.HistoryTools(self.reader, question(), {})
        self.assertEqual(self.backend.calls, [])

    def test_real_search_filters_phase_before_the_top_three_and_rejects_unknown_scope(self):
        # Every reviewed phase owns exactly one final event.  PH1 is discoverable
        # in the first search, but E1 becomes fourth globally for the second.
        # Filtering must happen before evidence_search applies its top-three cap.
        captions = {"E1": "target phase record", "E2": "outside record two",
                    "E3": "outside record three", "E4": "outside record four"}
        frames = {f"f{i}": {"id": f"f{i}", "time": float(i), "path": f"/f{i}.jpg",
                           "sha256": f"{i:x}" * 64}
                  for i in range(1, 5)}
        events = {eid: {"id": eid, "caption": caption, "frame_ids": [f"f{i}"],
                        "start": float(i), "end": float(i)}
                  for i, (eid, caption) in enumerate(captions.items(), start=1)}
        phases = {f"PH{i}": {"id": f"PH{i}", "title": f"phase {i}", "summary": "summary",
                              "status": "closed", "closed_at": 4., "start": float(i), "end": float(i),
                              "event_ids": [f"E{i}"]}
                  for i in range(1, 5)}
        episodes = {f"P{i}": {"id": f"P{i}", "title": f"episode {i}", "summary": "summary",
                              "status": "closed", "closed_at": 4., "start": float(i), "end": float(i),
                              "phase_ids": [f"PH{i}"]}
                    for i in range(1, 5)}
        state = {"frames": frames, "events": events, "phases": phases, "episodes": episodes,
                 "roots": [], "completed_phase_ids": list(phases), "observed_until": 4.,
                 "next_merge_id": 1, "next_event_id": 5}
        memory = {"schema": "rolling_phase_memory_v11", "schema_version": 11,
                  "status": "complete", "observed_until": 4., "store": state}
        embedder = TableEmbedder(
            {captions["E1"]: [1., 0.], captions["E2"]: [0., 1.],
             captions["E3"]: [0., 1.], captions["E4"]: [0., 1.]},
            {"discover target": [1., 0.], "outside dominates": [0., 1.]})
        reader = Mock()
        reader.backend = self.backend
        reader.build.return_value = (memory, Path("/pages"), "manager_merged_events")
        reader._get_embedder.return_value = embedder
        tools = agent.HistoryTools(reader, {**question(), "realtime": 4.}, {})

        initial = tools.search("discover target", "all")
        self.assertIn("PH1", {item["phase"]["id"] for item in initial["candidates"]})
        self.assertNotIn("event", initial["candidates"][0])
        self.assertNotIn("E1", __import__("json").dumps(initial, sort_keys=True))
        self.assertEqual(tools.hits["PH1"]["event_id"], "E1")
        scoped = tools.search("outside dominates", "PH1")
        self.assertEqual([item["phase"]["id"] for item in scoped["candidates"]], ["PH1"])
        with self.assertRaisesRegex(agent.ActionError, "P/PH ID returned"):
            tools.search("outside dominates", "PH999")

    def test_invalid_inspector_cell_keeps_raw_audit_without_minting_visual_ref(self):
        raw = action(observation="The vase appears on the floor.", frame_ids=["not-on-page"], unclear="")
        self.backend.replies = iter([raw])
        self.reader._get_embedder.return_value = object()
        with patch.object(agent, "validate_merged_memory"), patch.object(agent, "select_merged_page",
                                                                          return_value={
                                                                              "frame_ids": ["f1"], "sha256": "page-one",
                                                                              "path": "/one.png",
                                                                              "cells": [{"frame_id": "f1", "time_s": 1.0}],
                                                                          }):
            tools = agent.HistoryTools(self.reader, question(), {})
            tools.hits["PH1"] = {"event_id": "E1", "caption": "<0s - 1s>\nfirst"}
            result = tools.inspect({"phase": "PH1", "window": "<0s - 1s>", "question": "Where?"},
                                   1, "inspect:invalid-cell")
        self.assertEqual(result["raw_observation"], raw)
        self.assertIn("outside the delivered page", result["observation_error"])
        self.assertNotIn("visual_refs", result)
        self.assertFalse(any(ref.startswith("V1:") for ref in tools.refs))


if __name__ == "__main__":
    unittest.main()
