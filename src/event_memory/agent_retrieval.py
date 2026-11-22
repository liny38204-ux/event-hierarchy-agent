"""A bounded, caption-indexed QA controller over causal, manager-owned events."""
from __future__ import annotations

import json

from . import evidence_search
from .answer_options import parse_labeled_option
from .merged_retrieval import select_merged_page, validate_merged_memory
from .text_retrieval import CaptionRetriever
from .timed_caption import caption_blocks, header_range, time_header

POLICIES = (evidence_search.DEFAULT_POLICY, "inspect_agent")

CONTROLLER = """Answer the multiple-choice video question from the available evidence. Choose one action
as a JSON object. Every action needs a concise public `thought`: state either the observed fact
that supports this action, or the missing fact that the next action will check. This is a decision
note, not a hidden chain of thought. Use only IDs that appear in the tool results. A candidate's
relatedness alone does not establish the detail being asked. Read the options and check which fact
distinguishes them. SEARCH returns matching phases (PH). Each candidate gives its phase ID, title, summary, a matched
sentence, available time headers, and its parent episode (P) when available. For reading or
inspection, copy the candidate's phase.id into the "phase" field.
Every JSON object must include `evidence_sufficient` as a JSON boolean, never a string. Set it
to false for SEARCH, EXPAND_PHASE, FOCUS_TEXT, and INSPECT: those actions mean more evidence is
needed. Set it to true only for ANSWER when the cited evidence establishes the complete option.
Not seeing a detail, preferring an option because alternatives were not seen, or guessing by
exclusion is not evidence that establishes an option. If evidence is insufficient, explore rather
than emitting an option.
Use EXPAND_PHASE when a phase looks relevant but the deciding moment is unclear, or the question
depends on a sequence. It returns the phase's full text timeline in chronological order, with a
caption reference for each block. Cite a block if it establishes the answer. Use FOCUS_TEXT for
specific listed windows you already want to compare. Use INSPECT with a window and a visual
question to check a detail the text cannot establish.
A caption can be mistaken. If a visual detail is unclear, ask the visual reader
a specific, neutral question in your own words. It sees only the selected page and that question.
You can inspect another time in the same phase or change phases. If the evidence supports an answer,
return ANSWER directly. Available action formats:
{"action":"EXPAND_PHASE","phase":"listed PH ID","thought":"the missing fact to locate in this timeline","evidence_sufficient":false}
{"action":"FOCUS_TEXT","phase":"listed PH ID","windows":["listed time header","another listed time header"],"thought":"the detail to check or compare","evidence_sufficient":false}
{"action":"INSPECT","phase":"listed PH ID","window":"listed time header","question":"visual subquestion","thought":"the missing visual fact to check","evidence_sufficient":false}
{"action":"ANSWER","option":"one supplied letter","ref":"one listed evidence reference","thought":"the cited observed fact supporting this option","evidence_sufficient":true}
A caption ref identifies text that was returned to you. A visual ref identifies an actual cell
cited by the visual reader. Use the ref for the evidence supporting your answer."""
SEARCH_RULE = """
You may also use {"action":"SEARCH","query":"a precise retrieval query","scope":"all","thought":"the missing fact this search will locate","evidence_sufficient":false}.
The scope is all, or a parent episode ID (P) or phase ID (PH) from an earlier result. Write the query in the language best
suited to the visual target. Later queries may use the findings of an earlier search."""
INSPECTOR = """Inspect only the supplied timestamped video contact sheet to answer the supplied
visual question. Each occupied cell is one frame. Read the ID and time printed above that cell.
Read across the top row then the bottom row. Describe only visible facts that bear on the question.
If a relationship or action is unclear, say exactly what remains unclear. Return a JSON object
{"observation":"brief visible facts","frame_ids":["exact cell ID"],"unclear":"missing detail or empty string"}.
If no cell supports a relevant observation, use an empty frame_ids list. Do not choose an option."""


class ActionError(ValueError):
    """A correctable controller or visual-reader output error."""


def _json_object(raw):
    body = raw.strip() if isinstance(raw, str) else ""
    if body.startswith("```") and body.endswith("```"):
        body = body.partition("\n")[2].removesuffix("```").strip()
    try:
        value = json.loads(body)
    except ValueError as exc:
        raise ActionError("Return one valid JSON object.") from exc
    if not isinstance(value, dict):
        raise ActionError("Expected a JSON object.")
    return value


def _string(obj, key):
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ActionError(f"Missing text field: {key}")
    return value.strip()


def _bool(obj, key):
    """Require JSON's actual boolean type (not truthy strings or numbers)."""
    value = obj.get(key)
    if type(value) is not bool:
        raise ActionError(f"{key} must be a JSON boolean.")
    return value


def _compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _caption_excerpt(caption, header):
    lines = caption.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == header)
    end = next((i for i in range(start + 1, len(lines)) if header_range(lines[i])), len(lines))
    return "\n".join(lines[start:end])


def _window_planning_note(window_thought):
    """Render an optional recent-window note without promoting it to evidence.

    The window model has already seen the same frames, but its note is useful
    when choosing the first history action. It must never become a citeable
    evidence reference or a substitute for a returned caption or image.
    """
    if not isinstance(window_thought, str):
        return ""
    note = " ".join(window_thought.split())
    if not note:
        return ""
    return ("\nRecent-window planner note, unverified and not citeable evidence: "
            f"{note}\nDo not use this note as support for ANSWER. Verify it with returned evidence.\n")


class HistoryTools:
    def __init__(self, reader, question, job):
        self.reader = reader
        self.backend = reader.backend
        self.time = question["realtime"]
        self.memory, self.root, self.origin = reader.build(question, job)
        validate_merged_memory(self.memory, self.time)
        self.embedder = reader._get_embedder()
        self.index = None
        self.hits = {}
        self.scopes = {"all"}
        self.refs = {}
        self.pages = {}
        self.observations = {}

    def windows(self, hit):
        return {line.strip(): header_range(line) for line in hit["caption"].splitlines()
                if header_range(line) is not None}

    def search(self, query, scope):
        if scope not in self.scopes:
            raise ActionError("Use all or a P/PH ID returned by a previous search.")
        self.reader._move(self.embedder, "cuda:1")
        try:
            if self.index is None:
                self.index = CaptionRetriever(self.memory, self.embedder)
            ranking = self.index.search(query, top_k=max(1, len(self.memory["store"]["events"])),
                                        query_time=self.time)
        finally:
            self.reader._move(self.embedder, "cpu")
        if scope != "all":
            ranking = [h for h in ranking if scope in
                       ((h.get("phase") or {}).get("id"), (h.get("episode") or {}).get("id"))]
        candidates = []
        for hit in evidence_search._history_candidates(ranking):
            phase = hit["phase"]
            pid = phase["id"]
            self.hits[pid] = hit
            self.scopes.add(pid)
            anchor = hit["best_action_line"]
            windows = list(self.windows(hit))
            bounds = anchor.get("time_range")
            header = time_header(*bounds) if bounds else "matched"
            # Different queries can return different sentences for the same window.
            ref = f"C{len(self.refs) + 1}:{pid}"
            self.refs[ref] = {"kind": "caption", "phase": pid, "window": header, "text": anchor["text"]}
            item = {"phase": phase, "score": round(hit["score"], 5),
                    "matched_sentence": anchor["text"], "matched_window": header,
                    "caption_ref": ref, "windows": windows or ["matched"]}
            if hit.get("episode"):
                item["episode"] = hit["episode"]
                self.scopes.add(hit["episode"]["id"])
            if "event_pages" in self.memory:
                item["saved_pages"] = [{"page_id": p["sha256"], "times": [c["time_s"] for c in p["cells"]]}
                                       for p in self.memory["event_pages"][hit["event_id"]]]
                item["page_selection"] = "Optionally include a listed page_id in INSPECT to choose that exact page."
            candidates.append(item)
        return {"query": query, "scope": scope, "candidates": candidates}

    def _phase_hit(self, req):
        pid = _string(req, "phase")
        if pid not in self.hits:
            raise ActionError("Use a phase ID from a search result.")
        return pid, self.hits[pid]

    def expand_phase(self, req):
        pid, hit = self._phase_hit(req)
        caption = hit["caption"]
        try:
            blocks = [(time_header(*bounds), "\n".join(lines)) for bounds, lines in caption_blocks(caption)]
        except ValueError:
            # Older records can contain plain text without canonical time headers.
            blocks = [(None, caption)]

        timeline = []
        for index, (window, text) in enumerate(blocks, start=1):
            ref = f"X{index}:{pid}"
            self.refs[ref] = {"kind": "caption", "phase": pid, "window": window, "text": text}
            timeline.append({"window": window, "text": text, "caption_ref": ref})
        return {"phase": pid, "blocks": timeline}

    def focus_text(self, req, number):
        pid, hit = self._phase_hit(req)
        available = self.windows(hit)
        requested = req.get("windows")
        if not isinstance(requested, list) or not requested or not all(
            isinstance(header, str) and header in available for header in requested
        ):
            raise ActionError("FOCUS_TEXT needs a nonempty list of exact time headers from this phase.")
        headers = [header for header in available if header in requested]
        text = "\n".join(_caption_excerpt(hit["caption"], header) for header in headers)
        ref = f"F{number}:{pid}"
        self.refs[ref] = {"kind": "caption", "phase": pid, "windows": headers, "text": text}
        return {"phase": pid, "focused_windows": headers, "caption_ref": ref, "caption": text}

    def inspect(self, req, number, stage):
        pid, hit = self._phase_hit(req)
        windows = self.windows(hit)
        header = req.get("window", "matched")
        if not isinstance(header, str) or (header != "matched" and header not in windows):
            raise ActionError("Copy a window exactly from the phase directory.")
        view = req.get("view", "page")
        if view == "caption":
            caption = hit["caption"]
            if header != "matched":
                caption = _caption_excerpt(caption, header)
            ref = f"T{number}:{pid}"
            self.refs[ref] = {"kind": "caption", "phase": pid, "window": header if header != "matched" else "full",
                              "text": caption}
            return {"phase": pid, "caption_ref": ref, "caption": caption}
        if view != "page":
            raise ActionError("INSPECT view must be page or caption.")
        query = _string(req, "question")
        page_args = {"requested_window": windows.get(header)}
        if "page_id" in req:
            page_id = _string(req, "page_id")
            owned = self.memory.get("event_pages", {}).get(hit["event_id"], [])
            page_info = next((p for p in owned if p["sha256"] == page_id), None)
            if page_info is None:
                raise ActionError("Choose a saved page ID listed for this phase.")
            if header in windows:
                start, end = windows[header]
                if not any(start <= cell["time_s"] <= end for cell in page_info["cells"]):
                    raise ActionError("The requested page has no cells in that window.")
            page_args["requested_page_id"] = page_id
        page = select_merged_page(self.memory, hit, self.time, self.root, **page_args)
        evidence = evidence_search._page_evidence(page)
        digest = evidence["sha256"]
        seen_page = digest in self.pages
        self.pages[digest] = evidence
        key = (pid, digest, " ".join(query.casefold().split()))
        if key in self.observations:
            return {**self.observations[key], "reused": True, "same_page_seen": True}
        cells = {cell["frame_id"]: cell["time_s"] for cell in page["cells"]}
        raw = self.backend.generate(INSPECTOR, query, evidence=[evidence], stage=stage, max_tokens=256)
        result = {"phase": pid, "page": evidence, "cells": cells, "question": query, "window": header,
                  "page_id": digest, "raw_observation": raw, "same_page_seen": seen_page, "reused": False}
        try:
            obs = _json_object(raw)
            description = _string(obs, "observation")
            ids = obs.get("frame_ids")
            if not isinstance(ids, list) or not all(isinstance(i, str) and i in cells for i in ids):
                raise ActionError("The visual reader cited a cell outside the delivered page.")
            if not isinstance(obs.get("unclear"), str):
                raise ActionError("The visual reader omitted the uncertainty field.")
            refs = {}
            for fid in dict.fromkeys(ids):
                ref = f"V{number}:{fid}"
                self.refs[ref] = {"kind": "visual", "phase": pid, "frame_id": fid, "time": cells[fid],
                                  "page_id": digest, "observation": description}
                refs[ref] = cells[fid]
            result.update(observation=description, visual_refs=refs, unclear=obs["unclear"])
        except ActionError as exc:
            result["observation_error"] = str(exc)
        self.observations[key] = result
        return result


def _visible_result(result):
    # Keep physical paths, raw unverified VLM outputs and duplicated observations out of the planner prompt.
    return {k: v for k, v in result.items() if k not in ("raw_observation", "page", "page_id")}


def run_agent_history(reader, question, job, *, recent_frames, policy,
                      max_history_tools=evidence_search.DEFAULT_MAX_HISTORY_TOOLS,
                      window_thought=None):
    evidence_search.validate_history_tool_budget(max_history_tools)
    if policy not in POLICIES:
        raise ValueError(f"unknown agent policy {policy}")
    prompt, letters = evidence_search._prompt(question)
    tools = HistoryTools(reader, question, job)
    trace, context = [], []
    used = 0
    errors = 0
    answer = None
    blocked_candidates = []
    recovery_actions = []
    recovery_keys = set()
    allow_search = policy == "search_agent"
    prefix = f"{policy}:{question['id']}"

    def execute_search(req, raw, *, automatic=False, recovery=None):
        nonlocal used
        query, scope = _string(req, "query"), _string(req, "scope")
        out = tools.search(query, scope)
        used += 1
        thought = None if automatic else req["thought"]
        item = {"number": used, "action": "SEARCH", "thought": thought,
                      "reason": thought if thought is not None else req["reason"], "raw": raw,
                      "request": req, "result": out,
                      "evidence_sufficient": req.get("evidence_sufficient", False)}
        if recovery is not None:
            item["recovery"] = recovery
        trace.append(item)
        context.append(out)

    def execute_exploration(req, raw, thought, *, recovery=None):
        """Run one actual history tool and preserve the tool result in the trace."""
        nonlocal used, context
        action = req["action"]
        if action == "SEARCH":
            execute_search(req, raw, recovery=recovery)
            return
        if action == "EXPAND_PHASE":
            out = tools.expand_phase(req)
            # A repeat still returns the complete timeline, with the same refs.
            context = [item for item in context
                       if not ("blocks" in item and item["phase"] == out["phase"])]
        elif action == "FOCUS_TEXT":
            out = tools.focus_text(req, used + 1)
        elif action == "INSPECT":
            out = tools.inspect(req, used + 1, f"{prefix}:inspect:{used+1}")
        else:  # guarded by the controller parser; retain a narrow internal invariant.
            raise AssertionError(f"unknown exploration action {action}")
        used += 1
        item = {"number": used, "action": action, "raw": raw, "request": req,
                "thought": thought, "reason": thought, "result": out,
                "evidence_sufficient": False}
        ignored = {key: req[key] for key in ("option", "ref") if key in req}
        if ignored:
            # A false-sufficiency exploration request may carry stale option
            # fields.  Preserve them for audit, but never let them terminate.
            item["ignored_answer_fields"] = ignored
        if recovery is not None:
            item["recovery"] = recovery
        trace.append(item)
        context.append(_visible_result(out))

    def _headers(hit):
        return [line.strip() for line in hit.get("caption", "").splitlines()
                if header_range(line) is not None]

    def recovery_request():
        """Choose the next concrete evidence operation after a bad control turn."""
        if not used:
            key = ("SEARCH", "all", question["question"])
            recovery_keys.add(key)
            return key, {"action": "SEARCH", "query": question["question"], "scope": "all",
                         "reason": "Recovery first retrieval after invalid control output.",
                         "evidence_sufficient": False}
        for pid, hit in tools.hits.items():
            key = ("EXPAND_PHASE", pid)
            if key not in recovery_keys:
                recovery_keys.add(key)
                return key, {"action": "EXPAND_PHASE", "phase": pid,
                             "thought": "Recover by reading this returned phase timeline.",
                             "evidence_sufficient": False}
        for pid, hit in tools.hits.items():
            for window in _headers(hit):
                key = ("FOCUS_TEXT", pid, window)
                if key not in recovery_keys:
                    recovery_keys.add(key)
                    return key, {"action": "FOCUS_TEXT", "phase": pid, "windows": [window],
                                 "thought": "Recover by isolating one returned time window.",
                                 "evidence_sufficient": False}
        for pid, hit in tools.hits.items():
            for window in _headers(hit) or ["matched"]:
                key = ("INSPECT", pid, window)
                if key not in recovery_keys:
                    recovery_keys.add(key)
                    return key, {"action": "INSPECT", "phase": pid, "window": window,
                                 "question": question["question"],
                                 "thought": "Recover by checking the returned visual evidence.",
                                 "evidence_sufficient": False}
        # Empty retrievals still need an actual, bounded recovery attempt.  A
        # repeated causal search is more honest than inventing a phase/window.
        key = ("SEARCH", "all", question["question"], "repeat")
        return key, {"action": "SEARCH", "query": question["question"], "scope": "all",
                     "reason": "Recovery retrieval found no inspectable candidate yet.",
                     "evidence_sufficient": False}

    def recover(trigger, *, blocked=None):
        """Spend an available tool on a traceable recovery, if one exists."""
        if used >= max_history_tools:
            return False
        key, req = recovery_request()
        recovery = {"trigger": trigger, "tool": req["action"], "key": list(key)}
        if blocked is not None:
            recovery["blocked_candidate"] = blocked
        if req["action"] == "SEARCH":
            execute_search(req, None, automatic=True, recovery=recovery)
        else:
            execute_exploration(req, None, req["thought"], recovery=recovery)
        recovery_actions.append({"number": used, "action": req["action"], **recovery, "request": req})
        return True

    if not allow_search:
        execute_search({"action": "SEARCH", "query": question["question"], "scope": "all",
                        "reason": "Fixed first retrieval using the original question.",
                        "evidence_sufficient": False}, None, automatic=True)
    budget_terminal_attempt = False
    while answer is None:
        system = CONTROLLER + (SEARCH_RULE if allow_search else "")
        if not used:
            instruction = "First issue SEARCH with scope all to locate evidence."
        elif used == max_history_tools:
            instruction = ("Tool budget exhausted. Submit the best-supported option now with ANSWER and a valid ref. "
                           "If the evidence is still insufficient, honestly set evidence_sufficient to false: "
                           "this budget-terminal choice is not verification.")
        else:
            instruction = "Choose the next action or ANSWER."
        visible = context
        if context and "focused_windows" in context[-1]:
            visible = [context[-1]]
        text = (f"{prompt}\n"
                f"{_window_planning_note(window_thought)}\nTool results and feedback:\n{_compact(visible)}\n"
                f"Available evidence references: {', '.join(tools.refs) or 'none'}\n"
                f"Remaining tool calls: {max_history_tools-used}. {instruction}")
        raw = reader.backend.generate(system, text, evidence=[], stage=f"{prefix}:controller:{len(trace)+1}",
                                      max_tokens=220)
        req = {}
        try:
            req = _json_object(raw)
            thought = _string(req, "thought")
            action = _string(req, "action").upper()
            sufficient = _bool(req, "evidence_sufficient")
            if not used and action != "SEARCH":
                raise ActionError("Issue an initial SEARCH before using evidence.")
            if action == "ANSWER":
                option = parse_labeled_option(_string(req, "option"), question["options"])
                ref = _string(req, "ref")
                if option is None or option not in letters or ref not in tools.refs:
                    raise ActionError("ANSWER needs a valid option and an exact listed evidence reference.")
                if not sufficient and used < max_history_tools:
                    blocked = {"option": option, "ref": ref, "thought": thought,
                               "reason": "ANSWER declared evidence_sufficient=false while exploration remained."}
                    blocked_candidates.append(blocked)
                    trace.append({"number": used, "action": "BLOCKED_ANSWER", "thought": thought,
                                  "reason": thought, "raw": raw, "request": req,
                                  "blocked_candidate": blocked, "evidence_sufficient": False})
                    context.append({"blocked_answer": blocked,
                                    "instruction": "Explore returned evidence before answering."})
                    recover("insufficient_answer", blocked=blocked)
                    continue
                answer = {"prediction": option, "final_raw": raw,
                          "final_mode": "forced_budget_answer" if used == max_history_tools else "agent_answer",
                          "evidence_ref": ref, "cited_evidence": tools.refs[ref],
                          "evidence_sufficient": sufficient}
                trace.append({"number": used, "action": action, "thought": thought,
                              "reason": thought, "raw": raw, "request": req,
                              "evidence_sufficient": sufficient,
                              "terminal_mode": ("budget_answer_verified" if sufficient else "budget_answer_insufficient")
                              if used == max_history_tools else "agent_answer"})
                break
            if used >= max_history_tools:
                raise ActionError("No tool calls remain. Return ANSWER.")
            if action in ("SEARCH", "EXPAND_PHASE", "FOCUS_TEXT", "INSPECT"):
                if sufficient:
                    raise ActionError("Exploration actions must declare evidence_sufficient=false.")
                if action == "SEARCH" and not allow_search:
                    raise ActionError("Choose an available action.")
                execute_exploration({**req, "action": action}, raw, thought)
            else:
                raise ActionError("Choose an available action.")
        except ActionError as exc:
            errors += 1
            trace.append({"number": used, "action": "INVALID", "thought": None,
                          "raw": raw, "request": req, "error": str(exc),
                          "evidence_sufficient": req.get("evidence_sufficient")})
            context.append({"invalid_action": req, "error": str(exc)})
            if used >= max_history_tools:
                break
            recover("protocol_error")
        if used == max_history_tools:
            # Give the controller exactly one budget-terminal chance to select
            # an answer.  A non-answer/invalid output then goes to the bounded
            # forced chooser below; it cannot spin without remaining tools.
            if budget_terminal_attempt:
                break
            budget_terminal_attempt = True
    terminal_mode = (trace[-1].get("terminal_mode", "agent_answer")
                     if answer is not None else "budget_forced")
    result = {"id": question["id"], "retrieval_policy": policy, "memory_origin": tools.origin,
              "max_history_tools": max_history_tools, "actions": trace, "attempts": trace, "tool_calls": used, "selected_events": list(tools.hits.values()),
              "page_evidence": list(tools.pages.values()), "context": _compact(context),
              "evidence_refs": tools.refs, "protocol_errors": errors,
              "blocked_candidates": blocked_candidates, "recovery_actions": recovery_actions,
              "terminal_mode": terminal_mode}
    if answer is not None:
        return {**result, **answer, "status": "answered"}
    result["status"] = "invalid_response"
    forced = evidence_search.finalize_search(question, recent_frames, result, reader.backend)
    forced["final_mode"] = "forced_budget_answer" if used == max_history_tools else "forced_protocol_answer"
    forced["terminal_mode"] = "budget_forced"
    forced["actions"].append({"action": "ANSWER", "number": used, "thought": None,
                              "raw": forced["final_raw"],
                              "reason": "Constrained fallback after invalid controller output."})
    return forced
