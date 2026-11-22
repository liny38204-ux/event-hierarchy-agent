"""Evidence-first question answering over a recent visual window and manager-merged events.

The module intentionally does not know an answer key or a task label.  A recent
window can either answer directly or request a bounded search through
committed manager events.  Search answers are only accepted when the reader says
that the accumulated text and visual evidence is sufficient.
"""
from __future__ import annotations

from .answer_options import option_letters, parse_labeled_option, parse_option, reader_system
from .protocol import parse_json


DEFAULT_POLICY = "search_agent"
MAX_HISTORY_EVENTS = 3
DEFAULT_MAX_HISTORY_TOOLS = 20
DECISION_FORMAT = "action_with_evidence_sufficiency_v2"


def validate_history_tool_budget(value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError("max_history_tools must be a positive integer")


class RecoverableMemoryUnavailable(RuntimeError):
    """A known prefix-build limitation that may fall back to recent frames.

    Callers must raise this only for an already classified, recoverable absence
    of a causal memory prefix. Integrity checks, model failures, and malformed
    retrieval artifacts remain ordinary exceptions and must propagate.
    """

WINDOW_SYSTEM = """You see {frame_description} in chronological order. Choose your next action.
If a visible detail distinguishes one option for the target and time asked, answer now.
If that detail is missing, ambiguous, or belongs to another target or time, request history.
Set evidence_sufficient to a JSON boolean. Use true only when the visible evidence
supports the selected option. A plausible guess or not seeing an alternative is insufficient.
Return one JSON object. In "thought" give a brief evidence note in one sentence:
for ANSWER name the visible detail supporting the option; for SEARCH state what you can see
and what specific detail an earlier observation should establish. If nothing relevant is visible, say so.
{{"action":"ANSWER","evidence_sufficient":true,"option":"one supplied letter","thought":"brief supporting observation"}}
{{"action":"SEARCH","evidence_sufficient":false,"thought":"what is visible and which missing fact to look for"}}"""


FORCED_CHOICE_SYSTEM = """Choose the best-supported option for the video question from
the supplied visual and textual evidence. First come the recent frames in chronological order;
any following images are the pages of the retrieved phases, in the order listed in the text. You must choose one
of the provided options. Treat retrieved descriptions as evidence, preserve their time
order, and do not add details that are absent from the supplied evidence."""


def _frame_metadata(frames):
    required = ("id", "time", "sample_time", "sha256")
    values = []
    previous = None
    for frame in frames:
        if not isinstance(frame, dict) or any(key not in frame for key in required):
            raise ValueError("recent frame metadata is incomplete")
        if not isinstance(frame["id"], str) or not frame["id"]:
            raise ValueError("recent frame ID is invalid")
        if previous is not None and frame["time"] <= previous:
            raise ValueError("recent frames must be chronological")
        previous = frame["time"]
        values.append({key: frame[key] for key in required})
    if not 1 <= len(values) <= 4:
        raise ValueError("the evidence policy expects one to four recent frames")
    return values


def _prompt(question):
    if not isinstance(question, dict):
        raise ValueError("question must be a mapping")
    options = question.get("options")
    letters = option_letters(options)
    text = question.get("question")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("question text is required")
    choices = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))
    return text.strip() + "\n" + choices, letters


def _window_decision(raw, options):
    if not isinstance(raw, str):
        return None
    try:
        decision = parse_json(raw)
    except ValueError:
        return None
    action = decision.get("action")
    thought = decision.get("thought")
    if not isinstance(thought, str) or not thought.strip():
        return None
    sufficient = decision.get("evidence_sufficient")
    if type(sufficient) is not bool or action not in ("ANSWER", "SEARCH"):
        return None
    option = parse_labeled_option(decision.get("option"), options)
    if sufficient and (action != "ANSWER" or option is None):
        return None
    return {"action": "ANSWER" if sufficient else "SEARCH", "model_action": action,
            "evidence_sufficient": sufficient, "prediction": option if sufficient else None,
            "candidate_option": None if sufficient else option, "thought": thought.strip()}


def run_window(question, frames, backend, *, include_direct_control=True):
    """Return a direct four-image control and an evidence-gated route decision.

    ``backend`` receives the images as independent evidence items.  The
    execution adapter owns their message order and must place them before the
    question text.  No history is supplied in either call.
    """
    frame_meta = _frame_metadata(frames)
    prompt, _ = _prompt(question)
    identifier = question.get("id")
    direct = None
    if include_direct_control:
        direct_system = reader_system(question["options"]).replace(
            "supplied text memory of the video", "supplied recent video frames shown in chronological order")
        direct_raw = backend.generate(direct_system, prompt, evidence=list(frames),
                                      stage=f"evidence-direct:{identifier}", max_tokens=32)
        direct = {"id": identifier, "raw": direct_raw,
                  "prediction": parse_option(direct_raw, question["options"])}

    frame_description = "four recent video frames" if len(frames) == 4 else "the supplied recent video frames"
    system = WINDOW_SYSTEM.format(frame_description=frame_description)
    stage = f"evidence-window:{identifier}"
    raw = backend.generate(system, prompt, evidence=list(frames), stage=stage, max_tokens=160)
    decision = _window_decision(raw, question["options"])
    gate = {"id": identifier, "frames": frame_meta, "cited_frame_ids": []}
    if decision is None:
        gate["invalid_first_raw"] = raw
        repair = ("Return JSON with action, boolean evidence_sufficient and a brief thought. "
                  "Answer with a valid option only if evidence_sufficient is true; otherwise SEARCH.")
        raw = backend.generate(system, f"{prompt}\n\n{repair}", evidence=list(frames),
                               stage=f"{stage}:retry", max_tokens=160)
        decision = _window_decision(raw, question["options"])
    gate["raw"] = raw
    if decision is None:
        gate.update(status="invalid_response", route="history", action=None, model_action=None,
                    evidence_sufficient=None, prediction=None, candidate_option=None, thought=None)
    else:
        gate.update(decision)
        answered = decision["action"] == "ANSWER"
        gate.update(status="answered" if answered else "needs_history",
                    route="short_window" if answered else "history")
    return gate, direct


def _phase_key(hit):
    phase = hit.get("phase") if isinstance(hit, dict) else None
    return phase.get("id") if isinstance(phase, dict) else None


def _history_candidates(ranking):
    """Top-ranked event first, then diversify phases before repeating one."""
    unique, seen_events, phases, deferred = [], set(), set(), []
    for hit in ranking:
        event_id = hit.get("event_id") if isinstance(hit, dict) else None
        if not isinstance(event_id, str) or event_id in seen_events:
            continue
        seen_events.add(event_id)
        phase = _phase_key(hit)
        if unique and phase in phases:
            deferred.append(hit)
            continue
        unique.append(hit)
        phases.add(phase)
        if len(unique) == MAX_HISTORY_EVENTS:
            return unique
    for hit in deferred:
        unique.append(hit)
        if len(unique) == MAX_HISTORY_EVENTS:
            break
    return unique


def _page_evidence(page):
    return {"kind": "contact_sheet", "id": "contact_sheet", "time": page["cells"][-1]["time_s"],
            "frame_ids": page["frame_ids"], "path": page["path"], "sha256": page["sha256"],
            "rows": 2, "columns": 3}


def finalize_search(question, frames, history, backend):
    """Force an option only after the bounded evidence search is exhausted.

    The final call receives no new retrieval result. It revisits the original
    independent recent frames alongside exactly the pages and cumulative text
    already returned to the controller.
    """
    if not isinstance(history, dict):
        raise ValueError("history must be a mapping")
    if history.get("status") not in ("insufficient_evidence", "invalid_response"):
        raise ValueError("only unresolved history can be finalized")
    frame_meta = _frame_metadata(frames)
    pages = history.get("page_evidence")
    context = history.get("context")
    attempts = history.get("attempts")
    if not isinstance(pages, list) or not isinstance(context, str) or not isinstance(attempts, list):
        raise ValueError("unresolved history lacks auditable evidence")
    prompt, letters = _prompt(question)
    text = f"Video memory already inspected:\n{context}\n\nQuestion: {prompt}"
    raw = backend.choose_option(FORCED_CHOICE_SYSTEM, text,
                                evidence=[*frames, *pages],
                                options=question["options"],
                                stage=f"must-answer:{question['id']}")
    if not isinstance(raw, str) or raw.strip() not in letters:
        raise ValueError("forced choice was not an option letter")
    return {
        **history,
        "status": "answered",
        "prediction": raw.strip(),
        "final_mode": "forced_after_search",
        "final_raw": raw,
        "final_evidence": {
            "recent_frames": frame_meta,
            "history_pages": list(pages),
            "context": context,
        },
    }


def run_question(reader, question, job, frames, *, include_direct_control=True, policy=DEFAULT_POLICY,
                 max_history_tools=DEFAULT_MAX_HISTORY_TOOLS):
    """Run a complete-answer policy while preserving the direct-answer control.

    A classified unavailable history may return the already-recorded direct
    four-frame answer. No other exception is caught here.
    """
    from .agent_retrieval import POLICIES, run_agent_history
    validate_history_tool_budget(max_history_tools)
    if policy not in POLICIES:
        raise ValueError(f"unknown retrieval policy {policy}")
    gate, direct = run_window(question, frames, reader.backend, include_direct_control=include_direct_control)
    if gate["route"] == "short_window":
        return gate, direct
    try:
        history = run_agent_history(reader, question, job, recent_frames=frames, policy=policy,
                                    max_history_tools=max_history_tools, window_thought=gate["thought"])
    except RecoverableMemoryUnavailable as error:
        prediction = direct["prediction"] if direct else None
        final_mode = "direct_when_memory_unavailable"
        final_raw = direct["raw"] if direct else None
        if prediction is None:
            prompt, letters = _prompt(question)
            forced = reader.backend.choose_option(
                FORCED_CHOICE_SYSTEM,
                prompt,
                evidence=list(frames),
                options=question["options"],
                stage=f"must-answer-short-window-unavailable:{question['id']}",
            )
            if not isinstance(forced, str) or forced.strip() not in letters:
                raise ValueError("short-window forced choice was not an option letter")
            prediction = forced.strip()
            final_mode = "forced_short_window_when_memory_unavailable"
            final_raw = forced
        return {
            "id": question["id"],
            "status": "answered",
            "prediction": prediction,
            "route": "history",
            "short_window": gate,
            "final_mode": final_mode,
            "final_raw": final_raw,
            "memory_unavailable": str(error),
        }, direct
    return {**history, "route": "history", "short_window": gate}, direct
