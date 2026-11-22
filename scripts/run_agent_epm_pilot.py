#!/usr/bin/env python3
"""Run frozen EPM prefixes through agent retrieval policies.

The runner deliberately reads only label-free inference inputs.  Score results
afterwards with ``score_agent_epm_pilot.py`` so development labels can never
leak into the controller prompt or action trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic

from epm_coherent_full_history import HistoryReader
from event_memory.agent_retrieval import POLICIES, run_agent_history
from event_memory.backend_qwen3vl8 import Qwen3VL8BBackend
from event_memory.chat_reader import ChatReader
from event_memory.evidence_search import DECISION_FORMAT, DEFAULT_MAX_HISTORY_TOOLS, DEFAULT_POLICY, run_window, validate_history_tool_budget
from event_memory.short_window import recent_video_frames
from native_qwen_embedding import NativeQwenTextEmbedder
from event_memory.export import save



def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path):
    return json.loads(path.read_text())


def _load_inputs(configs: Path):
    reader = _load(configs / "reader.json")
    jobs = _load(configs / "jobs.json")
    inventory = _load(configs / "video_inventory.json")
    plan = _load(configs / "plan.json")
    questions = {item["id"]: item for item in reader["questions"]}
    jobs_by_question = {
        question_id: job
        for job in jobs["jobs"]
        for question_id in job["question_ids"]
    }
    videos = {item["video"]: item for item in inventory["rows"]}
    return questions, jobs_by_question, videos, plan


def _short_window_result(gate: dict, *, evaluation: str) -> dict:
    if gate["route"] != "short_window" or gate["prediction"] is None:
        raise ValueError("short-window result requires a direct gate prediction")
    return {
        "id": gate["id"],
        "status": "answered",
        "prediction": gate["prediction"],
        "final_mode": "short_window",
        "route": "short_window",
        "evaluation": evaluation,
        "short_window": gate,
        "tool_calls": 0,
        "actions": [],
        "attempts": [],
        "page_evidence": [],
    }


def _write_result(output: Path, evaluation: str, policy: str, question_id: int, result: dict):
    target = output / evaluation / policy / "answers" / f"{question_id}.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite frozen pilot result: {target}")
    save(target, result)


def _call_summary(calls: list[dict], *, elapsed=0.0, merge_size=2) -> dict:
    """Persist call accounting while raw inputs stay in reader.jsonl."""
    visual = None if any(c.get("evidence") and "image_grid_thw" not in c for c in calls) else sum(
        t * h * w // (merge_size ** 2) for call in calls for t, h, w in call.get("image_grid_thw", []))
    inputs = None if any(c.get("input_tokens") is None for c in calls) else sum(c["input_tokens"] for c in calls)
    return {
        "model_calls": len(calls),
        "input_tokens": inputs,
        "visual_tokens": visual,
        "other_input_tokens": inputs - visual if inputs is not None and visual is not None else None,
        "wall_seconds": elapsed,
        "output_tokens": (sum(c["output_tokens"] for c in calls)
                          if all(c.get("output_tokens") is not None for c in calls) else None),
        "latency_seconds": sum(float(call.get("seconds", 0) or 0) for call in calls),
        "stages": [call.get("stage") for call in calls],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument("--answer-model", type=Path, help="Local Qwen3-VL-8B-Instruct checkpoint")
    models.add_argument("--api-model", help="Vision model at CHAT_API_BASE_URL, authenticated by CHAT_API_KEY")
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--id", type=int, nargs="+", required=True)
    parser.add_argument("--policies", choices=POLICIES, nargs="+", default=[DEFAULT_POLICY])
    parser.add_argument("--evaluations", choices=("normal", "forced_history"), nargs="+",
                        default=("normal",),
                        help="Default: normal (answer from recent frames or request history). "
                             "Select forced_history explicitly for a retrieval diagnostic.")
    parser.add_argument("--max-history-tools", type=int, default=DEFAULT_MAX_HISTORY_TOOLS,
                        help="Maximum SEARCH, EXPAND_PHASE, FOCUS_TEXT and INSPECT calls per agent question (default: %(default)s)")
    args = parser.parse_args()
    try:
        validate_history_tool_budget(args.max_history_tools)
    except ValueError as exc:
        parser.error(str(exc))

    args.configs = args.configs.resolve()
    args.output = args.output.resolve()
    questions, jobs, videos, plan = _load_inputs(args.configs)
    selected = list(dict.fromkeys(args.id))
    if not selected or any(identifier not in questions or identifier not in jobs for identifier in selected):
        parser.error("one or more IDs are absent from the frozen question/jobs inputs")
    for identifier in selected:
        question, job = questions[identifier], jobs[identifier]
        if set(question) != {"id", "question", "options", "video", "realtime"}:
            raise ValueError("reader input includes fields outside the label-free contract")
        if (question["video"], question["realtime"]) != (job["video"], job["frontier"]):
            raise ValueError("question and causal memory frontier differ")
        video = videos[question["video"]]
        path = Path(video["path"])
        if path.stat().st_size != video["bytes"]:
            raise ValueError("video inventory byte count differs")

    api = ChatReader(args.api_model, args.output / "reader.jsonl") if args.api_model else None
    receipt = {
        "kind": "agent_retrieval_epm_pilot",
        "ids": selected,
        "max_history_tools": args.max_history_tools,
        "policies": list(dict.fromkeys(args.policies)),
        "evaluations": list(dict.fromkeys(args.evaluations)),
        "configs": str(args.configs),
        "source_root": str(args.source_root.resolve()),
        "answer_model": args.api_model if api else str(args.answer_model.resolve()),
        "api_base_url": api.base_url if api else None,
        "embedding_model": str(args.embedding_model.resolve()),
        "query_prompt": plan["query_prompt"],
        "reader_sha256": _sha256(args.configs / "reader.json"),
        "jobs_sha256": _sha256(args.configs / "jobs.json"),
        "window_policy": "frozen_four_recent_frames",
        "decision_format": DECISION_FORMAT,
        "retrieval_interface": "phase_with_expansion",
        "label_access": "none during inference",
    }
    receipt_path = args.output / "pilot_receipt.json"
    if receipt_path.exists():
        prior = _load(receipt_path)
        prior.setdefault("max_history_tools", 3)  # Older pilots used a hard-coded budget.
        if prior != receipt:
            raise ValueError("pilot receipt mismatch; use a new output directory")
    else:
        save(receipt_path, receipt)

    backend = api or Qwen3VL8BBackend(args.answer_model, args.output / "reader.jsonl", linear_patches=True)
    # The API reports aggregate token usage and owns its visual preprocessing.
    merge_size = 1 if api else int(backend.model.config.vision_config.spatial_merge_size)
    reader = HistoryReader(
        backend, args.output, source_root=args.source_root,
        embedder_factory=lambda: NativeQwenTextEmbedder(
            args.embedding_model, device="cpu", batch_size=8,
            query_prompt=plan["query_prompt"], document_prompt=""),
        move_embedder=(lambda embedder, device: None) if api else None,
    )
    for identifier in selected:
        question, job = questions[identifier], jobs[identifier]
        video = videos[question["video"]]
        frames = recent_video_frames(
            Path(video["path"]), job["frontier"], args.output / "frames" / job["job_id"],
            maximum_frames=4,
        )
        # One gate is shared by every policy and both evaluation modes.  Later
        # runs never invoke an additional direct-control answer call.
        gate_path = args.output / "gates" / f"{identifier}.json"
        if gate_path.exists():
            stored_gate = _load(gate_path)
            gate_runtime = stored_gate.pop("runtime", None)
            gate = stored_gate
            if [v["sha256"] for v in frames] != [v["sha256"] for v in gate["frames"]]:
                raise ValueError("gate frames differ from the original run")
        else:
            before = len(backend.calls)
            start = monotonic()
            gate, _ = run_window(question, frames, backend, include_direct_control=False)
            gate_runtime = _call_summary(backend.calls[before:], elapsed=monotonic()-start, merge_size=merge_size)
            save(gate_path, {**gate, "runtime": gate_runtime})
        for evaluation in dict.fromkeys(args.evaluations):
            for policy in dict.fromkeys(args.policies):
                target = args.output / evaluation / policy / "answers" / f"{identifier}.json"
                if target.exists():
                    continue
                other = "normal" if evaluation == "forced_history" else "forced_history"
                same_route = args.output / other / policy / "answers" / f"{identifier}.json"
                if gate["route"] == "history" and same_route.exists():
                    result = {**_load(same_route), "evaluation": evaluation, "reused_from": other}
                    _write_result(args.output, evaluation, policy, identifier, result)
                    continue
                if evaluation == "normal" and gate["route"] == "short_window":
                    result = _short_window_result(gate, evaluation=evaluation)
                    result["runtime"] = {"shared_gate": gate_runtime, "history": _call_summary([])}
                else:
                    before = len(backend.calls)
                    start = monotonic()
                    window_thought = gate["thought"] if gate["action"] == "SEARCH" else None
                    result = run_agent_history(reader, question, job, recent_frames=frames, policy=policy,
                                               max_history_tools=args.max_history_tools, window_thought=window_thought)
                    result = {**result, "route": "history", "evaluation": evaluation,
                              "short_window": gate,
                              "runtime": {"shared_gate": gate_runtime,
                                          "history": _call_summary(backend.calls[before:], elapsed=monotonic()-start,
                                                                   merge_size=merge_size)}}
                _write_result(args.output, evaluation, policy, identifier, result)


if __name__ == "__main__":
    main()
