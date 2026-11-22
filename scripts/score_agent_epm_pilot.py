#!/usr/bin/env python3
"""Score a completed agent-retrieval pilot after every prediction is frozen."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text())


def _metric(rows):
    total = len(rows)
    return {
        "questions": total,
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / total if total else None,
        "routes": dict(Counter(row["route"] for row in rows)),
        "final_modes": dict(Counter(row.get("final_mode") for row in rows)),
    }


def _cost(answer: dict):
    actions = answer.get("actions", [])
    evidence = answer.get("page_evidence", [])
    runtime = answer.get("runtime", {})
    gate = runtime.get("shared_gate", {})
    history = runtime.get("history", {})
    def total(field):
        parts = [section.get(field, 0) for section in (gate, history)]
        return None if None in parts else sum(parts)
    return {
        "tool_calls": answer.get("tool_calls"),
        "pages": len(evidence),
        "page_cells": sum(len(page.get("frame_ids", [])) for page in evidence),
        "shared_gate_model_calls": int(gate.get("model_calls", 0) or 0),
        "history_model_calls": int(history.get("model_calls", 0) or 0),
        "history_input_tokens": history.get("input_tokens", 0),
        "history_visual_tokens": history.get("visual_tokens", 0),
        "history_other_input_tokens": history.get("other_input_tokens", 0),
        "history_output_tokens": history.get("output_tokens", 0),
        "total_reader_calls": gate.get("model_calls", 0) + history.get("model_calls", 0),
        "total_visual_tokens": total("visual_tokens"),
        "total_other_input_tokens": total("other_input_tokens"),
        "total_output_tokens": total("output_tokens"),
        "total_qa_seconds": gate.get("wall_seconds", 0) + history.get("wall_seconds", 0),
        "history_latency_seconds": float(history.get("latency_seconds", 0) or 0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True,
                        help="Official annotations, read only after inference is complete")
    args = parser.parse_args()
    pilot = args.pilot.resolve()
    receipt = _load(pilot / "pilot_receipt.json")
    methods = receipt["policies"]
    frozen_answers = {}
    checksums = {}
    for evaluation in receipt["evaluations"]:
        for method in methods:
            for identifier in receipt["ids"]:
                path = pilot / evaluation / method / "answers" / f"{identifier}.json"
                if not path.is_file():
                    raise FileNotFoundError("inference is incomplete; do not read labels yet: " + str(path))
                answer = _load(path)
                if answer.get("id") != identifier or answer.get("prediction") is None:
                    raise ValueError("inference result is not a frozen prediction: " + str(path))
                frozen_answers[(evaluation, method, identifier)] = answer
                checksums[str(path.relative_to(pilot))] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Labels are read only after every method/evaluation answer exists.
    manifest = pilot / "predictions-frozen.json"
    if manifest.exists() and _load(manifest) != checksums:
        raise ValueError("answer receipts changed since the scoring freeze")
    if not manifest.exists():
        manifest.write_text(json.dumps(checksums, indent=2) + "\n")
    official = {item["id"]: item for item in _load(args.official) if item.get("task") == "EPM"}
    report = {"kind": "agent_retrieval_epm_pilot_score", "inference_label_access": receipt["label_access"],
              "ids": receipt["ids"], "scores": {}}
    for evaluation in receipt["evaluations"]:
        report["scores"][evaluation] = {}
        for method in methods:
            rows = []
            for identifier in receipt["ids"]:
                answer = frozen_answers[(evaluation, method, identifier)]
                source = official[identifier]
                gold = chr(65 + source["gt"])
                letters = [chr(65 + i) for i in range(len(source["options"]))]
                if answer["prediction"] not in letters or source["options"][source["gt"]].strip() != source["answer"].strip():
                    raise ValueError(f"invalid prediction or annotation for {identifier}")
                rows.append({
                    "id": identifier,
                    "prediction": answer.get("prediction"),
                    "gold": gold,
                    "correct": answer.get("prediction") == gold,
                    "route": answer.get("route"),
                    "final_mode": answer.get("final_mode"),
                    "cost": _cost(answer),
                })
            report["scores"][evaluation][method] = {"metric": _metric(rows), "rows": rows}
    target = pilot / "score.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite score: {target}")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
