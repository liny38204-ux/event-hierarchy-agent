#!/usr/bin/env python3
"""Freeze label-free OVO MC inputs for every task except FPD.

REC, SSR and CRR have count/binary active-response probes. They are retained
in a manifest for a dedicated adapter, never coerced into an MC reader job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

MC = {"EPM", "HLD", "ASI", "STU", "OJR", "ATR", "ACR", "OCR"}
ACTIVE = {"REC", "SSR", "CRR"}


def dump(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-fpd", action="store_true",
                        help="include FPD in the MC inventory for a separately selectable cohort")
    args = parser.parse_args()
    rows = json.loads(args.annotations.read_text())
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")

    reader, scorer, active = [], [], []
    counts = Counter()
    for row in rows:
        task = row.get("task")
        if task == "FPD" and not args.include_fpd:
            continue
        if task in MC or task == "FPD":
            required = {"id", "video", "realtime", "question", "options", "answer", "gt"}
            if not required <= row.keys():
                raise ValueError(f"bad {task} item {row.get('id')}")
            question = {key: row[key] for key in ("id", "video", "realtime", "question", "options")}
            if not isinstance(question["id"], int) or not isinstance(question["options"], list):
                raise ValueError(f"bad inference fields {question['id']}")
            reader.append(question)
            scorer.append({key: row[key] for key in ("id", "task", "gt", "answer")})
            counts[task] += 1
        elif task in ACTIVE:
            active.append({"id": row["id"], "task": task, "video": row["video"],
                           "query_count": len(row.get("test_info", []))})
        else:
            raise ValueError(f"unexpected OVO task {task!r}")
    reader.sort(key=lambda q: q["id"])
    scorer.sort(key=lambda q: q["id"])

    grouped = defaultdict(list)
    for question in reader:
        grouped[(question["video"], float(question["realtime"]))].append(question["id"])
    jobs = [{"job_id": f"ovo-{min(ids)}", "video_id": f"ovo-{min(ids)}", "video": video,
             "frontier": time, "question_ids": sorted(ids)}
            for (video, time), ids in sorted(grouped.items())]
    inventory = []
    for video in sorted({question["video"] for question in reader}):
        path = (args.video_root / video).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing video: {path}")
        inventory.append({"video": video, "path": str(path), "bytes": path.stat().st_size})

    dump(out / "reader.json", {"kind": "ovo_non_fpd_mc_reader_no_labels", "questions": reader})
    dump(out / "jobs.json", {"kind": "video_only_causal_prefix_jobs", "jobs": jobs})
    dump(out / "video_inventory.json", {"kind": "ovo_video_inventory", "rows": inventory})
    dump(out / "scorer_only.json", {"kind": "ovo_non_fpd_mc_scorer_only", "labels": scorer})
    dump(out / "active_response_manifest.json", {"kind": "ovo_active_response_requires_dedicated_adapter", "rows": active})
    dump(out / "plan.json", {"kind": "ovo_non_fpd_mc", "question_count": len(reader),
        "unique_prefixes": len(jobs), "video_count": len(inventory), "task_counts": dict(sorted(counts.items())),
        "excluded_task": None if args.include_fpd else "FPD", "query_prompt": "Instruct: Given a question about past events in a video, retrieve captions that answer the question.\nQuery: ",
        "reader_sha256": digest(out / "reader.json"), "jobs_sha256": digest(out / "jobs.json"),
        "active_response_note": "REC, SSR and CRR require a dedicated count/binary adapter."})


if __name__ == "__main__":
    main()
