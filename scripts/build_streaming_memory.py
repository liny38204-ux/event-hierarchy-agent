#!/usr/bin/env python3
"""Build causal v11 streaming-video memory from one video prefix.

The video, frontier, local model directories and output directory are explicit.
No question, option, answer, benchmark manifest, or future frame is accepted by
this entry point.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from epm_build_prefix import build_prefix
from event_memory.backend_qwen38 import Qwen38Backend
from event_memory.caption_runtime import LocalCLIPEncoder, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frontier", type=float, required=True,
                        help="Last permitted observed video second.")
    parser.add_argument("--video-id", required=True,
                        help="Stable identifier used for memory and frame paths.")
    parser.add_argument("--writer-model", type=Path, required=True,
                        help="Local verified Qwen3.8-27B model directory.")
    parser.add_argument("--vision-encoder", type=Path, required=True,
                        help="Local pretrained CLIP directory used for causal boundaries.")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory. Existing directories are rejected.")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.video_id):
        parser.error("--video-id may contain letters, numbers, dot, underscore and hyphen")
    if args.output.exists():
        parser.error("--output must be a new directory")
    if not args.video.is_file():
        parser.error("--video must be an existing file")

    receipt = {
        "kind": "streaming_memory_build_v11",
        "input_contract": "video prefix only; no QA fields",
        "video": str(args.video.resolve()),
        "video_sha256": sha256_file(args.video),
        "frontier": args.frontier,
        "video_id": args.video_id,
        "writer_model": str(args.writer_model.resolve()),
        "vision_encoder": str(args.vision_encoder.resolve()),
        "training": False,
    }
    try:
        backend = Qwen38Backend(args.writer_model, args.output / "memory_calls.jsonl")
        encoder = LocalCLIPEncoder(args.vision_encoder)
        result = build_prefix(args.video, args.frontier, args.output, args.video_id,
                              backend, encoder, memory_mode="rolling")
        receipt.update(status="complete", backend=backend.receipt,
                       vision_encoder=encoder.receipt, result=result)
    except Exception as exc:
        receipt.update(status="failed", error=repr(exc))
        raise
    finally:
        # build_prefix owns creation of --output.  This keeps its strict
        # "new directory" contract intact while still leaving a receipt next
        # to any partially built causal prefix.
        if args.output.is_dir():
            receipt_path = args.output / "build_receipt.json"
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
