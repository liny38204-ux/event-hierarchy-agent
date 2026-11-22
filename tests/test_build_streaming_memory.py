"""Public CLI contract for the question-free v11 video-memory builder."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import epm_build_prefix as PREFIX

SPEC = importlib.util.spec_from_file_location(
    "build_streaming_memory", ROOT / "scripts" / "build_streaming_memory.py"
)
CLI = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CLI)


class Backend:
    """A deterministic stand-in for the local writer/manager backend."""
    receipt = {"kind": "fixture"}

    def __init__(self, _model, _log):
        self.calls = []

    def token_count(self, text):
        return len(text.split())

    def generate(self, system, text, evidence, *, stage, max_tokens):
        self.calls.append({"stage": stage, "finish_reason": "stop"})
        if stage.startswith("write:"):
            return json.dumps({"caption": "一名顾客查看货架上的盒装玩具。"})
        if stage.startswith("rolling-organize:"):
            pending = json.loads(text.split("\n修复错误：")[0])["pending_events"]
            return json.dumps({"groups": [{
                "event_ids": [event["id"] for event in pending],
                "phase_id": "PH1", "phase_title": "玩具货架观察",
                "phase_summary": "顾客查看货架上的盒装玩具。",
                "episode_id": "P1", "episode_title": "商店浏览",
                "episode_summary": "顾客在商店内浏览商品。",
                "caption": "\n".join(event["caption"] for event in pending),
            }], "reason": "fixture"})
        raise AssertionError(stage)


class Encoder:
    receipt = {"kind": "fixture"}

    def __init__(self, _model):
        pass

    def encode(self, _pixels):
        return [1.0, 0.0]


class WriterBudget:
    def __init__(self, backend, *args, **kwargs):
        self.backend = backend

    def fits(self, *args, **kwargs):
        return True

    def generate(self, system, text, evidence, *, stage):
        return self.backend.generate(system, text, evidence, stage=stage, max_tokens=1536)


class Filter:
    audit = []

    def push(self, record, pixels):
        return [(record, pixels)]

    def flush(self):
        return []


def sampler(_video, _frontier, destination, _fps):
    destination.mkdir(parents=True, exist_ok=True)
    for index in (1, 2):
        path = destination / f"f{index:06d}.jpg"
        Image.new("RGB", (16, 16), (index * 20, 0, 0)).save(path)
        yield ({"id": path.stem, "time": float(index), "path": str(path),
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
               np.asarray(Image.open(path).convert("RGB")).copy())


class StreamingMemoryCliTests(unittest.TestCase):
    def test_main_builds_new_output_then_writes_receipt(self):
        originals = (PREFIX.sampled_frames, PREFIX.make_image_filter,
                     PREFIX.QwenCaptionWriter, PREFIX.candidate_batches,
                     CLI.Qwen38Backend, CLI.LocalCLIPEncoder)
        previous_argv = sys.argv
        try:
            PREFIX.sampled_frames = sampler
            PREFIX.make_image_filter = lambda _config: Filter()
            PREFIX.QwenCaptionWriter = WriterBudget
            PREFIX.candidate_batches = lambda records, *_args, **_kwargs: [{
                "id": "S0001", "start": 1.0, "end": 2.0,
                "evidence": records,
            }]
            CLI.Qwen38Backend, CLI.LocalCLIPEncoder = Backend, Encoder
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                video = root / "fixture.mp4"
                video.write_bytes(b"fixture video")
                output = root / "fresh-output"
                sys.argv = ["build_streaming_memory.py", "--video", str(video),
                            "--frontier", "2", "--video-id", "demo-001",
                            "--writer-model", str(root / "writer-model"),
                            "--vision-encoder", str(root / "vision-model"),
                            "--output", str(output)]
                CLI.main()
                self.assertTrue((output / "rolling" / "demo-001" / "memory.json").is_file())
                receipt = json.loads((output / "build_receipt.json").read_text())
                self.assertEqual(receipt["status"], "complete")
                self.assertEqual(receipt["video_id"], "demo-001")
                self.assertTrue((output / "build_receipt.json").read_bytes().endswith(b"\n"))
        finally:
            (PREFIX.sampled_frames, PREFIX.make_image_filter,
             PREFIX.QwenCaptionWriter, PREFIX.candidate_batches,
             CLI.Qwen38Backend, CLI.LocalCLIPEncoder) = originals
            sys.argv = previous_argv
