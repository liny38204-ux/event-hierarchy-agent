#!/usr/bin/env python3
"""Exercise v10 with generated images and scripted replies; no model inference."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from build_buffered_phase import build


class ScriptedBackend:
    def __init__(self):
        self.calls = []
        self.receipt = {"type": "synthetic_scripted_fixture", "real_model_inference": False}

    def generate(self, system, text, evidence, *, stage, max_tokens=2048):
        payload = json.loads(text)
        if stage.startswith("collapse:"):
            reply = {
                "phase_id": payload["phase"]["id"],
                "event_ids": [e["id"] for e in payload["events"]],
                "caption": "红色方块从左向右移动，三个观察位置按时间排列。",
                "phase_summary": "观察红色方块移动。",
                "support_frame_ids": payload["visible_frame_ids"],
            }
        else:
            pending = payload["pending_event_ids"]

            def group(members, ph, title):
                return {"event_ids": members, "phase_id": ph,
                        "phase_title": title, "phase_summary": title,
                        "episode_id": "P1", "episode_title": "图形展示",
                        "episode_summary": "先展示红色方块，再展示蓝色圆形。"}

            if payload["active_context"] is None:
                assert len(pending) == 5
                groups = [group(pending[:3], "PH1", "红色方块移动"),
                          group(pending[3:], "PH2", "蓝色圆形移动")]
            else:
                assert payload["active_context"]["phase"]["id"] == "PH2"
                groups = [group(pending, "PH2", "蓝色圆形移动")]
            reply = {"groups": groups, "deferred_event_ids": [], "refreshes": [],
                     "reason": "合成测试的固定决策，不是模型视觉判断。"}
        raw = json.dumps(reply, ensure_ascii=False)
        self.calls.append({"stage": stage, "system": system, "text": text,
                           "evidence": evidence, "raw": raw, "seconds": 0,
                           "finish_reason": "stop", "scripted": True})
        return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory")
    frame_root = args.output / "input_frames"
    directory = frame_root / "synthetic" / "frames"
    directory.mkdir(parents=True)
    frames, steps = {}, []
    for i in range(6):
        fid, at = f"f{i + 1}", float(i * 2)
        image = Image.new("RGB", (320, 180), "white")
        draw = ImageDraw.Draw(image)
        x = 30 + (i % 3) * 75
        if i < 3:
            draw.rectangle((x, 60, x + 40, 100), fill="red")
            caption = f"红色方块出现在第{i + 1}个水平位置。"
        else:
            draw.ellipse((x, 60, x + 40, 100), fill="blue")
            caption = f"蓝色圆形出现在第{i - 2}个水平位置。"
        path = directory / (fid + ".jpg")
        image.save(path)
        frames[fid] = {"id": fid, "time": at,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        steps.append({"segment_id": f"S{i + 1}", "now": at, "frame_ids": [fid],
                      "events": [{"source_event_id": f"input-{i + 1}", "op": "new",
                                  "caption": caption, "frame_ids": [fid], "start": at, "end": at}]})
    data = {"kind": "causal_frozen_event_input", "synthetic": True,
            "videos": [{"video_id": "synthetic", "frames": frames, "steps": steps}]}
    manifest = args.output / "events.json"
    manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    backend = ScriptedBackend()
    result = build(manifest, frame_root, args.output / "run", backend)
    # Keep scripted transactions explicitly separate from real inference counts.
    for entry in result["videos"]:
        stats = entry["statistics"]
        stats["scripted_backend_calls"] = stats["actual_model_calls"]
        stats["actual_model_calls"] = 0
        (args.output / "run" / entry["video_id"] / "verification.json").write_text(
            json.dumps(stats, indent=2) + "\n")
    result["synthetic"] = True
    (args.output / "run" / "index.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Synthetic demo complete; no GPU or language model was used.")


if __name__ == "__main__":
    main()
