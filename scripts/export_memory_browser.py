#!/usr/bin/env python3
"""Export current v10 memory as a standalone, offline HTML browser."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from event_memory.buffered_phase import BufferedPhaseStore, SCHEMA


def build_data(run):
    index = json.loads((run / "index.json").read_text())
    if index.get("schema") != SCHEMA or index.get("schema_version") != 10:
        raise ValueError("This exporter requires v10 memory")
    videos = []
    for entry in index["videos"]:
        vid = entry["video_id"]
        bundle = json.loads((run / vid / "memory.json").read_text())
        if bundle.get("schema") != SCHEMA or bundle.get("schema_version") != 10:
            raise ValueError("Mixed memory schemas")
        state = BufferedPhaseStore(bundle["store"]).snapshot()
        for fid, frame in state["frames"].items():
            file = run / vid / "frames" / (fid + ".jpg")
            if hashlib.sha256(file.read_bytes()).hexdigest() != frame["sha256"]:
                raise ValueError("Frame content changed: " + fid)
            with Image.open(file) as source:
                thumb = source.convert("RGB")
                thumb.thumbnail((240, 160))
                buffer = io.BytesIO()
                thumb.save(buffer, format="JPEG", quality=65)
            # Export no local filesystem paths or raw model call logs.
            frame.pop("path")
            frame["image"] = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        videos.append({"video_id": vid, "status": bundle.get("status"), "store": state})
    if not videos:
        raise ValueError("No memory to export")
    return {"schema": SCHEMA, "synthetic": index.get("synthetic", False), "videos": videos}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = build_data(args.run)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = Path(__file__).with_name("memory_browser.template.html").read_text()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace("__MEMORY_DATA__", payload))
    print(json.dumps({"output": str(args.output), "videos": len(data["videos"])}))


if __name__ == "__main__":
    main()
