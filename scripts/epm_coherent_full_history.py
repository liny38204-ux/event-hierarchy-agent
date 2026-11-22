"""Load the committed rolling memory for EPM answering without writer or replay inputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from event_memory.merged_retrieval import validate_merged_memory

def _read(path: Path):
    return json.loads(path.read_text())


class HistoryReader:
    """Query the manager-owned E records and write only selected visual pages."""
    def __init__(self, backend, output_root, *, source_root, embedder=None,
                 embedder_factory: Callable[[], Any] | None = None, move_embedder=None):
        self.backend = backend
        self.output_root = Path(output_root).resolve()
        self.source_root, self.embedder = Path(source_root), embedder
        self.embedder_factory, self.move_embedder = embedder_factory, move_embedder

    def _get_embedder(self):
        if self.embedder is None:
            if self.embedder_factory is None:
                raise ValueError("Supply the text embedding model for history retrieval.")
            self.embedder = self.embedder_factory()
        return self.embedder

    def _move(self, embedder, device):
        if self.move_embedder:
            self.move_embedder(embedder, device); return
        import gc
        if device.startswith("cuda") and hasattr(self.backend, "device_index"):
            device = f"cuda:{self.backend.device_index}"
        embedder.model.to(device); embedder.device = device; gc.collect(); self.backend.torch.cuda.empty_cache()

    def build(self, question, job):
        """Open one committed manager snapshot. Historical traces are not inputs."""
        if (question.get("video"), question.get("realtime")) != (job["video"], job["frontier"]):
            raise ValueError("question and job prefix differ")
        key, vid = job["job_id"], job["video_id"]
        if any(not isinstance(v, str) or not v or Path(v).name != v or v in (".", "..") for v in (key, vid)):
            raise ValueError("invalid prefix identity")
        path = self.source_root / "jobs" / key / "build" / "rolling" / vid / "memory.json"
        memory = _read(path)
        if memory.get("video_id") != vid or memory.get("observed_until") != job["frontier"]:
            raise ValueError("manager memory belongs to another video or frontier")
        validate_merged_memory(memory, question["realtime"])
        return memory, self.output_root / "merged_pages" / key, "manager_merged_events"
