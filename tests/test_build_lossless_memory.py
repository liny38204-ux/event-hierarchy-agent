"""Offline end-to-end check of scripts/build_lossless_memory.py with scripted models.

Patches the sampler (no real video), the writer runtime and the backend, so the
streaming loop, snapshots, placement reviews and v11 projection run on CPU.
This is an engineering test, not a model-quality check.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


class Encoder:
    def encode(self, pixels):
        return pixels.astype(float).mean(axis=(0, 1)) + 1e-3


class Backend:
    def __init__(self):
        self.calls = []
        self.log_path = None

    def token_count(self, value):
        return len(value) // 4

    def generate(self, system, text, evidence=(), *, stage, max_tokens=0):
        self.calls.append({'stage': stage, 'finish_reason': 'stop', 'text': text})
        n = len(self.calls)
        if stage.startswith('write'):
            return json.dumps({'caption': f'A person moves object {n} onto the counter.',
                               'summary': f'A person handles object {n} in a kitchen.',
                               'changes': [{'object': f'object {n}', 'change': 'placed', 'result': 'on the counter'}],
                               'keywords': ['object', 'kitchen counter']})
        if stage.startswith('summarize'):
            return json.dumps({'phase_summary': 'Someone handles objects at the counter.',
                               'episode_summary': 'Kitchen work.'})
        payload = json.loads(text.split('\nFix')[0])
        out = []
        for k, eid in enumerate(payload['pending_event_ids']):
            first = payload['current'] is None and k == 0
            out.append({'event_id': eid, 'phase': 'new' if first else 'current',
                        'episode': 'new' if first else '', 'phase_title': 'kitchen work' if first else '',
                        'episode_title': 'cooking' if first else '', 'step': 'new', 'step_label': f'step for {eid}'})
        return json.dumps({'placements': out})


class Writer:
    def __init__(self, backend, **_):
        self.backend = backend

    def fits(self, *_):
        return True

    def generate(self, system, text, evidence, *, stage):
        return self.backend.generate(system, text, evidence, stage=stage)


class BuildLosslessE2E(unittest.TestCase):
    def test_streaming_build_snapshots_and_projection(self):
        import build_lossless_memory as builder
        import epm_build_prefix
        from event_memory.merged_retrieval import validate_merged_memory

        tmp = Path(tempfile.mkdtemp())
        images = []
        for i in range(12):
            arr = np.full((64, 64, 3), (i // 4) * 80, np.uint8)
            arr[(i % 8) * 8:(i % 8) * 8 + 8] = 255
            path = tmp / 'src' / f'f{i + 1:06d}.jpg'
            path.parent.mkdir(exist_ok=True)
            Image.fromarray(arr).save(path)
            images.append(path)

        def fake_sample(video, frontier, root, allow_empty=True):
            records = [{'id': p.stem, 'time': float(k), 'path': str(p),
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                       for k, p in enumerate(images) if k <= frontier]
            return records, [{'frame_id': r['id'], 'time': r['time'], 'decision': 'keep_change'} for r in records]

        original_sample, original_writer = epm_build_prefix._sample_and_filter, builder.QwenCaptionWriter
        epm_build_prefix._sample_and_filter, builder.QwenCaptionWriter = fake_sample, Writer
        self.addCleanup(setattr, epm_build_prefix, '_sample_and_filter', original_sample)
        self.addCleanup(setattr, builder, 'QwenCaptionWriter', original_writer)

        video = tmp / 'v.mp4'
        video.write_bytes(b'x')
        out = tmp / 'run'
        backend = Backend()
        result = builder.build(video, 11.0, out, 'demo', backend, Encoder())
        self.assertEqual(result['status'], 'complete')
        projection = json.loads((out / 'build/rolling/demo/memory.json').read_text())
        validate_merged_memory(projection, 11.0)
        v12 = json.loads((out / 'build/v12/demo/memory_v12.json').read_text())['store']
        self.assertEqual(v12['roots'], [])
        self.assertEqual(set(v12['frames']), {p.stem for p in images})
        self.assertTrue(all(p['summary'] for p in v12['phases'].values()))
        # prefix builds stay causal
        short = builder.build(video, 4.0, tmp / 'run4', 'demo', Backend(), Encoder())
        s4 = json.loads(Path(short['memory_v12']).read_text())['store']
        self.assertEqual(max(f['time'] for f in s4['frames'].values()), 4.0)
        # The writer is never shown another event's caption, and the placer never sees captions.
        placer_inputs = [c['text'] for c in backend.calls if c['stage'].startswith('place')]
        self.assertTrue(placer_inputs)
        self.assertTrue(all('onto the counter.' not in t for t in placer_inputs))
        self.assertEqual(len(v12['steps']), len(v12['events']))


if __name__ == '__main__':
    unittest.main()
