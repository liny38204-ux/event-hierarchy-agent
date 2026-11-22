#!/usr/bin/env python3
"""Build a v12 lossless step memory from one causal video prefix.

Same front end as v11 (1 FPS sampling, VAM image filter, past-only CLIP
boundaries).  Each candidate segment becomes one immutable English E written by
the writer; a placement-only manager organizes P -> PH -> Step.  Nothing is
merged or rewritten.

Outputs (under --output):
  build/v12/<video-id>/memory_v12.json     source of truth (schema lossless_step_memory_v12)
  build/rolling/<video-id>/memory.json     v11-compatible projection for the existing agent loop
  build/rolling/<video-id>/frames/         hard links to retained frames
  build/status.json, filter_audit.json, boundary_decisions.json, calls.jsonl
No question, option, answer or future frame is accepted.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

from event_memory.caption_runtime import CaptionTruncated, QwenCaptionWriter, sha256_file
from event_memory.image_filter import VAMImageFilterConfig
from event_memory.lossless_hierarchy import (
    BUFFER_CAPACITY, POLICY, SCHEMA, SCHEMA_VERSION, WRITER, WRITER_REPAIR,
    LosslessManager, LosslessStore, ReviewFailure, parse_writer_output, project_v11_bundle, writer_event)
from event_memory.retained_frontend import candidate_batches

FPS = 1.0
WRITER_CONTEXT_TOKENS = 8192
WRITER_OUTPUT_TOKENS = 1536
VIDEO_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
FIT_PROBE = 'Observed segment S9999: causal retained frames in chronological order.'


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + '\n')
    tmp.replace(path)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class WriteFailure(RuntimeError):
    def __init__(self, sid, errors):
        super().__init__(f'writer failed for {sid}: {errors}')
        self.diagnostic = {'stage': 'write', 'segment': sid, 'attempt_errors': errors}


def write_segment(writer, segment, event_id):
    """One writer call (+ up to two repairs). Returns a validated immutable E."""
    base = (f"Observed segment {segment['id']}: {len(segment['evidence'])} retained frames in chronological order. "
            'Describe only what they show.')
    text, errors = base, []
    for attempt in range(3):
        try:
            raw = writer.generate(WRITER, text, segment['evidence'], stage=f"write:{segment['id']}:{attempt}")
            body, summary, changes, keywords = parse_writer_output(raw)
            return writer_event(event_id, body, summary, changes, keywords, segment['evidence']), attempt + 1
        except CaptionTruncated:
            errors.append('output_truncated')
            text = base + '\n' + WRITER_REPAIR + 'output was too long; describe repeated static details once.'
        except ValueError as exc:
            errors.append(str(exc)[:200])
            text = base + '\n' + WRITER_REPAIR + str(exc)[:300]
        if not writer.fits(WRITER, text, segment['evidence']):
            text = base
    raise WriteFailure(segment['id'], errors)


def build(video_path, frontier, output, video_id, backend, encoder):
    """Stream one causal prefix ending at `frontier` (one build per question time, as in v11).

    The candidate segmenter only closes its last segment at the frontier, so a
    query-time memory must come from a prefix that ends exactly there.
    """
    from epm_build_prefix import _sample_and_filter  # same sampling + VAM gates as v11

    video_path, output, frontier = Path(video_path).resolve(), Path(output).resolve(), float(frontier)
    if not video_path.is_file():
        raise ValueError('video must be an existing file')
    if not math.isfinite(frontier) or frontier <= 0:
        raise ValueError('frontier must be positive')
    if not VIDEO_ID.fullmatch(video_id):
        raise ValueError('invalid video id')
    if output.exists():
        raise ValueError('output must be a new directory')
    build_dir = output / 'build'
    build_dir.mkdir(parents=True)
    status = {'status': 'starting', 'schema': SCHEMA, 'policy': POLICY, 'video_id': video_id,
              'frontier': frontier, 'fps': FPS, 'video_sha256': _hash(video_path),
              'input_contract': 'video prefix only; no questions, options or labels'}
    _save(build_dir / 'status.json', status)
    if hasattr(backend, 'log_path'):
        backend.log_path = build_dir / 'calls.jsonl'
    t0 = time.monotonic()
    try:
        records, audit = _sample_and_filter(video_path, frontier, build_dir / 'sampled_frames', allow_empty=True)
        _save(build_dir / 'filter_audit.json', {'config': VAMImageFilterConfig().__dict__, 'decisions': audit})
        if not records:
            raise ValueError('VAM image filter retained no frame for this prefix')
        rolling_dir = build_dir / 'rolling' / video_id
        frame_dir = rolling_dir / 'frames'
        frame_dir.mkdir(parents=True)
        frames = {}
        for r in records:
            target = frame_dir / (r['id'] + '.jpg')
            os.link(Path(r['path']).resolve(), target)
            frames[r['id']] = {'id': r['id'], 'time': float(r['time']), 'path': str(target.resolve()),
                               'sha256': r['sha256']}
        writer = QwenCaptionWriter(backend, context_limit=WRITER_CONTEXT_TOKENS, output_tokens=WRITER_OUTPUT_TOKENS)
        boundary_audit = []
        batches = candidate_batches([{k: r[k] for k in ('id', 'time', 'path')} for r in records], encoder,
                                    frontier, audit=boundary_audit,
                                    fits=lambda images: writer.fits(WRITER, FIT_PROBE, images))
        store, manager = LosslessStore(), LosslessManager(backend)
        stats = {'segments': 0, 'writer_repairs': 0, 'reviews': 0, 'place_retries': 0}

        def review(now, force=False):
            result = manager.maybe_review(store, now, force=force)
            if result['reviewed']:
                stats['reviews'] += 1
                stats['place_retries'] += result['attempts'] - 1

        for index, candidate in enumerate(batches, 1):
            sid = f'S{index:04d}'
            now = float(candidate.get('closed_at', candidate['end']))
            if len(store.snapshot()['roots']) >= BUFFER_CAPACITY:
                review(store.snapshot()['observed_until'], force=True)
            segment = {'id': sid, 'start': candidate['start'], 'end': candidate['end'],
                       'evidence': [{**e, 'sha256': frames[e['id']]['sha256']} for e in candidate['evidence']]}
            event, attempts = write_segment(writer, segment, f"E{store.snapshot()['next_event_id']}")
            stats['segments'] += 1
            stats['writer_repairs'] += attempts - 1
            store.add_event(event, [frames[e['id']] for e in candidate['evidence']], now)
            review(now)
        if frontier > store.snapshot()['observed_until']:
            store.advance(frontier)
        review(frontier, force=True)
        final_state = manager.finalize_snapshot(store, frontier)
        if set(f for e in final_state['events'].values() for f in e['frame_ids']) != set(frames):
            raise RuntimeError('frame coverage incomplete')
        v12_path = build_dir / 'v12' / video_id / 'memory_v12.json'
        _save(v12_path, {'schema': SCHEMA, 'schema_version': SCHEMA_VERSION, 'video_id': video_id,
                         'observed_until': frontier, 'status': 'complete', 'store': final_state,
                         'manager_calls': manager.calls})
        projection = project_v11_bundle(final_state, video_id, provenance={'v12_memory': str(v12_path)})
        _save(rolling_dir / 'memory.json', projection)
        _save(build_dir / 'boundary_decisions.json', {'frontier': frontier, 'decisions': boundary_audit})
        metrics = {**stats, 'events': len(final_state['events']), 'steps': len(final_state['steps']),
                   'phases': len(final_state['phases']), 'episodes': len(final_state['episodes']),
                   'retained_frames': len(records), 'sampled_frames': len(audit),
                   'manager_calls': manager.calls, 'seconds': round(time.monotonic() - t0, 1)}
        status.update(status='complete', metrics=metrics, memory_v12=str(v12_path),
                      memory_v11_projection=str(rolling_dir / 'memory.json'))
        _save(build_dir / 'status.json', status)
        return status
    except Exception as exc:
        diag = getattr(exc, 'diagnostic', None)
        status.update(status='failed', error=repr(exc)[:500], **({'failure': diag} if diag else {}))
        _save(build_dir / 'status.json', status)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--frontier', type=float, required=True)
    ap.add_argument('--video-id', required=True)
    ap.add_argument('--writer-model', type=Path, required=True)
    ap.add_argument('--vision-encoder', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    from event_memory.backend_qwen38 import Qwen38Backend
    from event_memory.caption_runtime import LocalCLIPEncoder
    backend = Qwen38Backend(a.writer_model, a.output / 'build' / 'calls.jsonl')
    encoder = LocalCLIPEncoder(a.vision_encoder)
    result = build(a.video, a.frontier, a.output, a.video_id, backend, encoder)
    print(json.dumps(result.get('metrics', {}), ensure_ascii=False))


if __name__ == '__main__':
    main()
