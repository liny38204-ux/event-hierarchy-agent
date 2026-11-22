#!/usr/bin/env python3
"""Causal direct-E replay with a five-unit rolling working buffer (schema11)."""
import argparse
from copy import deepcopy
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from event_memory.rolling_phase import (
    SCHEMA, POLICY, BUFFER_CAPACITY, REVIEW_SECONDS,
    RollingPhaseStore, RollingPhaseManager, RollingReviewFailure, working_buffer,
)
from event_memory.direct_frame_memory import verify_evidence
from event_memory.merged_retrieval import persist_event_pages
from event_memory.backend_qwen38 import transient_memory_calls
from event_memory.export import save, markdown
from build_buffered_phase import ingest, state_hash


def build_video(video, frame_root, output, backend, *, retain_replay=False, visual_frame_groups=None):
    frame_root, output = Path(frame_root), Path(output)
    steps = video['steps']
    # The fresh-video bridge already emits one immutable E per transaction.
    # Reject legacy multi-E/continuation input before any files or inference.
    if retain_replay and (not steps or any(len(s['events']) != 1 or s['events'][0]['op'] != 'new' for s in steps)):
        raise ValueError('rolling input requires exactly one immutable new E per step')
    vid = video['video_id']
    folder = output / vid
    (folder / 'frames').mkdir(parents=True, exist_ok=False)
    frames = {}
    for fid, frame in video['frames'].items():
        source = frame_root / vid / 'frames' / (fid + '.jpg')
        destination = folder / 'frames' / source.name
        if hashlib.sha256(source.read_bytes()).hexdigest() != frame['sha256']:
            raise ValueError('source frame hash mismatch')
        if retain_replay:
            shutil.copy2(source, destination)
        else:
            try:
                os.link(source, destination)
            except OSError as exc:
                # A normal streaming build may place decoded frames and its
                # durable hierarchy on different mounts.  Keep the same
                # verified evidence when hard links are unavailable.
                if exc.errno != errno.EXDEV:
                    raise
                shutil.copy2(source, destination)
        frames[fid] = {**frame, 'path': str(destination.resolve())}
    verify_evidence(list(frames.values()))
    store, manager = RollingPhaseStore(), RollingPhaseManager(backend)
    replay_ids = {}
    steps_seen = max_buffer = review_count = compression_count = 0
    first_call = len(backend.calls)

    def review_at(now, *, force=False):
        # Live manager prompts contain provisional writer text.  Qwen and the
        # exact-cache wrapper therefore emit metrics only for this call.
        if retain_replay:
            return manager.maybe_review(store, now, force=force)
        with transient_memory_calls():
            return manager.maybe_review(store, now, force=force)

    def publish_bundle(bundle):
        temporary = folder / '.memory.json.tmp'
        try:
            save(temporary, bundle)
            temporary.replace(folder / 'memory.json')
        finally:
            temporary.unlink(missing_ok=True)

    def checkpoint(status):
        state = store.snapshot()
        bundle = {'schema': SCHEMA, 'schema_version': 11, 'video_id': vid,
                  'observed_until': state['observed_until'], 'store': state,
                  'status': status, 'manager_state': manager.checkpoint(),
                  'working_buffer_event_ids': working_buffer(state),
                  'provenance': {'policy': POLICY, 'kind': ('causal_direct_E_replay_rolling_phase_compression' if retain_replay else 'streaming_merged_events'),
                                 'note_zh': '每轮管理同时归组和压缩；进行中PH的单个E参与下一轮缓存。旧E删除、帧并集保留；没有问答输入。'}}
        if status == 'complete' and visual_frame_groups is not None:
            bundle['event_pages'] = persist_event_pages(bundle, visual_frame_groups, folder / 'pages')
        publish_bundle(bundle)
        (folder / 'memory_tree.md').write_text(markdown(bundle))

    def record(segment, now, review, before):
        nonlocal max_buffer, review_count, compression_count
        max_buffer = max(max_buffer, len(before))
        review_count += int(review['reviewed'])
        compression_count += sum(bool(r['consumed_event_ids']) for r in review.get('replacements', []))
        if not retain_replay:
            if review['reviewed']:
                checkpoint('running')
            return
        state = store.snapshot()
        entry = {'segment_id': segment, 'now': now, 'review': review,
                 'buffer_event_ids_before': before,
                 'buffer_event_ids_after': working_buffer(state),
                 'buffer_size_before': len(before), 'buffer_size_after': len(working_buffer(state)),
                 'state_sha256': state_hash(state), 'events': len(state['events']),
                 'phases': len(state['phases']), 'episodes': len(state['episodes']),
                 'pending_event_ids': state['roots'], 'frames': len(state['frames'])}
        with (folder / 'trace.jsonl').open('a') as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(json.dumps({k: entry[k] for k in ('segment_id', 'now', 'events', 'phases',
                                                'buffer_size_before', 'buffer_size_after')}, ensure_ascii=False), flush=True)
        checkpoint('running')

    try:
        now = None
        expected_frame_ids = [fid for fid, _ in sorted(
            frames.items(), key=lambda item: (item[1]['time'], item[0]))]
        observed_frame_ids = []
        for step in steps:
            if len(step['events']) != 1 or step['events'][0]['op'] != 'new':
                raise ValueError('rolling input requires exactly one immutable new E per step')
            now = step['now']
            incoming_ids = list(step.get('frame_ids', ()))
            expected_slice = expected_frame_ids[len(observed_frame_ids):len(observed_frame_ids) + len(incoming_ids)]
            if incoming_ids != expected_slice:
                raise ValueError('streaming steps must cover retained frames once in chronological order')
            observed_frame_ids.extend(incoming_ids)
            # Manager completion is synchronous backpressure: another event
            # cannot enter a full buffer while a review is pending/failed.
            if len(working_buffer(store.snapshot())) >= BUFFER_CAPACITY:
                raise ValueError('cannot ingest into a full working buffer')
            ingest(store, step, frames, replay_ids)
            if not retain_replay:
                replay_ids.clear()  # Live arrivals are new events and have no source aliases.
            before = working_buffer(store.snapshot())
            review = review_at(step['now'])
            record(step['segment_id'], step['now'], review, before)
            steps_seen += 1
            del step, review
        if now is None:
            raise ValueError('rolling input is empty')
        before = working_buffer(store.snapshot())
        review = review_at(now, force=True)
        # The reviewed event already contains the caption used by retrieval.
        review['event_narratives'] = []
        record('FINAL_REVIEW_0001', now, review, before)
        state = store.snapshot()
        store.validate_reviewed(state)
        if observed_frame_ids != expected_frame_ids:
            raise ValueError('streaming steps do not cover all retained frames exactly once')
        if set(state['frames']) != set(frames) or {
                f for e in state['events'].values() for f in e['frame_ids']} != set(frames):
            raise ValueError('input frame coverage incomplete')
        checkpoint('complete')
        stats = {'events': len(state['events']), 'phases': len(state['phases']),
                 'episodes': len(state['episodes']), 'pending_events': len(state['roots']),
                 'working_buffer_size': len(working_buffer(state)),
                 'max_buffer_size': max_buffer,
                 'frames': len(frames), 'steps': steps_seen,
                 'closed_phases': len(state['completed_phase_ids']),
                 'open_phases': sum(p['status'] == 'open' for p in state['phases'].values()),
                 'actual_model_calls': len(backend.calls) - first_call,
                 'edited_captions': 0,
                 'review_count': review_count, 'compression_count': compression_count}
        save(folder / 'verification.json', stats)
        return stats
    except Exception:
        if retain_replay:
            checkpoint('failed')
        elif (folder / 'memory.json').is_file():
            # Mark only the committed prefix. The live buffer may contain unpublished writer captions.
            last_commit = json.loads((folder / 'memory.json').read_text())
            last_commit['status'] = 'failed'
            publish_bundle(last_commit)
        raise


def build(input_path, frame_root, output, backend):
    input_path, frame_root, output = Path(input_path), Path(frame_root), Path(output)
    if output.exists():
        raise ValueError('output must be a new directory')
    data = json.loads(input_path.read_text())
    result = {'schema': SCHEMA, 'schema_version': 11, 'status': 'running', 'videos': []}
    save(output / 'index.json', result)
    # The standalone CLI is the deployment path.  It keeps provenance needed
    # to reproduce the input selection, without persisting writer captions or
    # raw manager inputs supplied by a replay-style source file.
    save(output / 'input_manifest.json', {
        'kind': 'streaming_merged_events_input_receipt_v1',
        'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
        'videos': [{'video_id': video['video_id'],
                    'frame_count': len(video.get('frames', {})),
                    'observed_until': max((frame['time'] for frame in video.get('frames', {}).values()), default=None)}
                   for video in data.get('videos', [])],
    })
    save(output / 'config.json', {'schema': SCHEMA, 'schema_version': 11, 'policy': POLICY,
                                 'working_buffer_capacity': BUFFER_CAPACITY,
                                 'review_video_seconds': REVIEW_SECONDS,
                                 'timer_policy': 'since_previous_review; checked_on_new_E_arrival',
                                 'open_compressed_event_counts_toward_capacity': True,
                                 'training': False, 'qa': False, 'global_memory_bounded': False,
                                 'model_receipt': getattr(backend, 'receipt', None),
                                 'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest()})
    try:
        for video in data['videos']:
            # Suppress trace replay and raw model I/O for the normal path.
            # Explicit diagnostic callers can still invoke build_video with
            # retain_replay=True.
            with transient_memory_calls():
                stats = build_video(video, frame_root, output, backend, retain_replay=False)
            result['videos'].append({'video_id': video['video_id'], 'statistics': stats})
            save(output / 'index.json', result)
        result['status'] = 'complete'
        save(output / 'index.json', result)
        return result
    except Exception as exc:
        result.update(status='failed', error=_failure_reason(exc))
        save(output / 'index.json', result)
        raise


def _failure_reason(exc):
    """Persist a stable validation category without serializing model prose."""
    if isinstance(exc, RollingReviewFailure):
        return {'type': type(exc).__name__, 'reason': 'rolling manager failed without changing memory',
                **exc.diagnostic}
    message = str(exc)
    known = (
        'source frame hash mismatch',
        'rolling input requires exactly one immutable new E per step',
        'streaming steps must cover retained frames once in chronological order',
        'streaming steps do not cover all retained frames exactly once',
        'cannot ingest into a full working buffer',
        'rolling input is empty',
        'input frame coverage incomplete',
        'rolling context exceeds cap; no captions silently omitted',
        'rolling manager failed without changing memory',
    )
    for reason in known:
        if message.startswith(reason):
            return {'type': type(exc).__name__, 'reason': reason}
    return {'type': type(exc).__name__, 'reason': 'streaming build validation failed'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--frames-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output exists')
    from event_memory.backend_qwen38 import Qwen38Backend
    backend = Qwen38Backend(args.model, args.output / 'calls.jsonl')
    # Backend creates its log parent eagerly only on generate, so build owns
    # output creation and refuses overwriting prior experiments.
    build(args.input, args.frames_run, args.output, backend)
