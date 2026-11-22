#!/usr/bin/env python3
"""Build one question-free, causal memory from one fresh video prefix."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import build_buffered_phase as buffered_builder
import build_rolling_phase as rolling_builder
# Keep this pipeline self-contained. The previous experiment-only import
# pulled in the legacy hierarchy runner and its unrelated private manifests.
LANGUAGE = ('\\nWrite captions, summaries and continuation reasons in Simplified Chinese. '
            'Keep JSON keys, IDs and operation names unchanged.')
from event_memory.caption_runtime import CaptionTruncated, QwenCaptionWriter, sampled_frames, sha256_file
from event_memory.caption_hierarchy import caption_segment
from event_memory.image_filter import VAMImageFilterConfig, make_image_filter
from event_memory import prompts_caption_hierarchy as caption_prompts
from event_memory.retained_frontend import candidate_batches
from event_memory.timed_caption import caption_blocks, time_header
from event_memory.backend_qwen38 import transient_memory_calls
from event_memory.rolling_phase import RollingReviewFailure

FPS = 1.0
WRITER_CONTEXT_TOKENS = 8192
WRITER_OUTPUT_TOKENS = 1536
VIDEO_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
MEMORY_MODES = {'buffered', 'rolling'}


def _timed_writer_caption(body, evidence):
    """Attach an evidence-window header without asking the writer to infer time.

    A header bounds *displayed observations*, rather than the unobserved onset
    or offset of an action.  The writer is deliberately restricted to the body
    so an LLM cannot fabricate a more precise-looking interval.
    """
    if not isinstance(body, str) or not (body := body.strip()):
        raise ValueError('writer caption body must be a nonempty string')
    if re.search(r'<\s*\d+(?:\.\d+)?\s*s\s*-\s*\d+(?:\.\d+)?\s*s\s*>', body):
        raise ValueError('writer caption body must not emit a time header')
    try:
        times = [float(item['time']) for item in evidence]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('writer evidence needs numeric observation times') from exc
    if not times or not all(math.isfinite(value) and value >= 0 for value in times):
        raise ValueError('writer evidence needs finite nonnegative observation times')
    caption = time_header(min(times), max(times)) + '\n' + body
    caption_blocks(caption)  # Reject malformed headers while the writer can still repair them.
    return caption


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_arguments(video_path, frontier, output, video_id, backend, encoder, memory_mode):
    if not Path(video_path).is_file():
        raise ValueError('video_path must be an existing regular file')
    if isinstance(frontier, bool) or not isinstance(frontier, (int, float)) or not math.isfinite(float(frontier)) or float(frontier) <= 0:
        raise ValueError('frontier must be a positive finite number')
    if not isinstance(video_id, str) or not VIDEO_ID.fullmatch(video_id):
        raise ValueError('video_id may contain letters, numbers, dot, underscore and hyphen')
    if Path(output).exists():
        raise ValueError('prefix output must be a new directory')
    if not hasattr(backend, 'generate') or not hasattr(backend, 'token_count'):
        raise TypeError('backend must provide generate and token_count')
    if not hasattr(encoder, 'encode'):
        raise TypeError('encoder must provide encode')
    if memory_mode not in MEMORY_MODES:
        raise ValueError(f'memory_mode must be one of {sorted(MEMORY_MODES)}')


class NoRetainedFrames(ValueError):
    """All sampling/filtering decisions are available in the build audit."""


class CaptionWriteFailure(RuntimeError):
    def __init__(self, sid, errors):
        super().__init__(f'caption writer failed for {sid} after one repair: ' + ', '.join(errors))
        self.diagnostic = {'stage': 'write', 'segment': sid, 'attempt_errors': errors}


def _sample_and_filter(video_path, frontier, sampled_root, *, allow_empty=False):
    """Independent 1-FPS prefix sampling and the current VAM image gates."""
    filter_ = make_image_filter(VAMImageFilterConfig())
    retained = []
    for record, pixels in sampled_frames(video_path, frontier, sampled_root, FPS):
        if record['time'] > frontier:
            raise ValueError('sampled a future frame')
        retained.extend(record for record, _ in filter_.push(record, pixels))
    retained.extend(record for record, _ in filter_.flush())
    if not retained and not allow_empty:
        raise NoRetainedFrames('VAM image filter retained no frame for this causal prefix')
    previous = -1.0
    for record in retained:
        if not previous < record['time'] <= frontier:
            raise ValueError('retained records are not strictly causal and ordered')
        if sha256_file(record['path']) != record['sha256']:
            raise ValueError('retained frame hash mismatch')
        previous = record['time']
    return retained, deepcopy(filter_.audit)



@dataclass
class PreparedSamples:
    video: Path
    frontier: float
    root: Path
    retained: list
    decisions: list


def prepare_prefix_samples(video_path, frontier, root):
    """Sample once for both the near-window reader and optional history build."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    video = Path(video_path).resolve()
    retained, decisions = _sample_and_filter(video, frontier, root / 'sampled_frames', allow_empty=True)
    _save(root / 'filter_audit.json', {'config': VAMImageFilterConfig().__dict__, 'decisions': decisions})
    return PreparedSamples(video, frontier, root, retained, decisions)


def _link_source_frames(records, source_root, video_id):
    """The selected hierarchy adapter gets hardlinks, never duplicate copies."""
    destination = Path(source_root) / video_id / 'frames'
    destination.mkdir(parents=True, exist_ok=False)
    for record in records:
        source = Path(record['path']).resolve()
        target = destination / (record['id'] + '.jpg')
        os.link(source, target)
        if _hash(source) != _hash(target):
            raise RuntimeError('hard-linked frame hash differs from source')


@contextmanager
def _hardlink_copy(builder):
    """Use a hierarchy builder unchanged except its local image copy is a link."""
    original = builder.shutil.copy2

    def link_copy(source, destination, *args, **kwargs):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError('hierarchy target frame already exists')
        os.link(Path(source), destination)
        return str(destination)

    builder.shutil.copy2 = link_copy
    try:
        yield
    finally:
        builder.shutil.copy2 = original


def _write_caption(writer, system, segment, *, timed):
    sid = segment['id']
    base = (f'Observed segment {sid}: causal retained frames are supplied in chronological order. '
            'Describe only what these frames show; the program assigns their observation window.'
            if timed else f"Observed segment {sid}: [{segment['start']:.3f}, {segment['end']:.3f}] seconds.")
    text = base
    issues = []
    for attempt in range(2):
        try:
            raw = writer.generate(system, text, segment['evidence'], stage=f'write:{sid}:{attempt}')
        except CaptionTruncated:
            issues.append('output_truncated')
            repair = ('The previous caption exceeded the output budget. Start fresh. Summarize repeated '
                      'static details once and keep distinct observed actions in order. Return only '
                      'one concise JSON object with a single caption string.')
        else:
            try:
                record, _ = caption_segment(raw, segment)
                if timed:
                    observation = record['observations'][0]
                    observation['caption'] = _timed_writer_caption(observation['caption'], segment['evidence'])
                break
            except ValueError as exc:
                issues.append('invalid_json' if isinstance(exc, json.JSONDecodeError) else 'invalid_caption')
                repair = caption_prompts.REPAIR + '\nValidation error: ' + str(exc)
        if attempt == 0:
            text = base + '\n' + repair
            if hasattr(writer, 'fits') and not writer.fits(system, text, segment['evidence']):
                # The original candidate already fit. Spend the same visual budget on a shorter request.
                text = 'Return one concise JSON caption grounded in these frames.'
                if not writer.fits(system, text, segment['evidence']):
                    raise ValueError(f'writer repair exceeds input budget for {sid}')
    else:
        raise CaptionWriteFailure(sid, issues)
    return record['observations'][0], raw, attempt + 1


def _writer_steps(writer, batches, frame_root, video_id, frontier, *, timed, frames, statistics):
    """Yield the current observation so a rolling manager can consume it immediately."""
    writer_prompt = caption_prompts.TIMED_WRITER if timed else caption_prompts.WRITER
    previous = -1.0
    for index, candidate in enumerate(batches, 1):
        sid = f'S{index:04d}'
        segment = {'id': sid, 'start': candidate['start'], 'end': candidate['end'],
                   'evidence': list(candidate['evidence'])}
        now = float(candidate.get('closed_at', segment['end']))
        if not previous <= now <= frontier or segment['end'] > now:
            raise ValueError('candidate caption frontier is outside the observed prefix')
        evidence_times = [float(item['time']) for item in segment['evidence']]
        if (not evidence_times or evidence_times != sorted(evidence_times)
                or evidence_times[0] < segment['start'] or evidence_times[-1] > segment['end']
                or evidence_times[-1] > now):
            raise ValueError('candidate evidence is not ordered and fully causal before writer call')
        caption, raw, attempts = _write_caption(writer, writer_prompt + LANGUAGE, segment, timed=timed)
        frame_ids = list(caption['evidence_ids'])
        if frame_ids != [item['id'] for item in segment['evidence']]:
            raise RuntimeError('caption E does not retain all candidate frames in order')
        incoming = []
        for evidence in segment['evidence']:
            path = Path(frame_root) / video_id / 'frames' / (evidence['id'] + '.jpg')
            if not path.is_file():
                raise ValueError('source frame missing for direct E transaction')
            item = {'id': evidence['id'], 'time': evidence['time'], 'sha256': _hash(path)}
            prior = frames.setdefault(item['id'], item)
            if prior != item:
                raise ValueError('frame provenance changed across direct E transactions')
            incoming.append(item['id'])
        step = {'segment_id': sid, 'now': now, 'frame_ids': incoming,
                      'events': [{'source_event_id': sid, 'op': 'new', 'caption': caption['caption'],
                                  'frame_ids': frame_ids, 'start': segment['start'], 'end': segment['end']}],
                      'caption_raw_sha256': hashlib.sha256(raw.encode()).hexdigest()}
        statistics.append({'segment_id': sid, 'writer_frame_ids': frame_ids,
                           'attempts': attempts, 'caption_sha256': hashlib.sha256(caption['caption'].encode()).hexdigest()})
        previous = now
        yield step
    if previous != frontier:
        raise ValueError('direct caption source did not reach the requested frontier exactly')


def _caption_statistics(writer_steps):
    return {'writer_segments': len(writer_steps), 'writer_repaired_segments': sum(s['attempts'] == 2 for s in writer_steps),
            'writer_steps': writer_steps}


def _captions_to_immutable_events(writer, batches, frame_root, video_id, frontier, *, timed=False):
    """Materialize input for the older buffered replay adapter."""
    frames, writer_steps = {}, []
    steps = list(_writer_steps(writer, batches, frame_root, video_id, frontier, timed=timed,
                               frames=frames, statistics=writer_steps))
    return ({'kind': 'epm_direct_caption_immutable_e_v1',
             'policy': 'one_candidate_caption_to_one_immutable_new_E; hierarchy_organization_delegated',
             'videos': [{'video_id': video_id, 'frames': frames, 'steps': steps}]},
            _caption_statistics(writer_steps))


def _validate_direct_e_coverage(buffered_input, records):
    """Require the direct-E bridge to cover every retained frame exactly once."""
    videos = buffered_input.get('videos')
    if not isinstance(videos, list) or len(videos) != 1:
        raise ValueError('direct E input must contain exactly one video')
    actual = [fid for step in videos[0].get('steps', []) for fid in step.get('frame_ids', [])]
    expected = [record['id'] for record in records]
    if actual != expected:
        raise RuntimeError('direct caption E frame coverage differs from retained frames')


def _validate_memory(memory_path, frontier, memory_mode):
    bundle = json.loads(Path(memory_path).read_text())
    expected = (('buffered_phase_memory_v10', 10) if memory_mode == 'buffered'
                else (rolling_builder.SCHEMA, 11))
    if (bundle.get('schema'), bundle.get('schema_version')) != expected:
        raise RuntimeError(f'{memory_mode} builder returned the wrong schema')
    if bundle.get('status') != 'complete' or bundle.get('observed_until') != frontier:
        raise RuntimeError(f'{memory_mode} build did not complete at the requested frontier')
    state = bundle['store']
    if state['observed_until'] != frontier or not state['events']:
        raise RuntimeError(f'{memory_mode} memory is incomplete or lacks events')
    return bundle


def build_prefix(video_path, frontier, output, video_id, backend, encoder, memory_mode='rolling', *, prepared_samples=None):
    """Build an independent selected-mode memory; inputs exclude all QA data."""
    video_path, output, frontier = Path(video_path).resolve(), Path(output).resolve(), float(frontier)
    _validate_arguments(video_path, frontier, output, video_id, backend, encoder, memory_mode)
    output.mkdir(parents=True)
    status = {'status': 'starting', 'video_id': video_id, 'frontier': frontier,
              'input_contract': 'video_path, frontier, video_id only; no questions/options/labels/answers',
              'fps': FPS, 'image_filter': 'VAMImageFilterConfig',
              'writer_context_tokens': WRITER_CONTEXT_TOKENS,
              'writer_output_tokens': WRITER_OUTPUT_TOKENS,
              'memory_mode': memory_mode,
              'video_sha256': _hash(video_path)}
    _save(output / 'status.json', status)
    try:
        if prepared_samples is None:
            records, filter_audit = _sample_and_filter(video_path, frontier, output / 'sampled_frames', allow_empty=True)
        else:
            if (prepared_samples.video, prepared_samples.frontier) != (video_path, frontier):
                raise ValueError('prepared samples belong to a different video prefix')
            records, filter_audit = prepared_samples.retained, prepared_samples.decisions
            (output / 'sampled_frames').symlink_to(
                os.path.relpath(prepared_samples.root / 'sampled_frames', output), target_is_directory=True)
        _save(output / 'filter_audit.json', {'config': VAMImageFilterConfig().__dict__,
                                             'decisions': filter_audit})
        if not records:
            raise NoRetainedFrames('VAM image filter retained no frame for this causal prefix')
        source_frames = output / 'source_frames'
        _link_source_frames(records, source_frames, video_id)
        writer = QwenCaptionWriter(backend, context_limit=WRITER_CONTEXT_TOKENS,
                                   output_tokens=WRITER_OUTPUT_TOKENS)
        boundary_audit = []

        timed_writer = memory_mode == 'rolling'
        writer_prompt = caption_prompts.TIMED_WRITER if timed_writer else caption_prompts.WRITER

        def fits(images):
            text = ('Observed segment S9999: causal retained frames are supplied in chronological order. '
                    'Describe only what these frames show; the program assigns their observation window.'
                    if timed_writer else 'Observed segment S9999: causal retained evidence; timestamps follow.')
            return writer.fits(writer_prompt + LANGUAGE, text, images)

        # Candidate construction sees only id/time/path records.  Keep SHA256
        # in the prefix manifest and hierarchy adapter, never in the writer prompt.
        writer_records = [{key: record[key] for key in ('id', 'time', 'path')}
                          for record in records]
        batches = candidate_batches(writer_records, encoder, frontier, audit=boundary_audit, fits=fits)
        if memory_mode == 'rolling':
            memory_root = output / 'rolling'
            writer_steps = []
            # Frame metadata is independent of writer prose. Only the current
            # transaction is yielded; consumed captions never enter a replay file.
            frames = {r['id']: {k: r[k] for k in ('id', 'time', 'sha256')} for r in records}
            steps = _writer_steps(writer, batches, source_frames, video_id, frontier,
                                  timed=True, frames=frames, statistics=writer_steps)
            if hasattr(backend, 'log_path'):
                backend.log_path = memory_root / 'calls.jsonl'
            with transient_memory_calls():
                rolling_builder.build_video(
                    {'video_id': video_id, 'steps': steps, 'frames': frames}, source_frames, memory_root, backend,
                    retain_replay=False, visual_frame_groups=(s['writer_frame_ids'] for s in writer_steps))
            caption_statistics = _caption_statistics(writer_steps)
        else:
            memory_root = output / 'v10'
            caption_root = output / 'caption_direct_e'
            if hasattr(backend, 'log_path'):
                backend.log_path = caption_root / 'calls.jsonl'
            buffered_input, caption_statistics = _captions_to_immutable_events(
                writer, batches, source_frames, video_id, frontier)
            _validate_direct_e_coverage(buffered_input, records)
            _save(caption_root / 'input_manifest.json', {
                'kind': 'fresh_epm_prefix_video_only', 'fps': FPS,
                'videos': [{'video_id': video_id, 'frontier': frontier,
                            'frames': [{'id': r['id'], 'time': r['time'], 'sha256': r['sha256']}
                                       for r in records]}]})
            _save(caption_root / 'config.json', {'schema': 'epm_direct_caption_immutable_e', 'schema_version': 1,
                                                 'episode_policy': 'buffered_hierarchy_direct_immutable_E_v1',
                                                 'memory_mode': memory_mode,
                                                 'fresh_visual_writer': True, 'training': False,
                                                 'qa': False, 'source_video_sha256': status['video_sha256']})
            _save(output / 'buffered_input.json', buffered_input)
            if hasattr(backend, 'log_path'):
                backend.log_path = memory_root / 'calls.jsonl'
            with _hardlink_copy(buffered_builder):
                buffered_builder.build_video(buffered_input['videos'][0], source_frames, memory_root, backend)
        _save(output / 'boundary_decisions.json', {'frontier': frontier, 'decisions': boundary_audit})
        memory_path = memory_root / video_id / 'memory.json'
        memory_bundle = _validate_memory(memory_path, frontier, memory_mode)
        state = memory_bundle['store']
        metrics = {'sampled_frames': len(filter_audit), 'retained_frames': len(records),
                   'candidate_batches': caption_statistics['writer_segments'], 'boundary_decisions': len(boundary_audit),
                   'caption_statistics': caption_statistics,
                   'caption_policy': 'one_candidate_caption_to_one_immutable_new_E; no_local_manage',
                   'memory_mode': memory_mode,
                   'memory_schema': memory_bundle['schema'],
                   'memory_schema_version': memory_bundle['schema_version'],
                   'memory_events': len(state['events']),
                   'memory_phases': len(state['phases']),
                   'memory_episodes': len(state['episodes']),
                   'memory_pending_events': len(state['roots'])}
        if memory_mode == 'buffered':
            metrics.update(v10_events=metrics['memory_events'], v10_phases=metrics['memory_phases'],
                           v10_episodes=metrics['memory_episodes'],
                           v10_pending_events=metrics['memory_pending_events'])
        status.update(status='complete', memory_path=str(memory_path), metrics=metrics,
                      memory_sha256=_hash(memory_path))
        _save(output / 'status.json', status)
        return {'memory_path': str(memory_path), 'metrics': metrics,
                'status_path': str(output / 'status.json')}
    except Exception as exc:
        if isinstance(exc, (RollingReviewFailure, CaptionWriteFailure)):
            reason = str(exc).split(':', 1)[0]
            status.update(status='failed', error=reason, failure=exc.diagnostic)
            _save(output / 'status.json', status)
            if isinstance(exc, CaptionWriteFailure):
                raise
            # Outer worker tracebacks must not archive provisional model text.
            raise ValueError(reason) from None
        status.update(status='failed', error=repr(exc))
        _save(output / 'status.json', status)
        raise
