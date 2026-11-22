"""Caption-only, causal front end. Candidate boundaries are not action labels.

Legacy conservative filter and content-change detection. Current builders use
image_filter.VAMImageFilter by default. No duration cap, frame quota per segment,
or post-segmentation uniform selection.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import re

import numpy as np
from PIL import Image

from .caption_style import CAPTION_STYLE_EN


@dataclass(frozen=True)
class FilterConfig:
    # Conservative starting settings, NOT benchmark-tuned VAM defaults.
    blur_min: float = 0.0  # Measure blur; do not drop fast-motion evidence by default.
    duplicate_mad_max: float = 1.0
    duplicate_ssim_min: float = 0.995
    duplicate_pixel_max: float = 8.0  # Protect small localized changes.


class FrameFilter:
    """Compare to the last retained image, never to a global duplicate bank.

    Keep the endpoint of a duplicate run, so a long static span retains its
    latest observation. All rejection/defer/endpoint decisions are auditable.
    """
    def __init__(self, config=FilterConfig()):
        values = asdict(config)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Filter settings must be finite")
        if min(config.blur_min, config.duplicate_mad_max, config.duplicate_pixel_max) < 0:
            raise ValueError("Negative filtering threshold")
        if not -1 <= config.duplicate_ssim_min <= 1:
            raise ValueError("Invalid SSIM threshold")
        self.config = config
        self.previous = None
        self.pending = None
        self.audit = []
        self.last_time = -1.0
        self.ids = set()

    @staticmethod
    def _ssim(a, b):
        # Local 7x7 population statistics on 64x64 luminance, range 0..255.
        def windows(x):
            small = np.asarray(Image.fromarray(x.astype(np.uint8)).resize((64, 64)), dtype=float)
            return np.lib.stride_tricks.sliding_window_view(small, (7, 7))
        x, y = windows(a), windows(b)
        mx, my = x.mean((-1, -2)), y.mean((-1, -2))
        vx = np.maximum(0, (x*x).mean((-1, -2)) - mx*mx)
        vy = np.maximum(0, (y*y).mean((-1, -2)) - my*my)
        cov = (x*y).mean((-1, -2)) - mx*my
        return float(np.mean(((2*mx*my + 6.5025) * (2*cov + 58.5225)) /
                             ((mx*mx + my*my + 6.5025) * (vx + vy + 58.5225))))

    def push(self, record, rgb):
        t = float(record['time'])
        if not math.isfinite(t) or t <= self.last_time or record['id'] in self.ids:
            raise ValueError("Sample IDs must be unique and times strictly increasing")
        pixels = np.asarray(rgb, dtype=float)
        if pixels.ndim != 3 or pixels.shape[-1] != 3 or min(pixels.shape[:2]) < 3:
            raise ValueError("Expected an RGB image at least 3x3")
        if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 255:
            raise ValueError("Invalid RGB values")
        self.last_time = t
        self.ids.add(record['id'])
        gray = pixels @ np.array([0.299, 0.587, 0.114])
        lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4*gray[1:-1, 1:-1]
        entry = {'frame_id': record['id'], 'time': t, 'sharpness': float(lap.var())}
        self.audit.append(entry)
        if entry['sharpness'] < self.config.blur_min:
            entry['decision'] = 'drop_blur'
            return []
        duplicate = False
        if self.previous is not None and self.previous[0].shape == pixels.shape:
            diff = np.abs(pixels - self.previous[0])
            entry.update(mad=float(diff.mean()), max_pixel_difference=float(diff.max()),
                         ssim=self._ssim(gray, self.previous[1]))
            duplicate = (entry['mad'] <= self.config.duplicate_mad_max
                         and entry['max_pixel_difference'] <= self.config.duplicate_pixel_max
                         and entry['ssim'] >= self.config.duplicate_ssim_min)
        if duplicate:
            if self.pending is not None:
                self.pending[2]['decision'] = 'drop_duplicate_interior'
            entry['decision'] = 'deferred_duplicate_endpoint'
            self.pending = (deepcopy(record), pixels.copy(), entry)
            return []
        result = self.flush()
        entry['decision'] = 'keep_change'
        self.previous = (pixels.copy(), gray.copy())
        return result + [(deepcopy(record), pixels.copy())]

    def flush(self):
        if self.pending is None:
            return []
        record, pixels, entry = self.pending
        entry['decision'] = 'keep_duplicate_endpoint'
        self.pending = None
        return [(record, pixels)]


def otsu_distance(values):
    """One-dimensional between-class variance; no image histogram is involved."""
    a = sorted(float(v) for v in values)
    if len(a) < 2 or a[0] == a[-1]:
        return None
    total, left, best, threshold = sum(a), 0.0, -1.0, None
    for i in range(1, len(a)):
        left += a[i-1]
        if a[i-1] == a[i]:
            continue
        score = i * (len(a)-i) * (left/i - (total-left)/(len(a)-i))**2
        if score > best:
            best, threshold = score, (a[i-1]+a[i])/2
    return threshold


class OnlineChangeDetector:
    def __init__(self, fallback_distance=0.2, minimum_distance=0.05, history_size=256):
        if not 0 < minimum_distance <= fallback_distance <= 2 or history_size < 5:
            raise ValueError("Invalid content-change settings")
        self.fallback, self.floor, self.history_size = fallback_distance, minimum_distance, history_size
        self.history, self.previous = [], None

    def observe(self, embedding):
        vector = np.asarray(embedding, dtype=float)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError("Embedding must be a finite nonempty vector")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Embedding must have finite positive norm")
        vector = vector / norm
        if self.previous is None:
            self.previous = vector
            return {'boundary': False, 'method': 'first_frame', 'distance': None, 'threshold': None}
        if self.previous.shape != vector.shape:
            raise ValueError("Embedding dimension changed")
        distance = float(np.clip(1 - np.dot(self.previous, vector), 0, 2))
        # Only earlier distances set the threshold: no future suffix statistics.
        threshold, method = self.fallback, 'fallback'
        if len(self.history) >= 5:
            split = otsu_distance(self.history)
            if split is not None:
                threshold = float(np.median(self.history)) + 0.6 * (split - float(np.median(self.history)))
                threshold, method = max(self.floor, min(split, threshold)), 'past_only_relaxed_otsu'
        result = {'boundary': distance > threshold, 'method': method,
                  'distance': distance, 'threshold': threshold, 'history_count': len(self.history)}
        self.previous = vector
        self.history = (self.history + [distance])[-self.history_size:]
        return result


CAPTION_SYSTEM = '''Describe the visible content of the NEW timestamped frames in chronological order.
Return exactly JSON {"caption":"factual line one\\nfactual line two"}; no entity/action/state field lists.
Preserve observed actions, repetitions and changes. State uncertainty in ordinary prose.
A pose in one image does not establish motion or duration. Do not invent unseen causes,
intentions, identity, or activity in sampling gaps. The optional previous caption and
bridge image are fallible context: do not count them again as a new occurrence.
Caption only this batch's new observations. At most 180 words.
''' + CAPTION_STYLE_EN


def parse_caption_response(raw):
    """Accept bare JSON or one complete JSON code fence; never repair content.

    Real Qwen responses can wrap valid JSON in Markdown despite the prompt.
    Explanatory prose, multiple blocks and truncated JSON still fail. The
    original response remains unchanged in the backend's calls.jsonl.
    """
    text = raw.strip()
    fenced = re.fullmatch(r'```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```', text)
    if fenced:
        text = fenced.group(1)
    parsed = json.loads(text)
    if (not isinstance(parsed, dict) or set(parsed) != {'caption'}
            or not isinstance(parsed['caption'], str) or not parsed['caption'].strip()):
        raise ValueError('Writer must return only a nonempty caption')
    return parsed['caption'].strip()


class CaptionMemory:
    """Online candidate segmentation with budget-driven, lossless input batching.

    writer exposes fits(system, text, evidence) and generate(...)->JSON string.
    All retained frames must occur as NEW input in a successful caption part.
    This proves input coverage, not that every visual detail survives in prose.
    """
    def __init__(self, writer, detector=None, on_update=None):
        self.writer = writer
        self.detector = detector or OnlineChangeDetector()
        self.on_update = on_update or (lambda _: None)
        self.segments, self.frames, self.decisions = [], {}, []
        self.current, self.pending, self.bridge = None, [], None
        self.previous_caption = ''
        self.last_time, self.frontier = -1.0, -1.0
        self.finished = False

    def _request(self, batch, bridge):
        payload = {'candidate_id': self.current['id'],
                   'new_frame_ids': [e['id'] for e in batch],
                   'new_time_range': [batch[0]['time'], batch[-1]['time']],
                   'context_frame_ids': [bridge['id']] if bridge else [],
                   'previous_caption_context': self.previous_caption[-2000:]}
        return json.dumps(payload, ensure_ascii=False), ([bridge] if bridge else []) + batch

    def _fits(self, batch, bridge):
        text, images = self._request(batch, bridge)
        return self.writer.fits(CAPTION_SYSTEM, text, images)

    def _flush(self):
        if not self.pending:
            return
        text, images = self._request(self.pending, self.bridge)
        stage = f"caption:{self.current['id']}:{len(self.current['caption_parts'])+1}"
        raw = self.writer.generate(CAPTION_SYSTEM, text, images, stage=stage)
        caption = parse_caption_response(raw)
        part = {'id': stage, 'caption': caption,
                'new_frame_ids': [e['id'] for e in self.pending],
                'input_frame_ids': [e['id'] for e in images],
                'evidence_start': self.pending[0]['time'], 'evidence_end': self.pending[-1]['time'],
                'written_at': self.frontier}
        self.current['caption_parts'].append(part)
        # Keep each part, instead of silently replacing old prose with a summary.
        self.current['caption'] = '\n'.join(p['caption'] for p in self.current['caption_parts'])
        self.previous_caption = part['caption']
        self.bridge, self.pending = self.pending[-1], []
        self.on_update(self.snapshot())

    def push(self, frame, embedding, observed_at=None):
        if self.finished:
            raise ValueError("Cannot append after input finished")
        frame = deepcopy(frame)
        t = float(frame['time'])
        now = t if observed_at is None else float(observed_at)
        if (not math.isfinite(t) or not math.isfinite(now) or t < 0 or t <= self.last_time
                or now < t or now < self.frontier or frame['id'] in self.frames):
            raise ValueError("Invalid causal frame order/frontier")
        decision = self.detector.observe(embedding)
        self.frontier = now
        self.decisions.append({'frame_id': frame['id'], 'detected_at': now, **decision})
        if self.current is None or decision['boundary']:
            if self.current is not None:
                self._flush()
                self.current.update(status='content_boundary', closed_at=now)
            self.current = {'id': f'C{len(self.segments)+1:04d}', 'status': 'open',
                            'meaning': 'candidate_visual_segment_not_verified_atomic_event',
                            'frame_ids': [], 'caption_parts': [], 'caption': '',
                            'time_semantics': 'retained_evidence_range_not_action_boundaries'}
            self.segments.append(self.current)
        self.frames[frame['id']] = frame
        self.current['frame_ids'].append(frame['id'])
        self.current['evidence_start'] = self.frames[self.current['frame_ids'][0]]['time']
        self.current['evidence_end'] = t
        self.last_time = t
        if self.pending and not self._fits(self.pending + [frame], self.bridge):
            self._flush()
        if not self.pending and not self._fits([frame], self.bridge):
            # Prefer all new evidence over an optional overlap image.
            self.bridge = None
            if not self._fits([frame], None):
                raise ValueError("One new frame exceeds writer budget; no frame was silently discarded")
        self.pending.append(frame)

    def checkpoint(self, now):
        """A query/consumer can request a current caption without ending an event."""
        if not math.isfinite(now) or now < self.frontier:
            raise ValueError("Invalid checkpoint frontier")
        self.frontier = now
        self._flush()
        return self.snapshot()

    def finish(self, now):
        self.checkpoint(now)
        if self.current is not None:
            self.current.update(status='input_ended', closed_at=now)
        self.finished = True
        result = self.snapshot()
        self.on_update(result)
        return result

    def snapshot(self):
        covered = [fid for s in self.segments for p in s['caption_parts'] for fid in p['new_frame_ids']]
        return deepcopy({'schema': 'caption_frontend_v3', 'observed_until': self.frontier,
                         'segments': self.segments, 'frames': self.frames, 'boundary_decisions': self.decisions,
                         'coverage': {'retained_frames': len(self.frames), 'captioned_frames': len(covered),
                                      'unread_frame_ids': [fid for fid in self.frames if fid not in set(covered)],
                                      'all_retained_frames_read': len(covered) == len(self.frames) and len(set(covered)) == len(covered)},
                         'semantic_accuracy_verified': False, 'hierarchy_manager_connected': False})
