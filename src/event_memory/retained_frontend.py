"""Online content boundaries over frozen, hash-checked VAM retained images."""
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .caption_frontend import OnlineChangeDetector
from .caption_runtime import sha256_file


def verified_frames(records, root, frontier):
    """Validate the complete declared prefix before any model writing."""
    root = Path(root).resolve()
    previous, seen, result = -1., set(), []
    if not math.isfinite(frontier) or frontier <= 0 or not records:
        raise ValueError('Nonempty retained prefix and positive frontier required')
    for raw in records:
        if set(raw) != {'id', 'time', 'file', 'sha256'}:
            raise ValueError('Only retained image metadata is accepted')
        at = raw['time']
        if (not isinstance(at, (float, int)) or not math.isfinite(at)
                or not previous < at <= frontier or at < 0 or raw['id'] in seen):
            raise ValueError('Invalid causal retained frame order')
        if not isinstance(raw['id'], str) or not raw['id'].startswith('f') or not raw['id'][1:].isdigit():
            raise ValueError('Invalid frame ID')
        path = (root/raw['file']).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != raw['sha256']:
            raise ValueError('Retained image path/hash mismatch')
        result.append({'id': raw['id'], 'time': float(at), 'path': str(path)})
        previous = at; seen.add(raw['id'])
    return result


def candidate_batches(records, encoder, frontier, *, audit, fits=None, detector=None):
    """All images are read once as new writer input; boundaries delete no frames.

    A capacity split retains its candidate identity. It is not an event boundary.
    Each yielded batch uses actual evidence endpoints and an explicit later
    closure frontier, so a future boundary image is not sent with the old batch.
    """
    detector = detector or OnlineChangeDetector()
    pending, candidate_number, previous = [], 1, -1.

    def close(now, reason):
        return {'candidate_id': f'C{candidate_number:04d}',
                'start': pending[0]['time'], 'end': pending[-1]['time'],
                'evidence': list(pending), 'closed_at': now, 'closure_reason': reason}

    for frame in records:
        at = frame['time']
        if not previous < at <= frontier:
            raise ValueError('Candidate frames must be causal and ordered')
        with Image.open(frame['path']) as image:
            pixels = np.asarray(image.convert('RGB')).copy()
        decision = detector.observe(encoder.encode(pixels))
        audit.append({'frame_id': frame['id'], 'detected_at': at, **decision})
        if pending and decision['boundary']:
            yield close(at, 'content_boundary')
            pending = []; candidate_number += 1
        if pending and fits is not None and not fits(pending+[frame]):
            yield close(at, 'writer_capacity')
            pending = []
        if not pending and fits is not None and not fits([frame]):
            raise ValueError('Single retained frame exceeds writer budget')
        pending.append(frame); previous = at
    if pending:
        yield close(frontier, 'input_ended')
