"""Observation windows in an event caption. Bounds are sampled video times in seconds."""
from decimal import Decimal
import re

from .direct_frame_memory import finite

_TIME = r'(?:0|[1-9][0-9]*)(?:\.[0-9]+)?'
_HEADER = re.compile(rf'<({_TIME})s - ({_TIME})s>')


def time_header(start, end):
    start, end = finite(start), finite(end)
    if not 0 <= start <= end:
        raise ValueError('invalid observation time range')

    def seconds(value):
        return format(Decimal(str(value)).normalize(), 'f') if value else '0'

    return f'<{seconds(start)}s - {seconds(end)}s>'


def header_range(line):
    match = _HEADER.fullmatch(line.strip())
    if match:
        return tuple(map(float, match.groups()))
    return None


def caption_blocks(caption):
    """Parse canonical, nonempty, chronological blocks; reject stray action text."""
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError('expected a caption containing timed action blocks')
    blocks = []
    for raw in caption.splitlines():
        line = raw.strip()
        if not line:
            continue
        bounds = header_range(line)
        if bounds is not None:
            if time_header(*bounds) != line:
                raise ValueError(f'copy the exact input time header: {line}')
            if blocks and (not blocks[-1][1] or bounds == blocks[-1][0] or bounds[0] < blocks[-1][0][1]):
                raise ValueError('time blocks must be nonempty, unique and chronological')
            blocks.append((bounds, []))
        elif not blocks or line.startswith('<') or re.search(r'<\s*\d[^<>\n]*s\s*>', line):
            raise ValueError('use an exact input time header on its own line, then action lines')
        else:
            blocks[-1][1].append(line)
    if not blocks or not blocks[-1][1]:
        raise ValueError('each time block needs at least one observation')
    return blocks


def render_blocks(blocks):
    return '\n'.join(line for bounds, actions in blocks for line in [time_header(*bounds), *actions])


def evidence_caption(event, frames):
    """Old untimed input gets only its frame envelope. Never infer intra-event times."""
    caption = event['caption'].strip()
    if not caption.startswith('<'):
        caption = time_header(event['start'], event['end']) + '\n' + caption
    blocks = caption_blocks(caption)
    times = {frames[f]['time'] for f in event['frame_ids']}
    if any(t not in times for bounds, _ in blocks for t in bounds):
        raise ValueError(f"caption time has no supporting frame in {event['id']}")
    return render_blocks(blocks)
