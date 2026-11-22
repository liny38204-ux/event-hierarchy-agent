"""Question-time visual answering with causal recent frames and a history fallback."""
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import re

from .answer_options import option_letters
from .contact_sheet import write_contact_sheet
from .protocol import parse_json

LEGACY_POLICY = 'short_window_4s_then_history_v1'
POLICY = 'short_window_2fps_6frames_then_history_v2'
FRAME_REVIEW_POLICY = 'short_window_2fps_6frames_review_then_history_v3'
CONTACT_SHEET_POLICY = 'short_window_2fps_6frames_contact_sheet_review_then_history_v4'
ROW_CONTACT_SHEET_POLICY = 'short_window_2fps_4frames_row_contact_sheet_review_then_history_v5'
VISUAL_ROUTES = ('short_window', 'short_window_frame')
FPS = 2
MAX_FRAMES = 6
WINDOW_CAPACITIES = {LEGACY_POLICY: 4, POLICY: MAX_FRAMES, FRAME_REVIEW_POLICY: MAX_FRAMES,
                     CONTACT_SHEET_POLICY: MAX_FRAMES, ROW_CONTACT_SHEET_POLICY: 4}
_FRAME_ID = re.compile(r'f[0-9]{6,}')
LEGACY_SYSTEM = """你是流式视频的回答者。这里是提问时最近四秒内的真实采样帧。先看图，再判断能否直接区分选项。
仅当帧里可以辨认问题指定的对象、属性、动作及关键条件时作答。只看到颜色但没看清属于哪个物体、只看到动作结果却被问之前的位置、或只看到一次却被问累计次数，都应请求历史记忆。看到画面中的文字时，将它视为视觉内容。
能够作答时，只输出 JSON {"answer":"选项的大写字母","frame_ids":["直接支持答案的帧ID"]}。
证据不足时只输出 {"answer":null,"frame_ids":[]}。不要猜测，不要输出解释。"""
SYSTEM = LEGACY_SYSTEM.replace("最近四秒内的真实采样帧", "最近按每秒两帧采样的至多六张真实画面")
SHEET_SYSTEM = SYSTEM + """
这些画面合在一张两行三列的拼图里。按先上排后下排、每排从左到右的顺序阅读。
每个有画面的格子上方标有原帧ID和视频时间。引用能支持答案的原帧ID，跳过没有标签的空格。"""
ROW_SHEET_SYSTEM = SYSTEM.replace("至多六张", "至多四张") + """
这些画面合在一张从左到右的一行四列拼图里。每个有画面的格子上方标有原帧ID和视频时间。
引用能支持答案的原帧ID，跳过没有标签的空格。"""


def recent_frames(sampled_root, query_time):
    """Use actual sampled PTS from the audit, including frames rejected by the VAM filter."""
    if type(query_time) not in (float, int) or not math.isfinite(query_time) or query_time < 0:
        raise ValueError('invalid short-window query time')
    root = Path(sampled_root)
    records = json.loads((root.parent / 'filter_audit.json').read_text())['decisions']
    selected = []
    last_time = -1.0
    for record in records:
        t = record['time']
        if type(t) not in (int, float) or not math.isfinite(t) or t < 0 or t <= last_time:
            raise ValueError('sample timestamps must be strictly increasing')
        last_time = t
        if query_time - 4 < t <= query_time:
            selected.append((record['frame_id'], t))
    frames = []
    for fid, t in selected[-4:]:
        if not isinstance(fid, str) or not _FRAME_ID.fullmatch(fid):
            raise ValueError('invalid sampled frame ID')
        path = root / (fid + '.jpg')
        frames.append({'id': fid, 'time': t, 'path': str(path.resolve()),
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if len({frame['id'] for frame in frames}) != len(frames):
        raise ValueError('duplicate sampled frame ID')
    return frames


def _video_tail(container, stream, query_time, maximum_frames):
    """Emit each source frame once, on the first 2-Hz tick for which it is the latest frame."""
    tail = deque(maxlen=maximum_frames)
    end_tick = math.floor(query_time * FPS)
    previous = None
    for frame in container.decode(stream):
        if frame.pts is None:
            raise ValueError('missing source timestamp')
        pts = frame.pts * stream.time_base
        if pts < 0:
            continue
        if previous is not None:
            previous_pts, tick, _ = previous
            if pts < previous_pts:
                raise ValueError('nonmonotonic source timestamp')
            if tick <= end_tick and pts * FPS > tick:
                tail.append(previous)
        if pts > query_time:
            break
        previous = (pts, math.ceil(pts * FPS), frame)
    else:
        if previous is not None and previous[1] <= end_tick:
            tail.append(previous)
    return tail


def recent_video_frames(video_path, query_time, destination, *, maximum_frames=MAX_FRAMES):
    """Recover the last requested samples of a causal 2-Hz stream by source PTS."""
    import av

    if type(query_time) not in (int, float) or not math.isfinite(query_time) or query_time < 0:
        raise ValueError('invalid short-window query time')
    if not 1 <= maximum_frames <= MAX_FRAMES:
        raise ValueError('unsupported short-window frame capacity')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    end_tick = math.floor(query_time * FPS)
    lookback = maximum_frames
    # Usually one keyframe seek suffices. Sparse and low-FPS sources may need a longer prefix.
    while True:
        start_tick = max(0, end_tick + 1 - lookback)
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            if start_tick:
                try:
                    container.seek(math.floor(start_tick / (FPS * stream.time_base)),
                                   stream=stream, backward=True)
                except av.error.FFmpegError:
                    lookback = end_tick + 1
                    continue
            tail = _video_tail(container, stream, query_time, maximum_frames)
        if len(tail) == maximum_frames or not start_tick:
            break
        lookback *= 2

    frames = []
    for pts, tick, frame in tail:
        fid = f'f{tick + 1:06d}'
        path = destination / (fid + '.jpg')
        frame.to_image().convert('RGB').save(path, quality=95)
        frames.append({'id': fid, 'time': float(pts), 'sample_time': tick / FPS, 'path': str(path),
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return frames


def parse_window_decision(raw, options, frames):
    letters = option_letters(options)
    try:
        decision = parse_json(raw) if isinstance(raw, str) else {}
        if set(decision) != {'answer', 'frame_ids'}:
            raise ValueError('unexpected gate output')
        answer, cites = decision['answer'], decision['frame_ids']
        if answer is None and cites == []:
            return 'needs_history', None, []
        valid_ids = {frame['id'] for frame in frames}
        if (answer not in letters or not isinstance(cites, list) or not cites
                or any(not isinstance(fid, str) or fid not in valid_ids for fid in cites)
                or len(set(cites)) != len(cites)):
            raise ValueError('invalid answer or evidence')
        return 'answered', answer, cites
    except (TypeError, ValueError):
        return 'invalid_gate', None, []


def answer_from_window(question, options, frames, backend, *, stage, policy=LEGACY_POLICY, sheet_path=None):
    """Run the configured visual path before retrieving any text memory."""
    if policy not in WINDOW_CAPACITIES:
        raise ValueError('unsupported window policy')
    if policy in (FRAME_REVIEW_POLICY, CONTACT_SHEET_POLICY, ROW_CONTACT_SHEET_POLICY):
        if policy in (CONTACT_SHEET_POLICY, ROW_CONTACT_SHEET_POLICY) and frames and sheet_path is None:
            raise ValueError('a contact sheet destination is required')
        return review_window_frames(question, options, frames, backend, stage=stage,
                                    policy=policy,
                                    sheet_path=sheet_path if policy in (CONTACT_SHEET_POLICY, ROW_CONTACT_SHEET_POLICY) else None)
    return _window_attempt(question, options, frames, backend, stage, policy)


def _window_attempt(question, options, frames, backend, stage, policy, sheet_path=None):
    raw = None
    metadata = {}
    if frames:
        prompt = json.dumps({'question': question, 'options': dict(zip(option_letters(options), options))},
                            ensure_ascii=False)
        system = SYSTEM if policy in (POLICY, FRAME_REVIEW_POLICY, CONTACT_SHEET_POLICY, ROW_CONTACT_SHEET_POLICY) else LEGACY_SYSTEM
        evidence = frames
        if sheet_path is not None:
            row = policy == ROW_CONTACT_SHEET_POLICY
            sheet = write_contact_sheet(frames, sheet_path, rows=1 if row else 2, columns=4 if row else 3)
            metadata['contact_sheet'] = sheet
            # Visual input is one image. The answer gate still validates original frame IDs.
            evidence = [{'kind': 'contact_sheet', 'id': 'contact_sheet', 'time': frames[-1]['time'],
                         'path': str(sheet_path), **sheet}]
            system = ROW_SHEET_SYSTEM if row else SHEET_SYSTEM
        raw = backend.generate(system, prompt, evidence=evidence, stage=stage, max_tokens=160)
        status, answer, cites = parse_window_decision(raw, options, frames)
    else:
        status, answer, cites = 'no_frames', None, []
    fields = ('id', 'time', 'sample_time', 'sha256') if policy in (POLICY, FRAME_REVIEW_POLICY, CONTACT_SHEET_POLICY, ROW_CONTACT_SHEET_POLICY) else ('id', 'time', 'sha256')
    return {'status': status, 'raw': raw, 'prediction': answer, 'cited_frame_ids': cites,
            'frames': [{k: frame[k] for k in fields} for frame in frames], **metadata}


def review_window_frames(question, options, frames, backend, *, stage, policy=FRAME_REVIEW_POLICY, sheet_path=None):
    """Use the first cited single-frame answer after a non-answer on the full window."""
    whole = _window_attempt(question, options, frames, backend, stage, policy, sheet_path)
    attempts = []
    decisive = whole
    route = 'short_window' if whole['status'] == 'answered' else 'history'
    if whole['status'] != 'answered':
        for frame in frames:
            one = answer_from_window(question, options, [frame], backend,
                                     stage=f"{stage}:frame:{frame['id']}", policy=POLICY)
            attempts.append(one)
            if one['status'] == 'answered':
                decisive = one
                route = 'short_window_frame'
                break
    return {**decisive, 'frames': whole['frames'], 'route': route,
            'window_attempt': whole, 'frame_attempts': attempts}
