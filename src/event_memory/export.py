"""JSON and Markdown export for current direct-frame memory."""
import json


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def markdown(bundle):
    s = bundle['store']; lines = ['# 当前层级记忆：事件直接引用帧', '',
        f"视频 {bundle['video_id']}，观察至 {s['observed_until']:.3f} 秒。", '']
    def event(eid):
        e = s['events'][eid]
        lines.extend([f"#### {eid} · {e['start']:.3f}–{e['end']:.3f} 秒", '', e['caption'], '',
                      '帧：' + '、'.join(f"[{f} @ {s['frames'][f]['time']:.3f}s](frames/{f}.jpg)" for f in e['frame_ids']), ''])
    for p in s['episodes'].values():
        lines.extend([f"## {p['id']} · {p['title']}", '', p['summary'], ''])
        for phid in p['phase_ids']:
            ph = s['phases'][phid]
            status = f"已结束，确认于 {ph['closed_at']:.3f} 秒" if ph['status'] == 'closed' else '进行中'
            lines.extend([f"### {phid} · {ph['title']} · {status}", '', ph['summary'], ''])
            for eid in ph['event_ids']: event(eid)
    for eid in s['roots']: event(eid)
    return '\n'.join(lines)
