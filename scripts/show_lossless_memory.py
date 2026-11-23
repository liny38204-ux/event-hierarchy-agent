#!/usr/bin/env python3
"""Render a v12 build as a readable Markdown tree: entity table, then P -> PH -> Step -> E.

usage: show_lossless_memory.py BUILD_DIR [OUT.md]
BUILD_DIR is the `build/` directory written by build_lossless_memory.py.
"""
import json
import sys
from pathlib import Path


def span(a, b):
    return f'{a:g}s' if a == b else f'{a:g}–{b:g}s'


def render(build_dir):
    build_dir = Path(build_dir)
    status = json.loads((build_dir / 'status.json').read_text())
    path = Path(status['memory_v12'])
    if not path.is_file():  # copied build: fall back to the local layout
        path = next((build_dir / 'v12').glob('*/memory_v12.json'))
    doc = json.loads(path.read_text())
    s, m, out = doc['store'], status.get('metrics', {}), []
    s.setdefault('entities', {})  # revision 1 builds have no registry
    w = out.append
    w(f"# {doc['video_id']} — v12 memory (r{doc.get('schema_revision', 1)}, observed until {doc['observed_until']:g}s)\n")
    w(' · '.join(f'{k}={m[k]}' for k in ('events', 'steps', 'phases', 'episodes', 'entities', 'reviews', 'place_retries',
                                          'writer_repairs', 'camera_lines_dropped', 'camera_changes_dropped',
                                          'entities_reused', 'seconds') if k in m) + '\n')

    w('## Entities\n')
    w('| ID | kind | label | latest appearance | seen | events |')
    w('| --- | --- | --- | --- | --- | --- |')
    for n in sorted(s['entities'].values(), key=lambda n: (n['id'][0], int(n['id'][1:]))):
        w(f"| {n['id']} | {n['kind']} | {n['label']} | {n['appearance']} | {span(n['first_seen'], n['last_seen'])} | "
          f"{', '.join(n['event_ids'])} |")
    w('')

    step_of = {e: sid for sid, st in s['steps'].items() for e in st['event_ids']}
    labels = {n['id']: n['label'] for n in s['entities'].values()}

    def event(eid, pad):
        e = s['events'][eid]
        w(f"{pad}- **{eid}** `{span(e['start'], e['end'])}` {len(e['frame_ids'])} frames — {e['summary']}")
        for line in e['caption'].splitlines()[1:]:
            w(f'{pad}  > {line}')
        if e.get('entities'):
            w(f"{pad}  - entities: " + '; '.join(f"{x['id']} ({labels[x['id']]})" for x in e['entities']))
        for c in e['changes']:
            ref = f"[{c['ref']}] " if c.get('ref') else ''
            w(f"{pad}  - change: {ref}{c['object']} {c['change']} → {c['result']}")
        w(f"{pad}  - keywords: {', '.join(e['keywords'])}")

    w('## Hierarchy\n')
    for p in sorted(s['episodes'].values(), key=lambda p: p['start']):
        w(f"### {p['id']} {p['title']} `{span(p['start'], p['end'])}`\n")
        if p.get('summary'):
            w(f"*{p['summary']}*\n")
        for pid in p['phase_ids']:
            ph = s['phases'][pid]
            w(f"#### {pid} {ph['title']} `{span(ph['start'], ph['end'])}` · {len(ph['event_ids'])} E · {ph['status']}\n")
            if ph.get('summary'):
                w(f"*{ph['summary']}*\n")
            for eid in ph['event_ids']:
                sid = step_of.get(eid)
                if sid and s['steps'][sid]['event_ids'][0] == eid:
                    st = s['steps'][sid]
                    w(f"- **Step {sid}: {st['label']}** `{span(st['start'], st['end'])}` · {len(st['event_ids'])} E")
                event(eid, '  ' if sid else '')
            w('')
    if s['roots']:
        w('## Unplaced\n')
        for eid in s['roots']:
            event(eid, '')
    return '\n'.join(out) + '\n'


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    text = render(sys.argv[1])
    if len(sys.argv) == 3:
        Path(sys.argv[2]).write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == '__main__':
    main()
