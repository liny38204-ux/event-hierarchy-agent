import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from event_memory.lossless_hierarchy import (
    BUFFER_CAPACITY, LosslessManager, LosslessStore, PlacementError, apply_placements, neighbor_index,
    parse_writer_output, placement_input, project_v11_bundle, writer_event)
from event_memory.merged_retrieval import validate_merged_memory
from event_memory.text_retrieval import CaptionRetriever
from event_memory.timed_caption import caption_blocks


def writer_json(caption='A man places a red cup on the kitchen counter.', summary='A man places a red cup on the kitchen counter.',
                changes=None, keywords=None):
    return json.dumps({'caption': caption, 'summary': summary,
                       'changes': changes if changes is not None else [{'object': 'red cup', 'change': 'placed', 'result': 'on the kitchen counter'}],
                       'keywords': keywords if keywords is not None else ['red cup', 'kitchen counter']})


def place(eid, phase='current', episode='', phase_title='', episode_title='', step='none', step_label=''):
    return {'event_id': eid, 'phase': phase, 'episode': episode, 'phase_title': phase_title,
            'episode_title': episode_title, 'step': step, 'step_label': step_label}


class ScriptedBackend:
    """Placement replies are scripted; summary calls get a fixed valid answer."""
    def __init__(self, placements):
        self.placements = list(placements)
        self.calls = []

    def token_count(self, value):
        return len(value) // 4

    def generate(self, system, text, evidence=(), *, stage, max_tokens=0):
        self.calls.append({'stage': stage, 'text': text, 'images': [e['id'] for e in evidence], 'finish_reason': 'stop'})
        if stage.startswith('summarize'):
            return json.dumps({'phase_summary': 'Someone prepares food at the counter.',
                               'episode_summary': 'Cooking in a kitchen.'})
        return json.dumps({'placements': self.placements.pop(0)})


class LosslessTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.frames = {}
        for i in range(1, 21):
            p = self.dir / f'f{i:06d}.jpg'; p.write_bytes(f'img{i}'.encode())
            self.frames[f'f{i:06d}'] = {'id': f'f{i:06d}', 'time': float(i), 'path': str(p),
                                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

    def event(self, store, i, caption=None, summary=None, keywords=None, changes=None):
        frames = [self.frames[f'f{i:06d}']]
        body, s, c, k = parse_writer_output(writer_json(caption or f'Observation {i}.', summary or f'Event {i} summary.',
                                                        changes, keywords))
        e = writer_event(f"E{store.snapshot()['next_event_id']}", body, s, c, k, frames)
        store.add_event(e, frames, float(i))
        return e['id']

    # writer -------------------------------------------------------------
    def test_writer_parser_validates_fields_and_rejects_time_headers(self):
        body, summary, changes, keywords = parse_writer_output(
            '```json\n' + writer_json('Line one.\nLine two.', keywords=['Red Cup', 'red cup', 'counter']) + '\n```')
        self.assertEqual(body, 'Line one.\nLine two.')
        self.assertEqual(keywords, ['red cup', 'counter'])
        self.assertEqual(changes[0]['result'], 'on the kitchen counter')
        for bad in [json.dumps({'caption': 'x'}),
                    writer_json('<3s - 4s>\nA cup.'),
                    writer_json(keywords=[]),
                    writer_json(changes=[{'object': 'cup'}]),
                    writer_json(summary='x' * 400)]:
            with self.assertRaises(ValueError):
                parse_writer_output(bad)

    def test_writer_event_owns_the_time_header(self):
        e = writer_event('E1', 'A cup.', 's', [], ['cup'], [self.frames['f000003'], self.frames['f000005']])
        self.assertEqual(e['caption'].splitlines()[0], '<3s - 5s>')
        self.assertEqual((e['start'], e['end']), (3.0, 5.0))

    # placement ----------------------------------------------------------
    def test_place_new_phase_steps_and_close_previous(self):
        store = LosslessStore()
        for i in range(1, 4):
            self.event(store, i)
        s = apply_placements(store.snapshot(), {'placements': [
            place('E1', 'new', 'new', 'cutting board work', 'making a salad', 'new', 'chop the onion'),
            place('E2', step='current'),
            place('E3', 'new', 'current', 'stove', '', 'new', 'heat the pan')]}, 3.0)
        self.assertEqual(s['phases']['PH1']['status'], 'closed')
        self.assertEqual(s['phases']['PH2']['status'], 'open')
        self.assertEqual(s['steps']['S1']['event_ids'], ['E1', 'E2'])
        self.assertEqual(s['episodes']['P1']['phase_ids'], ['PH1', 'PH2'])
        self.assertEqual(s['roots'], [])
        # every event kept verbatim
        self.assertEqual({e: s['events'][e]['caption'] for e in s['events']},
                         {e: store.snapshot()['events'][e]['caption'] for e in s['events']})

    def test_placement_contract_errors(self):
        store = LosslessStore()
        for i in range(1, 3):
            self.event(store, i)
        state = store.snapshot()
        bad = [
            {'placements': [place('E1')]},                                   # missing E2
            {'placements': [place('E1'), place('E2')]},                      # no open phase yet
            {'placements': [place('E2', 'new', 'new', 't', 'e'), place('E1')]},  # wrong order
            {'placements': [place('E1', 'new', 'new', 't', 'e', 'current'), place('E2')]},  # step current w/o previous
            {'placements': [place('E1', 'new', 'new', 't', 'e', 'none', 'oops'), place('E2')]},
            {'placements': [], 'reason': 'x'},
        ]
        for decision in bad:
            with self.assertRaises(ValueError):
                apply_placements(state, decision, 2.0)

    def test_manager_input_uses_summaries_only_and_is_bounded(self):
        store = LosslessStore()
        for i in range(1, 5):
            self.event(store, i, caption='\n'.join(f'Very long detailed line {k}.' for k in range(60)))
        payload, evidence = placement_input(store.snapshot(), 4.0)
        dumped = json.dumps(payload)
        self.assertNotIn('Very long detailed line', dumped)
        self.assertEqual(payload['pending_event_ids'], ['E1', 'E2', 'E3', 'E4'])
        self.assertLessEqual(len(evidence), 8)

    def test_manager_review_retry_and_summary(self):
        store = LosslessStore()
        backend = ScriptedBackend([
            [place('E1')],   # invalid: wrong length -> repaired
            [place('E1', 'new', 'new', 'counter', 'cooking', 'new', 'add ham'),
             place('E2', step='current'), place('E3', step='new', step_label='add rice'),
             place('E4', step='current'), place('E5', step='new', step_label='stir')],
        ])
        manager = LosslessManager(backend)
        for i in range(1, 6):
            self.event(store, i)
            result = manager.maybe_review(store, float(i))
        self.assertTrue(result['reviewed'])
        self.assertEqual(result['attempts'], 2)
        state = manager.finalize_snapshot(store, 5.0)
        self.assertEqual([state['steps'][s]['label'] for s in state['phases']['PH1']['step_ids']],
                         ['add ham', 'add rice', 'stir'])
        self.assertIsNotNone(state['phases']['PH1']['summary'])
        self.assertTrue(all(not c['images'] for c in backend.calls if c['stage'].startswith('summarize')))

    def test_buffer_capacity_enforced(self):
        store = LosslessStore()
        for i in range(1, BUFFER_CAPACITY + 1):
            self.event(store, i)
        with self.assertRaises(ValueError):
            self.event(store, BUFFER_CAPACITY + 1)

    # projection for the unchanged v11 agent loop -------------------------
    def build_two_phase_state(self):
        store = LosslessStore()
        backend = ScriptedBackend([[
            place('E1', 'new', 'new', 'living room tour', 'home tour', 'none'),
            place('E2'),
            place('E3', 'new', 'current', 'kitchen counter', '', 'none'),
            place('E4'), place('E5')]])
        manager = LosslessManager(backend)
        texts = ['A sofa with teal pillows stands against the wall.', 'A glass coffee table holds a vase.',
                 'An open cookbook lies on the kitchen counter.', 'Bar stools stand at the counter.',
                 'A white cabinet holds plates.']
        for i, t in enumerate(texts, 1):
            self.event(store, i, caption=t, summary=t, keywords=['cookbook', 'counter'] if i == 3 else ['room'],
                       changes=[])
            manager.maybe_review(store, float(i))
        return manager.finalize_snapshot(store, 5.0)

    def test_projection_passes_v11_validators_and_keeps_every_line(self):
        state = self.build_two_phase_state()
        bundle = project_v11_bundle(state, 'demo')
        validate_merged_memory(bundle, 5.0)
        captions = '\n'.join(e['caption'] for e in bundle['store']['events'].values())
        for e in state['events'].values():
            for line in e['caption'].splitlines():
                self.assertIn(line, captions)
        self.assertIn('Keywords: cookbook, counter', captions)
        kitchen = next(e for e in bundle['store']['events'].values() if 'cookbook' in e['caption'])
        self.assertEqual([b for b, _ in caption_blocks(kitchen['caption'])], [(3.0, 3.0), (4.0, 4.0), (5.0, 5.0)])

    def test_projection_is_searchable_by_line(self):
        class Bag:
            vocab = ['cookbook', 'counter', 'sofa', 'pillows', 'vase', 'stools', 'cabinet', 'plates', 'room']

            def vec(self, t):
                t = t.lower(); return [float(w in t) + 1e-3 for w in self.vocab]

            def encode_documents(self, texts):
                return [self.vec(t) for t in texts]

            def encode_query(self, t):
                return self.vec(t)
        bundle = project_v11_bundle(self.build_two_phase_state(), 'demo')
        hit = CaptionRetriever(bundle, Bag()).search('open cookbook', top_k=1)[0]
        self.assertEqual(hit['phase']['title'], 'kitchen counter')
        self.assertEqual(hit['matched']['time_range'], [3.0, 3.0])

    def test_neighbor_index_crosses_phases(self):
        state = self.build_two_phase_state()
        order = neighbor_index(state)
        self.assertEqual(order, ['E1', 'E2', 'E3', 'E4', 'E5'])
        self.assertNotEqual(state['events']['E2'], None)


if __name__ == '__main__':
    unittest.main()
