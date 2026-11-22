import copy
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from event_memory.merged_retrieval import persist_event_pages, select_merged_page, validate_merged_memory
from event_memory.text_retrieval import CaptionRetriever


class TableEmbedder:
    def __init__(self, vectors, query):
        self.vectors, self.query = vectors, query

    def encode_documents(self, texts):
        return [self.vectors[text] for text in texts]

    def encode_query(self, text):
        return self.query[text]


class MergedRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        from PIL import Image
        frames = {}
        for number in range(6):
            path = self.root / f'f{number}.png'
            Image.new('RGB', (18, 12), (number * 20, 0, 0)).save(path)
            frames[f'f{number}'] = {'id': f'f{number}', 'time': float(number), 'path': str(path),
                                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        caption = '<0s - 2s>\n入口画面。\n<3s - 5s>\n红色花瓶在地上。'
        state = {
            'frames': frames,
            'events': {'E6': {'id': 'E6', 'caption': caption, 'frame_ids': list(frames), 'start': 0., 'end': 5.}},
            'phases': {'PH1': {'id': 'PH1', 'title': '走廊观察', 'summary': '观察走廊与花瓶。',
                                'status': 'open', 'start': 0., 'end': 5., 'event_ids': ['E6']}},
            'episodes': {'P1': {'id': 'P1', 'title': '室内巡视', 'summary': '巡视室内。',
                                'status': 'open', 'start': 0., 'end': 5., 'phase_ids': ['PH1']}},
            'roots': [], 'completed_phase_ids': [], 'observed_until': 5.,
            'next_merge_id': 1, 'next_event_id': 7,
        }
        self.memory = {'schema': 'rolling_phase_memory_v11', 'schema_version': 11,
                       'status': 'complete', 'observed_until': 5., 'store': state}
        self.caption = caption

    def test_line_match_uses_merged_caption_window_and_only_its_frames(self):
        vectors = {self.caption: [0., 1.], '入口画面。': [-1., 0.], '红色花瓶在地上。': [1., 0.]}
        hit = CaptionRetriever(self.memory, TableEmbedder(vectors, {'花瓶': [1., 0.]})).search('花瓶', query_time=5.)[0]
        self.assertEqual(hit['event_id'], 'E6')
        self.assertEqual(hit['matched']['time_range'], [3., 5.])
        page = select_merged_page(self.memory, hit, 5., self.root / 'pages')
        self.assertEqual(page['anchor_policy'], 'matched_line')
        self.assertEqual(page['frame_ids'], ['f3', 'f4', 'f5'])
        self.assertEqual(page['anchor_time_range'], [3., 5.])
        self.assertTrue(Path(page['path']).is_file())
        self.assertEqual(hashlib.sha256(Path(page['path']).read_bytes()).hexdigest(), page['sha256'])

    def test_whole_caption_match_anchors_to_best_action_line(self):
        vectors = {self.caption: [1., 0.], '入口画面。': [-1., 0.], '红色花瓶在地上。': [0., 1.]}
        hit = CaptionRetriever(self.memory, TableEmbedder(vectors, {'问题': [1., 0.]})).search('问题', query_time=5.)[0]
        self.assertEqual(hit['matched']['kind'], 'caption')
        self.assertEqual(hit['best_action_line']['time_range'], [3., 5.])
        page = select_merged_page(self.memory, hit, 5., self.root / 'whole')
        self.assertEqual(page['anchor_policy'], 'best_action_line')
        self.assertEqual(page['frame_ids'], ['f3', 'f4', 'f5'])

    def test_reader_can_request_a_different_caption_window(self):
        vectors = {self.caption: [0., 1.], '入口画面。': [-1., 0.], '红色花瓶在地上。': [1., 0.]}
        hit = CaptionRetriever(self.memory, TableEmbedder(vectors, {'花瓶': [1., 0.]})).search('花瓶')[0]
        page = select_merged_page(self.memory, hit, 5., self.root / 'pages', requested_window=(0., 2.))
        self.assertEqual(hit['matched']['time_range'], [3., 5.])
        self.assertEqual(page['anchor_policy'], 'reader_requested_window')
        self.assertEqual(page['frame_ids'], ['f0', 'f1', 'f2'])
        with self.assertRaisesRegex(ValueError, 'observed header'):
            select_merged_page(self.memory, hit, 5., self.root / 'pages', requested_window=(1., 3.))

    def test_merged_text_and_stored_pages_work_after_original_images_are_gone(self):
        self.memory['event_pages'] = persist_event_pages(
            self.memory, [['f0', 'f1', 'f2'], ['f3', 'f4', 'f5']], self.root / 'pages')
        for frame in self.memory['store']['frames'].values():
            Path(frame['path']).unlink()
        vectors = {self.caption: [0., 1.], '入口画面。': [-1., 0.], '红色花瓶在地上。': [1., 0.]}
        hit = CaptionRetriever(self.memory, TableEmbedder(vectors, {'花瓶': [1., 0.]})).search('花瓶')[0]
        page = select_merged_page(self.memory, hit, 5., self.root / 'scratch')
        self.assertEqual(page['path'], self.memory['event_pages']['E6'][1]['path'])
        self.assertEqual(page['frame_ids'], ['f3', 'f4', 'f5'])
        self.assertFalse((self.root / 'scratch').exists())

    def test_reader_can_select_an_alternate_stored_page_in_the_same_event_scope(self):
        self.memory['event_pages'] = persist_event_pages(
            self.memory, [['f0', 'f1', 'f2'], ['f3', 'f4', 'f5']], self.root / 'pages')
        memory = copy.deepcopy(self.memory)
        memory['store']['events']['E6']['caption'] = '<0s - 5s>\n室内完整观察。'
        hit = {'event_id': 'E6', 'caption': memory['store']['events']['E6']['caption'],
               'matched': {'kind': 'caption'}, 'best_action_line': {}}
        alternate = memory['event_pages']['E6'][1]
        page = select_merged_page(memory, hit, 5., self.root / 'scratch',
                                  requested_page_id=alternate['sha256'])
        self.assertEqual((page['anchor_policy'], page['sha256'], page['frame_ids']),
                         ('reader_requested_page', alternate['sha256'], ['f3', 'f4', 'f5']))

    def test_requested_stored_page_checks_membership_scope_bytes_and_legacy_support(self):
        self.memory['event_pages'] = persist_event_pages(
            self.memory, [['f0', 'f1', 'f2'], ['f3', 'f4', 'f5']], self.root / 'pages')
        vectors = {self.caption: [0., 1.], '入口画面。': [-1., 0.], '红色花瓶在地上。': [1., 0.]}
        hit = CaptionRetriever(self.memory, TableEmbedder(vectors, {'花瓶': [1., 0.]})).search('花瓶')[0]
        first = self.memory['event_pages']['E6'][0]
        with self.assertRaisesRegex(ValueError, 'time scope'):
            select_merged_page(self.memory, hit, 5., self.root / 'scratch',
                               requested_window=hit['matched']['time_range'], requested_page_id=first['sha256'])
        with self.assertRaisesRegex(ValueError, 'does not belong'):
            select_merged_page(self.memory, hit, 5., self.root / 'scratch', requested_page_id='missing')
        Path(first['path']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            select_merged_page(self.memory, hit, 5., self.root / 'scratch',
                              requested_page_id=first['sha256'])
        legacy = copy.deepcopy(self.memory)
        legacy.pop('event_pages')
        with self.assertRaisesRegex(ValueError, 'requires stored'):
            select_merged_page(legacy, hit, 5., self.root / 'scratch', requested_page_id='page')

    def test_requested_stored_page_rejects_a_cell_after_the_query_time(self):
        self.memory['event_pages'] = persist_event_pages(
            self.memory, [['f0', 'f1', 'f2'], ['f3', 'f4', 'f5']], self.root / 'pages')
        vectors = {self.caption: [0., 1.], '入口画面。': [-1., 0.], '红色花瓶在地上。': [1., 0.]}
        hit = CaptionRetriever(self.memory, TableEmbedder(vectors, {'花瓶': [1., 0.]})).search('花瓶')[0]
        memory = copy.deepcopy(self.memory)
        page = memory['event_pages']['E6'][1]
        memory['store']['frames']['f5']['time'] = 6.
        page['cells'][-1]['time_s'] = 6.
        with patch('event_memory.merged_retrieval.validate_merged_memory', return_value=memory['store']):
            with self.assertRaisesRegex(ValueError, 'after the causal query time'):
                select_merged_page(memory, hit, 5., self.root / 'scratch', requested_page_id=page['sha256'])

    def test_v11_splits_actions_on_one_caption_line_without_splitting_decimals(self):
        caption = '<0s - 2s>\n先观察门口。3.5 米外是花瓶。 A hand moves. It picks up the vase.'
        memory = copy.deepcopy(self.memory)
        memory['store']['events']['E6']['caption'] = caption
        vectors = {caption: [0., 1.], '先观察门口。': [-1., 0.],
                   '3.5 米外是花瓶。': [1., 0.], 'A hand moves.': [0., 1.],
                   'It picks up the vase.': [-1., 0.]}
        hit = CaptionRetriever(memory, TableEmbedder(vectors, {'花瓶': [1., 0.]})).search('花瓶', query_time=5.)[0]
        self.assertEqual(hit['caption'], caption)
        self.assertEqual(hit['matched']['text'], '3.5 米外是花瓶。')
        self.assertEqual(hit['matched']['time_range'], [0., 2.])

    def test_future_incomplete_or_tampered_memory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'precedes'):
            validate_merged_memory(self.memory, 4.)
        incomplete = copy.deepcopy(self.memory); incomplete['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'completed'):
            validate_merged_memory(incomplete, 5.)
        broken = copy.deepcopy(self.memory)
        broken['store']['events']['E6']['frame_ids'] = ['f0', 'f2', 'f1']
        with self.assertRaisesRegex(ValueError, 'time order'):
            validate_merged_memory(broken, 5.)


if __name__ == '__main__':
    unittest.main()
