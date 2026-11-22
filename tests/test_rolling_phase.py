"""CPU protocol tests only; scripted captions are not model-quality evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from event_memory.rolling_phase import (
    separate_action_lines,
    ORGANIZER, SCHEMA, RollingPhaseStore, RollingPhaseManager, working_buffer,
)
from event_memory.text_retrieval import CaptionRetriever, render_context
from build_rolling_phase import build, build_video


def group(events, ph='PH1', parent='P1', caption=None):
    return {'event_ids': events, 'phase_id': ph, 'phase_title': '观察物品',
            'phase_summary': '持续观察和操作物品。', 'episode_id': parent,
            'episode_title': '活动过程', 'episode_summary': '观察到连续的活动过程。',
            'caption': caption}


class Backend:
    def __init__(self, policy=None):
        self.calls = []
        self.policy = policy

    def token_count(self, text):
        return len(text) // 3

    def generate(self, system, text, evidence, *, stage, max_tokens):
        payload = json.loads(text.split('\n修复错误：')[0])
        self.calls.append({'stage': stage, 'payload': payload,
                           'frame_ids': [f['id'] for f in evidence]})
        if self.policy:
            decision = json.loads(self.policy(payload))
            pending = {e['id']: e['caption'] for e in payload['pending_events']}
            for g in decision.get('groups', []):
                if g['caption'] is None:
                    g['caption'] = '\n'.join(pending[e] for e in g['event_ids'])
            return json.dumps(decision, ensure_ascii=False)
        active = payload['active_context']
        ph, parent = (active['phase']['id'], active['episode']['id']) if active else ('PH1', 'P1')
        captions = [e['caption'] for e in payload['pending_events']]
        return json.dumps({'groups': [group(payload['pending_event_ids'], ph, parent, '\n'.join(captions))],
                           'reason': '继续同一具体活动。'}, ensure_ascii=False)


class OneVectorEmbedder:
    """A deterministic retriever fixture, not evidence that a model understands actions."""
    def encode_documents(self, texts):
        return [[1.0] for _ in texts]

    def encode_query(self, text):
        return [1.0]


class RollingPhaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def frame(self, n, time=None):
        path = self.directory / f'f{n}.jpg'
        path.write_bytes(f'fixture-{n}'.encode())
        return {'id': f'f{n}', 'time': float(n if time is None else time),
                'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def add(self, store, n, *, now=None, time=None, caption=None):
        f = self.frame(n, time)
        eid = f"E{store.snapshot()['next_event_id']}"
        event = {'id': eid, 'caption': caption or f'第{n}次独立观察。', 'frame_ids': [f['id']],
                 'start': f['time'], 'end': f['time']}
        store.write(now=f['time'] if now is None else now, frames=[f], events=[event])
        return eid

    def test_five_then_carry_plus_four_new_rolls_with_all_frames(self):
        store, backend = RollingPhaseStore(), Backend()
        manager = RollingPhaseManager(backend)
        for n in range(1, 6):
            self.add(store, n)
            result = manager.maybe_review(store, float(n))
            self.assertEqual(result['reviewed'], n == 5)
        first = store.snapshot()
        first_carry = working_buffer(first)[0]
        self.assertEqual(len(first['events']), 1)
        self.assertEqual(first['phases']['PH1']['status'], 'open')
        self.assertEqual(first['completed_phase_ids'], [])
        self.assertEqual(first['events'][first_carry]['frame_ids'], [f'f{i}' for i in range(1, 6)])
        for n in range(6, 10):
            self.add(store, n)
            result = manager.maybe_review(store, float(n))
            self.assertEqual(result['reviewed'], n == 9)
        state = store.snapshot()
        carry = working_buffer(state)[0]
        self.assertNotIn(first_carry, state['events'])
        self.assertGreater(int(carry[1:]), int(first_carry[1:]))
        self.assertEqual(state['events'][carry]['frame_ids'], [f'f{i}' for i in range(1, 10)])
        self.assertEqual(set(state), set(first))  # no archive/source/history bank
        self.assertEqual(len(backend.calls), 2)  # grouping and compression share one call
        self.assertEqual(len(backend.calls[-1]['payload']['buffer_event_ids']), 5)
        for event in backend.calls[-1]['payload']['pending_events']:
            self.assertTrue(set(event['frame_ids']) & set(backend.calls[-1]['frame_ids']))

    def test_append_preserves_opening_and_allows_unchanged_scene(self):
        store = RollingPhaseStore()
        drafts = iter(['[E1]\n她打开笔盒，拿出蓝笔。', '[E4]\n对面的人递回便签，她又写了一行。', ''])
        def policy(payload):
            return json.dumps({'groups': [group(payload['pending_event_ids'], caption=next(drafts))],
                               'reason': '观察到同一活动。'})
        manager = RollingPhaseManager(Backend(policy))
        for n in range(1, 5):
            self.add(store, n)
            if n >= 2:
                review = manager.maybe_review(store, float(n), force=True)
                self.assertEqual(review['payload']['carry_caption_mode'], None if n == 2 else 'append')
        state = store.snapshot()
        event, = state['events'].values()
        self.assertEqual(event['caption'], '<1s - 1s>\n她打开笔盒，拿出蓝笔。\n<3s - 3s>\n对面的人递回便签，她又写了一行。')
        self.assertEqual(event['frame_ids'], ['f1', 'f2', 'f3', 'f4'])

    def test_manager_action_lines_are_preserved_in_native_retrieval_context(self):
        """The script supplies captions. It checks storage/retrieval, not VLM quality."""
        store = RollingPhaseStore()
        initial = '桌面中央有一叠扑克牌和一个纸杯。\n男子拿着手牌看向桌面。'
        developments = ('男子先把一张牌放到牌堆旁。\n他整理手中的牌。\n'
                        '男子又把一张牌放到同一牌堆旁。')

        def policy(payload):
            caption = '[E1]\n' + initial if not payload['active_context'] else '[E2]\n' + developments
            return json.dumps({'groups': [group(payload['pending_event_ids'], caption=caption)],
                               'reason': '同一局部活动仍在继续。'}, ensure_ascii=False)

        manager = RollingPhaseManager(Backend(policy))
        self.add(store, 1, caption='原始写入：桌面上有纸杯，男子拿牌。')
        manager.maybe_review(store, 1., force=True)
        self.add(store, 2, caption='原始写入：男子放牌、整理、再放牌。')
        manager.maybe_review(store, 2., force=True)

        state = store.snapshot()
        event, = state['events'].values()
        expected = '<1s - 1s>\n' + initial + '\n<2s - 2s>\n' + developments
        self.assertEqual(event['caption'], expected)
        self.assertEqual(event['caption'].count('纸杯'), 1)
        self.assertEqual(event['caption'].splitlines()[-3:], developments.splitlines())
        self.assertEqual(event['frame_ids'], ['f1', 'f2'])
        bundle = {'schema': SCHEMA, 'schema_version': 11,
                  'observed_until': state['observed_until'], 'store': state}
        context = render_context(CaptionRetriever(bundle, OneVectorEmbedder()).search(
            '放牌', query_time=state['observed_until']))
        self.assertIn(developments, context)

    def test_manager_caption_accepts_more_than_three_action_lines(self):
        store = RollingPhaseStore()
        actions = '\n'.join(f'第{i}次有意义的动作变化。' for i in range(1, 8))
        event_ids = [self.add(store, n) for n in range(1, 6)]
        manager = RollingPhaseManager(Backend(lambda payload: json.dumps({
            'groups': [group(event_ids, caption='[E1]\n' + actions)], 'reason': '这些动作属于同一阶段。'},
            ensure_ascii=False)))
        manager.maybe_review(store, 5.)
        event, = store.snapshot()['events'].values()
        self.assertEqual(event['caption'].splitlines()[1:], actions.splitlines())

    def test_long_caption_survives_initial_review_and_compaction(self):
        store = RollingPhaseStore()
        opening = '<1s - 1s>\n' + '手里拿着一本书。' * 450
        ending = opening + '\n<2s - 2s>\n把书放回原处。'
        drafts = iter([opening.replace('<1s - 1s>', '[E1]'),
                       ending.replace('<1s - 1s>', '[E1]').replace('<2s - 2s>', '[E2]')])
        def policy(payload):
            return json.dumps({'groups': [group(payload['pending_event_ids'], caption=next(drafts))],
                               'reason': '同一活动。'})
        manager = RollingPhaseManager(Backend(policy))
        for n, expected in enumerate((opening, ending), 1):
            self.add(store, n)
            manager.maybe_review(store, float(n), force=True)
            event, = store.snapshot()['events'].values()
            self.assertGreater(len(expected), 3000)
            self.assertEqual(event['caption'], expected)

    def test_manager_rejects_invented_or_cross_phase_or_reordered_windows(self):
        store = RollingPhaseStore()
        first = self.add(store, 1)
        second = self.add(store, 2)
        before = store.snapshot()
        for decision in [
            [group([first, second], caption='<1s - 2s>\n凭空扩大范围。')],
            [group([first], caption='[E2]\n借用另一阶段。'), group([second], 'PH2')],
            [group([first, second], caption='[E2]\n先写后段。\n[E1]\n倒叙。')],
            [group([first, second], caption='没有时间标题。')],
        ]:
            manager = RollingPhaseManager(Backend(lambda p: json.dumps({'groups': decision, 'reason':'测试。'})))
            with self.assertRaises(ValueError):
                manager.maybe_review(store, 2., force=True)
            self.assertEqual(store.snapshot(), before)

    def test_event_ref_restores_original_fractional_window(self):
        store, backend = RollingPhaseStore(), Backend()
        t = 27.994633333333333
        self.add(store, 1, time=t)
        output = RollingPhaseManager(backend).maybe_review(store, t, force=True)
        self.assertEqual(output['decision']['groups'][0]['caption'].splitlines()[0], '[E1]')
        self.assertEqual(store.snapshot()['events']['E1']['caption'].splitlines()[0],
                         '<27.994633333333333s - 27.994633333333333s>')

    def test_replay_requires_anchored_subwindows_and_keeps_shared_endpoints(self):
        store = RollingPhaseStore()
        frames = [self.frame(n, time=t) for n, t in enumerate([1.25, 2.0, 3.5], 1)]
        caption = '<1.25s - 2s>\n打开盒子。\n<2s - 3.5s>\n把盒子合上。'
        def add(value):
            store.write(now=3.5, frames=frames, events=[{
                'id':'E1', 'caption':value, 'frame_ids':[f['id'] for f in frames], 'start':1.25, 'end':3.5}])
        add(caption)
        backend = Backend()
        RollingPhaseManager(backend).maybe_review(store, 3.5, force=True)
        self.assertEqual(store.snapshot()['events']['E1']['caption'], caption)
        self.assertEqual(backend.calls[0]['payload']['pending_events'][0]['caption'].splitlines()[::2],
                         ['[E1:1]', '[E1:2]'])
        store = RollingPhaseStore()
        add(caption.replace('1.25s', '1.5s'))
        with self.assertRaisesRegex(ValueError, 'supporting frame'):
            RollingPhaseManager(backend).maybe_review(store, 3.5, force=True)
        self.assertEqual(len(backend.calls), 1)

    def test_compaction_cannot_drop_an_old_time_block(self):
        store = RollingPhaseStore()
        self.add(store, 1)
        self.add(store, 2)
        first = '[E1]\n' + '完整记录。' * 530 + '\n[E2]\n另一次操作。'
        def policy(payload):
            draft = first if not payload['active_context'] else '[E4]\n只交付新信息。'
            return json.dumps({'groups':[group(payload['pending_event_ids'], caption=draft)], 'reason':'测试。'})
        manager = RollingPhaseManager(Backend(policy))
        manager.maybe_review(store, 2., force=True)
        self.add(store, 3)
        before = store.snapshot()
        with self.assertRaisesRegex(ValueError, 'retain each existing'):
            manager.maybe_review(store, 3., force=True)
        self.assertEqual(store.snapshot(), before)

    def test_prompt_explicitly_separates_action_changes_from_static_environment(self):
        """Prompt contract only. A real VLM replay is required to assess compliance."""
        self.assertIn('每行', ORGANIZER)
        self.assertRegex(ORGANIZER, r'动作.*变化|变化.*动作')
        self.assertRegex(ORGANIZER, r'环境|背景|陈设')
        self.assertRegex(ORGANIZER, r'一次|不重复|重复')

    def test_near_limit_receives_explicit_compaction_request(self):
        store = RollingPhaseStore()
        first = '完整观察。' * 550
        replacement = '<1s - 1s>\n她打开笔盒。\n<3s - 3s>\n她合上笔盒。'
        drafts = iter(['[E1]\n' + first,
                       replacement.replace('<1s - 1s>', '[E3]').replace('<3s - 3s>', '[E4]')])
        def policy(payload):
            return json.dumps({'groups': [group(payload['pending_event_ids'], caption=next(drafts))],
                               'reason': '同一阶段。'})
        manager = RollingPhaseManager(Backend(policy))
        self.add(store, 1)
        self.add(store, 2)
        manager.maybe_review(store, 2., force=True)
        self.add(store, 3)
        review = manager.maybe_review(store, 3., force=True)
        self.assertEqual(review['payload']['carry_caption_mode'], 'compact')
        event, = store.snapshot()['events'].values()
        self.assertEqual(event['caption'], replacement)

    def test_explicit_transitions_split_without_collapsing_repetitions_or_uncertainty(self):
        draft = ('她将一本书放在架上。\n她整理手上的书，并将另一本放在架上。\n'
                 '另一只手拿起书并放回架上。\n她可能在整理或放置物体，随后位置不清。')
        self.assertEqual(separate_action_lines(draft).splitlines(), [
            '她将一本书放在架上。', '她整理手上的书。', '将另一本放在架上。',
            '另一只手拿起书。', '放回架上。', '她可能在整理或放置物体，随后位置不清。'])
        repeated = '他按了一下。\n他按了一下。'
        self.assertEqual(separate_action_lines(repeated), repeated)

    def test_thirty_second_timer_resets_and_old_carry_age_does_not_retrigger(self):
        store, manager = RollingPhaseStore(), RollingPhaseManager(Backend())
        self.add(store, 1, now=30., time=0.)
        result = manager.maybe_review(store, 30.)
        self.assertEqual(result['trigger'], 'time_trigger')
        self.add(store, 2, now=31., time=31.)
        self.assertFalse(manager.maybe_review(store, 31.)['reviewed'])
        self.add(store, 3, now=60., time=60.)
        self.assertEqual(manager.maybe_review(store, 60.)['trigger'], 'time_trigger')
        self.assertEqual(manager.checkpoint()['last_review_at'], 60.)

    def test_phase_boundary_archives_old_carry_and_only_new_phase_stays_in_buffer(self):
        store, backend = RollingPhaseStore(), Backend()
        manager = RollingPhaseManager(backend)
        for n in range(1, 6):
            self.add(store, n)
            manager.maybe_review(store, float(n))
        old = deepcopy(store.snapshot()['events'])
        self.add(store, 6)
        backend.policy = lambda p: json.dumps({'groups': [group(p['pending_event_ids'], 'PH2')],
                                               'reason': '观察焦点发生转换。'})
        manager.maybe_review(store, 6., force=True)
        state = store.snapshot()
        self.assertEqual(state['completed_phase_ids'], ['PH1'])
        self.assertEqual(state['phases']['PH1']['status'], 'closed')
        self.assertEqual(state['phases']['PH2']['status'], 'open')
        self.assertEqual(state['episodes']['P1']['phase_ids'], ['PH1', 'PH2'])
        self.assertEqual(working_buffer(state), state['phases']['PH2']['event_ids'])
        for eid, event in old.items():
            self.assertEqual(state['events'][eid], event)
        self.assertEqual(len(backend.calls), 2)  # no redundant close-time model call

    def test_identical_consecutive_phase_fragments_match_single_group(self):
        captions = ['[E1]\n她放下牌。', '[E2]\n她再次放下牌。']
        whole = [group(['E1', 'E2'], caption='\n'.join(captions))]
        split = [group([eid], caption=cap) for eid, cap in zip(['E1', 'E2'], captions)]
        states = []
        for groups in (whole, split):
            store = RollingPhaseStore()
            self.add(store, 1)
            self.add(store, 2)
            backend = Backend(lambda _: json.dumps({'groups': groups, 'reason': '连续观察。'}))
            result = RollingPhaseManager(backend).maybe_review(store, 2., force=True)
            self.assertEqual(result['attempts'], 1)
            states.append(store.snapshot())
        self.assertEqual(*states)

    def test_split_phase_cannot_borrow_refs_or_resolve_conflicts_by_overwrite(self):
        first = group(['E1'], caption='[E1]\n开始。')
        second = group(['E2'], caption='[E2]\n继续。')
        for groups in ([first, dict(second, phase_summary='另一个描述。')],
                       [dict(first, caption='[E2]\n借用。'), second],
                       [first, group(['E2'], 'PH2'), group(['E3'])]):
            store = RollingPhaseStore()
            for i in range(1, len(groups) + 1):
                self.add(store, i)
            before = store.snapshot()
            manager = RollingPhaseManager(Backend(lambda _: json.dumps({'groups': groups, 'reason': '测试。'})))
            with self.assertRaises(ValueError):
                manager.maybe_review(store, float(len(groups)), force=True)
            self.assertEqual(store.snapshot(), before)

    def test_retry_feedback_fits_same_text_budget(self):
        from event_memory.rolling_phase import CONTEXT_LIMIT
        class NearLimit(Backend):
            def token_count(self, text):
                return CONTEXT_LIMIT + 1 if 'previous-invalid-body' in text else CONTEXT_LIMIT - 1
            def generate(self, system, text, evidence, **kwargs):
                if 'previous-invalid-body' in text:
                    raise AssertionError('repair leaked an oversized previous response')
                raw = super().generate(system, text, evidence, **kwargs)
                return 'previous-invalid-body' if len(self.calls) == 1 else raw
        store = RollingPhaseStore()
        self.add(store, 1)
        result = RollingPhaseManager(NearLimit()).maybe_review(store, 1., force=True)
        self.assertEqual(result['attempts'], 2)

    def test_one_review_can_create_multiple_phases_and_episodes(self):
        store = RollingPhaseStore()
        events = [self.add(store, n) for n in range(1, 6)]
        def policy(payload):
            return json.dumps({'groups': [group(events[:2]), group(events[2:4], 'PH2'),
                                          group(events[4:], 'PH3', 'P2')], 'reason': '观察到局部与整体转换。'})
        RollingPhaseManager(Backend(policy)).maybe_review(store, 5.)
        state = store.snapshot()
        self.assertEqual(state['episodes']['P1']['status'], 'closed')
        self.assertEqual(state['episodes']['P2']['status'], 'open')
        self.assertEqual(state['completed_phase_ids'], ['PH1', 'PH2'])
        self.assertTrue(all(len(ph['event_ids']) == 1 for ph in state['phases'].values()))
        self.assertEqual(len(working_buffer(state)), 1)
        self.assertEqual({f for e in state['events'].values() for f in e['frame_ids']}, {f'f{i}' for i in range(1, 6)})

    def test_wait_is_repaired_not_silently_allowed_at_capacity(self):
        store = RollingPhaseStore()
        for n in range(1, 6): self.add(store, n)
        backend = Backend(lambda p: json.dumps({'groups': [], 'reason': '继续等。'}))
        manager = RollingPhaseManager(backend)
        before, schedule = store.snapshot(), manager.checkpoint()
        with self.assertRaisesRegex(ValueError, 'WAIT'):
            manager.maybe_review(store, 5.)
        self.assertEqual(store.snapshot(), before)
        self.assertEqual(manager.checkpoint(), schedule)
        self.assertEqual(len(backend.calls), 3)
        with self.assertRaisesRegex(ValueError, 'exceeds five'):
            self.add(store, 6)
        self.assertEqual(store.snapshot(), before)

    def test_bad_late_group_and_bad_caption_leave_no_partial_mutation(self):
        for bad in ('missing_event', 'caption'):
            store = RollingPhaseStore()
            events = [self.add(store, n) for n in range(1, 6)]
            groups = [group(events[:2]), group(events[2:], 'PH2')]
            if bad == 'missing_event': groups[-1]['event_ids'] = events[2:4]
            else: groups[-1]['caption'] = '<3s - 3s>\n   '
            backend = Backend(lambda p: json.dumps({'groups': groups, 'reason': '错误夹具。'}))
            manager = RollingPhaseManager(backend)
            before = store.snapshot()
            with self.assertRaises(ValueError): manager.maybe_review(store, 5.)
            self.assertEqual(store.snapshot(), before)
            self.assertEqual(manager.checkpoint(), {'last_review_at': 0.})

    def test_final_prefix_groups_remaining_but_does_not_close_or_recompress_carry(self):
        store = RollingPhaseStore()
        backend = Backend(lambda payload: json.dumps({
            'groups': [group(payload['pending_event_ids'], caption='[E1]\n单例阶段的整理文本。')],
            'reason': '开始一个新的局部活动。'}, ensure_ascii=False))
        manager = RollingPhaseManager(backend)
        self.add(store, 1)
        result = manager.maybe_review(store, 1., force=True)
        self.assertEqual(result['trigger'], 'forced_final_review')
        self.assertEqual(store.snapshot()['phases']['PH1']['status'], 'open')
        self.assertEqual(store.snapshot()['events']['E1']['caption'], '<1s - 1s>\n单例阶段的整理文本。')
        self.assertFalse(manager.maybe_review(store, 1., force=True)['reviewed'])
        self.assertEqual(len(backend.calls), 1)

    def test_runner_41_events_bounds_buffer_and_preserves_full_union(self):
        frame_root = self.directory / 'source'
        folder = frame_root / 'ovo-fixture' / 'frames'
        folder.mkdir(parents=True)
        frames, steps = {}, []
        for n in range(1, 42):
            frame = self.frame(n)
            (folder / f'f{n}.jpg').write_bytes(Path(frame.pop('path')).read_bytes())
            frames[frame['id']] = frame
            steps.append({'segment_id': f'S{n}', 'now': float(n), 'frame_ids': [frame['id']],
                          'events': [{'source_event_id': f'S{n}', 'op': 'new', 'caption': f'第{n}次观察。',
                                      'frame_ids': [frame['id']], 'start': float(n), 'end': float(n)}]})
        video = {'video_id': 'ovo-fixture', 'frames': frames, 'steps': steps}
        output = self.directory / 'output'
        backend = Backend()
        stats = build_video(video, frame_root, output, backend)
        self.assertEqual(stats['max_buffer_size'], 5)
        self.assertEqual(stats['events'], 1)
        self.assertEqual(stats['review_count'], 10)
        self.assertEqual(stats['frames'], 41)
        self.assertEqual(stats['edited_captions'], 0)
        bundle = json.loads((output / 'ovo-fixture' / 'memory.json').read_text())
        self.assertEqual(bundle['schema_version'], 11)
        self.assertEqual(bundle['status'], 'complete')
        event, = bundle['store']['events'].values()
        self.assertNotIn('event-narrative:', '\n'.join(call['stage'] for call in backend.calls))
        self.assertEqual(len(event['frame_ids']), 41)
        self.assertEqual(bundle['store']['completed_phase_ids'], [])
        self.assertEqual(bundle['observed_until'], 41.)

    def test_runner_rejects_legacy_multi_event_or_continuation_before_any_output(self):
        for events in ([{'op': 'new'}, {'op': 'new'}], [{'op': 'continue'}]):
            output = self.directory / 'rejected'
            with self.assertRaisesRegex(ValueError, 'one immutable new E'):
                build_video({'steps': [{'events': events}]}, self.directory, output, Backend(), retain_replay=True)
            self.assertFalse(output.exists())

    def test_standalone_build_keeps_only_merged_memory_not_replay_caption(self):
        frame_root = self.directory / 'source'
        source = frame_root / 'ovo-fixture' / 'frames'
        source.mkdir(parents=True)
        frame = self.frame(1)
        (source / 'f1.jpg').write_bytes(Path(frame['path']).read_bytes())
        raw_caption = '<1s - 1s>\nWRITER_ONLY_SENTINEL'

        def merged_caption(payload):
            return json.dumps({'groups': [group(payload['pending_event_ids'], caption='[E1]\n管理器合并后的观察。')],
                               'reason': '形成一个阶段。'}, ensure_ascii=False)

        video = {'video_id': 'ovo-fixture', 'frames': {frame['id']: {k: frame[k] for k in ('id', 'time', 'sha256')}},
                 'steps': [{'segment_id': 'S1', 'now': 1., 'frame_ids': ['f1'],
                            'events': [{'source_event_id': 'S1', 'op': 'new', 'caption': raw_caption,
                                        'frame_ids': ['f1'], 'start': 1., 'end': 1.}]}]}
        input_path = self.directory / 'raw_replay_input.json'
        input_path.write_text(json.dumps({'videos': [video]}, ensure_ascii=False))
        output = self.directory / 'standalone'
        build(input_path, frame_root, output, Backend(merged_caption))

        manifest = (output / 'input_manifest.json').read_text()
        memory = (output / 'ovo-fixture' / 'memory.json').read_text()
        self.assertNotIn('WRITER_ONLY_SENTINEL', manifest)
        self.assertNotIn('WRITER_ONLY_SENTINEL', memory)
        self.assertIn('管理器合并后的观察。', memory)
        self.assertFalse((output / 'ovo-fixture' / 'trace.jsonl').exists())

    def test_streaming_build_rejects_duplicate_or_out_of_order_frame_delivery(self):
        frame_root = self.directory / 'source'
        source = frame_root / 'ovo-fixture' / 'frames'
        source.mkdir(parents=True)
        frames = {}
        for n in (1, 2):
            frame = self.frame(n)
            (source / f'f{n}.jpg').write_bytes(Path(frame['path']).read_bytes())
            frames[frame['id']] = {k: frame[k] for k in ('id', 'time', 'sha256')}
        video = {'video_id': 'ovo-fixture', 'frames': frames,
                 'steps': [{'segment_id': 'S1', 'now': 2., 'frame_ids': ['f2', 'f1'],
                            'events': [{'source_event_id': 'S1', 'op': 'new', 'caption': '乱序输入。',
                                        'frame_ids': ['f2', 'f1'], 'start': 1., 'end': 2.}]}]}
        with self.assertRaisesRegex(ValueError, 'cover retained frames once in chronological order'):
            build_video(video, frame_root, self.directory / 'out', Backend(), retain_replay=False)

    def test_failure_marks_last_commit_without_publishing_pending_captions(self):
        source = self.directory/'case'/'frames'
        source.mkdir(parents=True)
        manifest = {}
        steps = []
        for i in range(1, 4):
            frame = self.frame(i, time=30. if i == 1 else 30. + i)
            (source/f'f{i}.jpg').write_bytes(Path(frame['path']).read_bytes())
            manifest[frame['id']] = {k:frame[k] for k in ('id','time','sha256')}
            event = {'source_event_id':f'S{i}', 'op':'new', 'caption':f'writer-only-{i}',
                     'frame_ids':[frame['id']], 'start':frame['time'], 'end':frame['time']}
            steps.append({'segment_id':f'S{i}', 'now':frame['time'], 'frame_ids':event['frame_ids'], 'events':[event]})
        def interrupted():
            yield from steps[:2]
            raise RuntimeError('fixture: next writer failed')
        output = self.directory/'out'
        with self.assertRaisesRegex(RuntimeError, 'next writer failed'):
            build_video({'video_id':'case', 'frames':manifest, 'steps':interrupted()},
                        self.directory, output, Backend())
        saved = (output/'case/memory.json').read_text()
        bundle = json.loads(saved)
        self.assertEqual((bundle['status'], bundle['observed_until']), ('failed', 30.))
        self.assertEqual(list(bundle['store']['frames']), ['f1'])
        self.assertNotIn('writer-only-2', saved)

    def test_streaming_failure_receipt_does_not_persist_manager_output(self):
        frame_root = self.directory / 'source'
        source = frame_root / 'ovo-fixture' / 'frames'
        source.mkdir(parents=True)
        frame = self.frame(1)
        (source / 'f1.jpg').write_bytes(Path(frame['path']).read_bytes())
        video = {'video_id': 'ovo-fixture', 'frames': {frame['id']: {k: frame[k] for k in ('id', 'time', 'sha256')}},
                 'steps': [{'segment_id': 'S1', 'now': 1., 'frame_ids': ['f1'],
                            'events': [{'source_event_id': 'S1', 'op': 'new', 'caption': '<1s - 1s>\nwriter prose',
                                        'frame_ids': ['f1'], 'start': 1., 'end': 1.}]}]}
        input_path = self.directory / 'input.json'
        input_path.write_text(json.dumps({'videos': [video]}, ensure_ascii=False))

        class InvalidManager(Backend):
            def generate(self, *args, **kwargs):
                return 'MANAGER_RAW_SENTINEL'

        output = self.directory / 'failed'
        with self.assertRaisesRegex(ValueError, 'rolling manager failed'):
            build(input_path, frame_root, output, InvalidManager())
        receipt = (output / 'index.json').read_text()
        self.assertNotIn('MANAGER_RAW_SENTINEL', receipt)
        self.assertNotIn('writer prose', receipt)
        error = json.loads(receipt)['error']
        self.assertEqual((error['time'], error['stage']), (1., 'rolling-organize'))
        self.assertEqual(error['attempt_errors'], ['invalid_json'] * 3)


if __name__ == '__main__':
    unittest.main()
