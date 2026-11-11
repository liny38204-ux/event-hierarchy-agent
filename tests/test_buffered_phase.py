import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from event_memory.buffered_phase import BufferedPhaseStore, BufferedPhaseManager


class BufferedPhaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.frames=[]
        for i in range(1,6):
            p=Path(self.tmp.name)/f'f{i}.jpg';p.write_bytes(f'image{i}'.encode())
            self.frames.append({'id':f'f{i}','time':float(i),'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})

    def store(self, count=3):
        s=BufferedPhaseStore();events=[]
        for i in range(1,count+1):
            events.append({'id':f'E{i}','caption':f'小事件{i}','frame_ids':[f'f{i}'],'start':float(i),'end':float(i)})
        s.write(now=5.,frames=self.frames[:count],events=events);return s

    @staticmethod
    def wait(): return {'groups':[],'deferred_event_ids':['E1','E2','E3'],'refreshes':[],'reason':'需要更多上下文'}
    @staticmethod
    def first_group():
        return {'groups':[{'event_ids':['E1','E2'],'phase_id':'PH1','phase_title':'片段介绍','phase_summary':'前两段属于同一已观察片段。','episode_id':'P1','episode_title':'视频开头','episode_summary':'视频开头展示同一段已观察内容。'}], 'deferred_event_ids':['E3'],'refreshes':[],'reason':'前缀已足够，后缀继续等待'}

    def test_wait_keeps_roots_and_no_phase(self):
        s=self.store();calls=[]
        class B:
            calls=[]
            def generate(me,*args,**kwargs): calls.append(args);return json.dumps(BufferedPhaseTests.wait())
        m=BufferedPhaseManager(B(),event_trigger=3,seconds_trigger=15.)
        r=m.maybe_review(s,5.,new_event_ids=['E1','E2','E3'])
        self.assertTrue(r['reviewed']);self.assertEqual(s.snapshot()['roots'],['E1','E2','E3'])
        self.assertFalse(s.snapshot()['phases']);self.assertEqual(m.checkpoint()['new_event_ids_since_review'],[])
        self.assertEqual(len(calls),1)

    def test_trigger_and_prefix_assignment_only(self):
        s=self.store();seen=[]
        class B:
            calls=[]
            def generate(me,system,prompt,evidence,**kwargs):
                seen.append((json.loads(prompt),[x['id'] for x in evidence]));return json.dumps(BufferedPhaseTests.first_group())
        m=BufferedPhaseManager(B(),event_trigger=3,image_limit=3)
        r=m.maybe_review(s,5.,new_event_ids=['E1','E2'])
        self.assertFalse(r['reviewed']);self.assertEqual(r['trigger'],'buffering')
        r=m.maybe_review(s,5.,new_event_ids=['E3'])
        self.assertTrue(r['reviewed']);after=s.snapshot()
        self.assertEqual(after['phases']['PH1']['event_ids'],['E1','E2']);self.assertEqual(after['roots'],['E3'])
        self.assertEqual(after['phases']['PH1']['status'],'open');self.assertEqual(after['episodes']['P1']['status'],'open')
        self.assertEqual([x['id'] for x in seen[0][0]['pending_events']],['E1','E2','E3'])
        self.assertLessEqual(len(seen[0][1]),3)

    def test_new_phase_closes_predecessor_only_after_decision(self):
        s=self.store(3)
        class B:
            calls=[]
            def generate(me,*args,**kwargs): return json.dumps(BufferedPhaseTests.first_group())
        m=BufferedPhaseManager(B(),event_trigger=3,seconds_trigger=15.);m.maybe_review(s,5.,new_event_ids=['E1','E2','E3'])
        # Add two pending events, then assign them to a confirmed following PH.
        extra=[]
        for i in (4,5): extra.append({'id':f'E{i}','caption':f'后续{i}','frame_ids':[f'f{i}'],'start':float(i),'end':float(i)})
        s.write(now=5.,frames=self.frames[3:],events=extra)
        decision={'groups':[{'event_ids':['E3','E4','E5'],'phase_id':'PH2','phase_title':'后续步骤','phase_summary':'进入另一个已观察步骤。','episode_id':'P1','episode_title':'完整过程','episode_summary':'已观察到前后两个步骤。'}], 'deferred_event_ids':[],'refreshes':[],'reason':'内容发生了确认的步骤变化'}
        m.backend.generate=lambda *a,**k:json.dumps(decision)
        r=m.maybe_review(s,5.,new_event_ids=['E4','E5'],force=True);after=s.snapshot()
        self.assertTrue(r['reviewed']);self.assertEqual(after['phases']['PH1']['status'],'closed')
        self.assertEqual(after['phases']['PH2']['status'],'open');self.assertEqual(after['phases']['PH2']['event_ids'],['E3','E4','E5'])
        self.assertNotIn('PH1',after['completed_phase_ids'])

    def test_one_reply_can_create_two_ordered_phases_in_same_episode(self):
        s=self.store()
        decision={'groups':[
            {'event_ids':['E1'],'phase_id':'PH1','phase_title':'引入','phase_summary':'已观察到开始片段。','episode_id':'P1','episode_title':'过程','episode_summary':'视频已观察到两个连续步骤。'},
            {'event_ids':['E2'],'phase_id':'PH2','phase_title':'正文','phase_summary':'已观察到后续步骤。','episode_id':'P1','episode_title':'过程','episode_summary':'视频已观察到两个连续步骤。'}],
            'deferred_event_ids':['E3'],'refreshes':[],'reason':'前两段已经呈现不同的已观察用途'}
        class B:
            calls=[]
            def generate(me,*args,**kwargs):return json.dumps(decision)
        m=BufferedPhaseManager(B(),event_trigger=3,seconds_trigger=15.);m.maybe_review(s,5.,new_event_ids=['E1','E2','E3'])
        a=s.snapshot();self.assertEqual(a['episodes']['P1']['phase_ids'],['PH1','PH2'])
        self.assertEqual(a['phases']['PH1']['status'],'closed');self.assertEqual(a['phases']['PH2']['status'],'open')

    def test_invalid_nonprefix_rolls_back_store_and_checkpoint(self):
        s=self.store();before=s.snapshot()
        class B:
            calls=[]
            def generate(me,*args,**kwargs):
                return json.dumps({'groups':[{'event_ids':['E1','E3'],'phase_id':'PH1','phase_title':'坏','phase_summary':'坏','episode_id':'P1','episode_title':'坏','episode_summary':'坏'}], 'deferred_event_ids':['E2'],'refreshes':[],'reason':'坏'})
        m=BufferedPhaseManager(B(),event_trigger=3,seconds_trigger=15.);old=m.checkpoint()
        with self.assertRaises(ValueError):m.maybe_review(s,5.,new_event_ids=['E1','E2','E3'])
        self.assertEqual(s.snapshot(),before);self.assertEqual(m.checkpoint(),old)

    def test_time_and_force_final_review_without_forcing_assignment(self):
        s=self.store(1);calls=[]
        class B:
            calls=[]
            def generate(me,*args,**kwargs):calls.append(1);return json.dumps({'groups':[],'deferred_event_ids':['E1'],'refreshes':[],'reason':'仍不足以判断'})
        m=BufferedPhaseManager(B(),event_trigger=3,seconds_trigger=15.)
        self.assertFalse(m.maybe_review(s,5.,new_event_ids=['E1'])['reviewed'])
        self.assertTrue(m.maybe_review(s,20.,force=False)['reviewed'])
        self.assertTrue(m.maybe_review(s,20.,force=True)['reviewed'])
        self.assertEqual(s.snapshot()['roots'],['E1']);self.assertFalse(s.snapshot()['phases'])

    def test_dirty_assigned_event_refreshes_short_parents_without_roots(self):
        s=self.store();first=self.first_group()
        class B:
            calls=[]
            def generate(me,*args,**kwargs):return json.dumps(first)
        m=BufferedPhaseManager(B(),event_trigger=3,seconds_trigger=15.);m.maybe_review(s,5.,new_event_ids=['E1','E2','E3'])
        changed=dict(s.snapshot()['events']['E1']);changed['caption']='更新后的描述'
        s.write(now=5.,events=[changed],phases=[s.snapshot()['phases']['PH1']],episodes=[s.snapshot()['episodes']['P1']])
        fresh={'groups':[],'deferred_event_ids':['E3'],'refreshes':[{'phase_id':'PH1','phase_title':'片段更新','phase_summary':'该阶段内容已根据新增观察更新。','episode_id':'P1','episode_title':'视频更新','episode_summary':'视频已有阶段的已观察概括更新。'}],'reason':'仅更新已归属事件的上层摘要'}
        m.backend.generate=lambda *a,**k:json.dumps(fresh)
        result=m.maybe_review(s,20.,changed_event_ids=['E1'])
        self.assertTrue(result['reviewed']);self.assertEqual(s.snapshot()['phases']['PH1']['title'],'片段更新')

    def test_default_waits_for_five_new_events_and_preserves_wait_decision(self):
        s=BufferedPhaseStore();calls=[]
        class B:
            calls=[]
            def generate(me,system,prompt,evidence,**kwargs):
                pending=json.loads(prompt)['pending_event_ids'];calls.append(pending)
                return json.dumps({'groups':[],'deferred_event_ids':pending,
                                   'refreshes':[],'reason':'需要更多上下文'})
        m=BufferedPhaseManager(B())
        for i in range(1,6):
            eid=f'E{i}'
            s.write(now=float(i),frames=[self.frames[i-1]],events=[{
                'id':eid,'caption':f'事件{i}','frame_ids':[f'f{i}'],
                'start':float(i),'end':float(i)}])
            result=m.maybe_review(s,float(i),new_event_ids=[eid])
            self.assertEqual(result['reviewed'],i==5)
        self.assertEqual(result['trigger'],'event_trigger')
        self.assertEqual(calls,[['E1','E2','E3','E4','E5']])
        self.assertEqual(s.snapshot()['roots'],calls[0])
        self.assertFalse(s.snapshot()['phases'])

    def test_default_time_threshold_is_thirty_seconds_with_dirty_event(self):
        s=self.store(1);calls=[]
        class B:
            calls=[]
            def generate(me,*args,**kwargs):
                calls.append(1)
                return json.dumps({'groups':[],'deferred_event_ids':['E1'],
                                   'refreshes':[],'reason':'继续等待'})
        m=BufferedPhaseManager(B())
        self.assertFalse(m.maybe_review(s,5.,new_event_ids=['E1'])['reviewed'])
        for now in (20.,34.999):
            s.write(now=now)
            self.assertFalse(m.maybe_review(s,now,changed_event_ids=['E1'])['reviewed'])
        s.write(now=35.)
        result=m.maybe_review(s,35.,changed_event_ids=['E1'])
        self.assertTrue(result['reviewed']);self.assertEqual(result['trigger'],'time_trigger')
        self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
