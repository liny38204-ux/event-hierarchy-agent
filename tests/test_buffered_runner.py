from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from event_memory.buffered_phase import BufferedPhaseStore, BufferedPhaseManager
from build_buffered_phase import ingest


class BufferedReplayTests(unittest.TestCase):
    def test_overlong_phase_summary_repair_names_path_and_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'f1.jpg';p.write_bytes(b'image')
            s=BufferedPhaseStore();s.write(now=1.,frames=[{'id':'f1','time':1.,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}],
                events=[{'id':'E1','caption':'观察','frame_ids':['f1'],'start':1.,'end':1.}])
            before=s.snapshot();prompts=[];snapshots=[]
            # Exact 82-character PH summary returned by the failed 20261111-02 organizer call.
            overlong='镜头从卖场过道深入影音区，扫过“movies”和“COLLECTIBLES”标识，重点展示了各类游戏与收藏品货架，并记录了一只手拿起黑色包装盒进行近距离观察的过程。'
            self.assertEqual(len(overlong),82)
            bad={'groups':[{'event_ids':['E1'],'phase_id':'PH1','phase_title':'影音与收藏品货架浏览',
                            'phase_summary':overlong,'episode_id':'P1','episode_title':'卖场环境观察',
                            'episode_summary':'镜头观察卖场中的不同区域。'}],
                 'deferred_event_ids':[],'refreshes':[],'reason':'进入新的局部区域。'}
            good=deepcopy(bad);good['groups'][0]['phase_summary']='镜头进入影音与收藏品区域，展示货架和手持商品。'
            class Backend:
                def generate(me, system, prompt, *args, **kwargs):
                    prompts.append(prompt);snapshots.append(s.snapshot())
                    return json.dumps(bad if len(prompts)==1 else good,ensure_ascii=False)
            result=BufferedPhaseManager(Backend()).maybe_review(s,1.,force=True)
            self.assertEqual(snapshots,[before,before])
            self.assertIn('groups[0].phase_summary length 82 exceeds maximum 80; shorten it',prompts[1])
            self.assertEqual(result['attempts'],2)
            self.assertEqual(s.snapshot()['phases']['PH1']['summary'],good['groups'][0]['phase_summary'])

    def test_bad_refresh_id_is_repaired_or_rejected_without_mutating_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'f1.jpg';p.write_bytes(b'image')
            s=BufferedPhaseStore();s.write(now=1.,frames=[{'id':'f1','time':1.,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}],
                events=[{'id':'E1','caption':'观察','frame_ids':['f1'],'start':1.,'end':1.}])
            before=s.snapshot();calls=[]
            class Backend:
                def generate(self,*args,**kwargs):
                    calls.append(1)
                    return json.dumps({'groups':[],'deferred_event_ids':['E1'],'reason':'等待','refreshes':[
                        {'phase_id':[],'phase_title':'错误','phase_summary':'错误','episode_id':'P1','episode_title':'错误','episode_summary':'错误'}]})
            with self.assertRaises(ValueError):BufferedPhaseManager(Backend()).maybe_review(s,1.,force=True)
            self.assertEqual(len(calls),3);self.assertEqual(s.snapshot(),before)

    def test_source_id_mapping_survives_collapse_and_continuation_refreshes_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames={}
            for i in range(1,7):
                p=Path(tmp)/f'f{i}.jpg';p.write_bytes(str(i).encode())
                frames[f'f{i}']={'id':f'f{i}','time':float(i),'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            def raw(i,fs,op='new'):
                return {'source_event_id':f'E{i}','op':op,'caption':f'观察{i}',
                        'frame_ids':fs,'start':frames[fs[0]]['time'],'end':frames[fs[-1]]['time']}
            s=BufferedPhaseStore();mapping={}
            ingest(s,{'now':3.,'frame_ids':['f1','f2','f3'],
                      'events':[raw(1,['f1']),raw(2,['f2']),raw(3,['f3'])]},frames,mapping)
            s.write(now=3.,phases=[{'id':'PH1','title':'第一段','summary':'第一段','event_ids':['E1','E2'],'status':'open','start':1.,'end':2.},
                                  {'id':'PH2','title':'第二段','summary':'第二段','event_ids':['E3'],'status':'open','start':3.,'end':3.}],
                    episodes=[{'id':'P1','title':'过程','summary':'过程','phase_ids':['PH1','PH2'],'status':'open','start':1.,'end':3.}])
            merged=s.collapse_phase({'phase_id':'PH1','event_ids':['E1','E2'],'caption':'完整描述','phase_summary':'第一段',
                                    'support_frame_ids':['f1','f2']},3.)
            self.assertEqual(merged,'E4')
            ingest(s,{'now':4.,'frame_ids':['f4'],'events':[raw(3,['f3','f4'],'continue')]},frames,mapping)
            self.assertEqual(s.snapshot()['phases']['PH2']['end'],4.)
            self.assertEqual(s.snapshot()['episodes']['P1']['end'],4.)
            new,_=ingest(s,{'now':5.,'frame_ids':['f5'],'events':[raw(4,['f5'])]},frames,mapping)
            self.assertEqual(new,['E5']);self.assertEqual(s.snapshot()['events']['E4']['caption'],'完整描述')
            self.assertEqual(s.snapshot()['roots'],['E5'])
            before=s.snapshot()
            with self.assertRaisesRegex(ValueError,'collapsed'):
                ingest(s,{'now':6.,'frame_ids':['f6'],'events':[raw(1,['f1','f6'],'continue')]},frames,mapping)
            self.assertEqual(s.snapshot(),before)


if __name__=='__main__':unittest.main()
