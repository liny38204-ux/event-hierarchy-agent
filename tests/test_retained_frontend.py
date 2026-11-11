"""No dropped frames or future evidence in the retained-image hierarchy bridge."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from event_memory.caption_runtime import sha256_file
from event_memory.retained_frontend import candidate_batches, verified_frames


class Encoder:
    def encode(self, pixels):
        return [1., 0.] if pixels[0,0,0] < 100 else [0., 1.]


class RetainedFrontendTests(unittest.TestCase):
    def make(self, root, colors):
        rows=[]
        for i,color in enumerate(colors):
            path=root/f'f{i+1:06d}.png';Image.new('RGB',(16,16),(color,0,0)).save(path)
            rows.append({'id':path.stem,'time':float(i),'file':path.name,'sha256':sha256_file(path)})
        return rows

    def test_content_boundary_capacity_split_all_frames_and_closure_time(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=self.make(root,[0,0,0,255,255])
            frames=verified_frames(raw,root,5.);audit=[]
            batches=list(candidate_batches(frames,Encoder(),5.,audit=audit,fits=lambda xs:len(xs)<=2))
            self.assertEqual([f['id'] for b in batches for f in b['evidence']], [f['id'] for f in frames])
            self.assertEqual([b['candidate_id'] for b in batches],['C0001','C0001','C0002'])
            self.assertEqual([b['closure_reason'] for b in batches],['writer_capacity','content_boundary','input_ended'])
            self.assertEqual(batches[1]['closed_at'],3.)
            self.assertEqual(batches[1]['end'],2.)
            self.assertEqual(len(audit),5)

    def test_future_corrupted_and_extra_fields_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=self.make(root,[0,255])
            with self.assertRaises(ValueError):verified_frames(raw,root,.5)
            extra=[dict(raw[0],answer='not allowed')]
            with self.assertRaises(ValueError):verified_frames(extra,root,2.)
            (root/raw[0]['file']).write_bytes(b'corrupted')
            with self.assertRaises(ValueError):verified_frames(raw,root,2.)



if __name__=='__main__':unittest.main()
