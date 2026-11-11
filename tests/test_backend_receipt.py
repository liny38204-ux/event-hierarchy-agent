import json
import unittest
from event_memory.backend_qwen38 import json_ready


class ReceiptTests(unittest.TestCase):
    def test_transformers_set_diagnostics_remain_structured(self):
        info={'missing_keys':set(),'unexpected_keys':{'mtp.a','mtp.b'},
              'mismatched_keys':{('weight',(2,3),(3,2))}}
        saved=json.loads(json.dumps(json_ready(info)))
        self.assertEqual(saved['missing_keys'],[])
        self.assertEqual(saved['unexpected_keys'],['mtp.a','mtp.b'])
        self.assertEqual(saved['mismatched_keys'],[['weight',[2,3],[3,2]]])
