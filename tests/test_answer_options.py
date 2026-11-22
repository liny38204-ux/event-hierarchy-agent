import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from event_memory.answer_options import reader_system, parse_option

class OptionsTests(unittest.TestCase):
    def test_each_arity_rejects_nonexistent_options(self):
        for n in (2,3,4,5):
            options=['x']*n
            for i in range(n):self.assertEqual(parse_option(chr(65+i),options),chr(65+i))
            self.assertIsNone(parse_option(chr(65+n),options))
            for invalid in ('a','A.','Answer: A','AB',None):self.assertIsNone(parse_option(invalid,options))
            self.assertIn(chr(64+n),reader_system(options))
        self.assertEqual(parse_option(' E\n',['x']*5),'E')
    def test_four_option_prompt_preserved(self):
        self.assertEqual(reader_system(['x']*4),
            'Answer the multiple-choice question using the supplied text memory of the video. '
            'Choose the best supported option. Output exactly one uppercase option letter A, B, C, or D. '
            'Do not output explanations or any other text.')

if __name__=='__main__':unittest.main()
