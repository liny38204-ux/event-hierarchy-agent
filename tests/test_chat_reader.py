import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from event_memory.chat_reader import ChatReader


class ChatReaderTests(unittest.TestCase):
    def test_images_precede_question_and_only_model_output_is_logged(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
            "CHAT_API_BASE_URL": "https://example.invalid/v1", "CHAT_API_KEY": "unit-test-placeholder",
        }):
            folder = Path(folder)
            image = folder / "frame.jpg"
            image.write_bytes(b"fixture")
            frame = {"path": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
            backend = ChatReader("vision-fixture", folder / "calls.jsonl")
            reply = {"model": "vision-fixture", "choices": [{"message": {"content": "B"}, "finish_reason": "stop"}]}
            with patch.object(backend._opener, "open", return_value=io.BytesIO(json.dumps(reply).encode())) as send:
                answer = backend.choose_option("choose one", "Question", evidence=[frame],
                                               options=["first", "second"], stage="terminal")
            self.assertEqual(answer, "B")
            req = send.call_args.args[0]
            self.assertEqual(req.full_url, "https://example.invalid/v1/chat/completions")
            content = json.loads(req.data)["messages"][1]["content"]
            self.assertEqual([item["type"] for item in content], ["image_url", "text"])
            log = (folder / "calls.jsonl").read_text()
            self.assertNotIn("unit-test-placeholder", log)
            self.assertIsNone(json.loads(log)["input_tokens"])
