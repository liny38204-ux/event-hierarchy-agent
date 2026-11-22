"""Use an explicitly configured Chat Completions vision model in the same QA loop."""
import base64
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .answer_options import option_letters


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ChatReader:
    def __init__(self, model, log_path):
        self.model = model
        self.log_path = Path(log_path)
        self.base_url = os.environ["CHAT_API_BASE_URL"].rstrip("/")
        url = urlsplit(self.base_url)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("CHAT_API_BASE_URL must be a server URL with an optional API path.")
        self._key = os.environ["CHAT_API_KEY"]
        self._opener = build_opener(_NoRedirect())
        self.calls = []

    def generate(self, system, text, evidence=(), *, stage, max_tokens=2048):
        content = []
        for item in evidence:
            data = Path(item["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("Evidence image differs from the stored checksum.")
            media = "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"
            image_url = f"data:{media};base64," + base64.b64encode(data).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        content.append({"type": "text", "text": text})
        payload = {"model": self.model, "max_completion_tokens": max(1200, 3 * max_tokens),
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        req = Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"})
        started = time.monotonic()
        with self._opener.open(req, timeout=240) as response:
            result = json.load(response)
        choice = result["choices"][0]
        raw = choice["message"]["content"]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("The chat model returned no answer.")
        usage = result.get("usage") or {}
        record = {"stage": stage, "system": system, "text": text, "evidence": list(evidence),
                  "raw": raw, "reported_model": result.get("model"), "finish_reason": choice.get("finish_reason"),
                  "seconds": time.monotonic() - started, "input_tokens": usage.get("prompt_tokens"),
                  "output_tokens": usage.get("completion_tokens")}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.calls.append(record)
        return raw

    def choose_option(self, system, text, *, evidence, options, stage):
        raw = self.generate(system, text + "\nReturn only one supplied option letter.", evidence,
                            stage=stage, max_tokens=16).strip()
        if raw not in option_letters(options):
            raise ValueError("The chat model did not submit a valid option.")
        return raw
