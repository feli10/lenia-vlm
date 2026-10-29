"""The generative VLM interface (lenia.describe): prompt rendering, JSON parsing, cache keys, caching, spend cap,
and the OpenRouter request shape. Offline. Runs under pytest or as a plain script (python3 tests/test_describe.py).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lenia.describe import Ask, Describer, parse, render        # noqa: E402
from lenia.judge import BudgetError                             # noqa: E402
from lenia.prompts import NAMER, REVIEW                         # noqa: E402

IMGS = [np.full((64, 64, 3), 40 * k, np.uint8) for k in range(5)]


def test_render_and_parse():
    s, u = render(NAMER, {"names": "(none yet)"})
    assert "Image 1 is the flagged world" in u and u.endswith("(none yet)")
    try:
        render(REVIEW, {"mutants": 16})
        assert False, "unfilled fields must raise"
    except ValueError:
        pass
    assert parse('```json\n{"sees": "a ring", "push": "make two rings"}\n```', ["sees", "push"]) == \
        {"sees": "a ring", "push": "make two rings"}
    assert parse('{"sees": "", "push": "x"}', ["sees", "push"]) is None
    assert parse("no json here", ["sees"]) is None


def test_cache_cap_keys():
    with tempfile.TemporaryDirectory() as d:
        dsc = Describer("fake", {"backend": "fake"}, Path(d) / "c.sqlite", log=lambda m: None)
        a = Ask(NAMER, IMGS, {"names": "(none yet)"})
        [x] = dsc.ask_many([a])
        assert x.ok and set(x.data) == {"sees", "push"} and not x.cached
        [y] = dsc.ask_many([a])
        assert y.cached and y.data == x.data
        assert dsc.key(a) != dsc.key(Ask(NAMER, IMGS, {"names": "a ring"}))          # names already given key it
        assert dsc.key(a) != dsc.key(Ask(NAMER, IMGS[::-1], {"names": "(none yet)"}))  # so does picture order
        d2 = Describer("fake", {"backend": "fake", "temperature": 0.2}, Path(d) / "c.sqlite", log=lambda m: None)
        assert d2.key(a) != dsc.key(a)                                                 # and the settings
        capped = Describer("fake", {"backend": "fake", "est_cost_per_call_usd": 1.0}, Path(d) / "k.sqlite", max_usd=0.5,
                           log=lambda m: None)
        try:
            capped.ask_many([a])
            assert False, "the cap must refuse"
        except BudgetError:
            pass


def test_openrouter_request():
    seen = {}

    class R:
        def __init__(self, b): self.b = b
        def __enter__(self): return self
        def __exit__(self, *x): pass
        def read(self): return self.b

    import urllib.request
    old = urllib.request.urlopen

    def fake_open(req, timeout=None):
        seen["body"] = json.loads(req.data.decode())
        return R(json.dumps({"model": "q", "choices": [{"message": {"content": '{"sees": "s", "push": "make p"}'}}],
                             "usage": {"prompt_tokens": 900, "completion_tokens": 30, "cost": 0.0004}}).encode())
    urllib.request.urlopen = fake_open
    os.environ.setdefault("OPENROUTER_API_KEY", "x")
    try:
        with tempfile.TemporaryDirectory() as d:
            dsc = Describer("q", {"backend": "openrouter", "model": "qwen/x", "max_output_tokens": 100}, Path(d) / "c.sqlite",
                            log=lambda m: None)
            [x] = dsc.ask_many([Ask(NAMER, IMGS, {"names": "(none yet)"})])
    finally:
        urllib.request.urlopen = old
    assert x.ok and x.data == {"sees": "s", "push": "make p"} and abs(x.cost - 0.0004) < 1e-12
    b = seen["body"]
    assert b["model"] == "qwen/x" and b["max_tokens"] == 100 and b["usage"] == {"include": True}
    c = b["messages"][1]["content"]
    assert [p["type"] for p in c] == ["image_url"] * 5 + ["text"] and b["messages"][0]["role"] == "system"


def test_vertex_rest_request():
    """The REST path (no google-genai SDK): request body, thought parts skipped, usage and cost."""
    import types as _t
    seen = {}

    class R:
        def __init__(self, b): self.b = b
        def __enter__(self): return self
        def __exit__(self, *x): pass
        def read(self): return self.b

    import urllib.request
    old = urllib.request.urlopen

    def fake_open(req, timeout=None):
        seen["url"], seen["body"], seen["auth"] = req.full_url, json.loads(req.data.decode()), req.headers.get("Authorization")
        return R(json.dumps({"modelVersion": "gemini-x", "candidates": [{"content": {"parts": [
            {"text": "thinking...", "thought": True}, {"text": '{"sees": "s", "push": "make p"}'}]}}],
            "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 20, "thoughtsTokenCount": 5}}).encode())
    urllib.request.urlopen = fake_open
    os.environ["GOOGLE_CLOUD_PROJECT"] = "proj-x"
    try:
        with tempfile.TemporaryDirectory() as d:
            dsc = Describer("v", {"backend": "vertex", "model": "gemini-x", "thinking": {"level": "low"}, "temperature": 1.0,
                                  "max_output_tokens": 100, "usd_per_m_input": 2.0, "usd_per_m_output": 10.0},
                            Path(d) / "c.sqlite", log=lambda m: None)
            dsc._client = _t.SimpleNamespace(valid=True, token="tok")        # stands in for google-auth credentials
            text, usage, cost, ver = dsc._vertex_rest("sys", "user text", [b"png1", b"png2"])
    finally:
        urllib.request.urlopen = old
    assert text == '{"sees": "s", "push": "make p"}' and ver == "gemini-x"
    assert usage == {"input": 1000, "output": 20, "thinking": 5} and abs(cost - (1000 * 2 + 25 * 10) / 1e6) < 1e-12
    assert seen["url"] == ("https://aiplatform.googleapis.com/v1/projects/proj-x/locations/global/publishers/google/models/"
                           "gemini-x:generateContent") and seen["auth"] == "Bearer tok"
    b = seen["body"]
    assert b["systemInstruction"]["parts"][0]["text"] == "sys" and b["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}
    parts = b["contents"][0]["parts"]
    assert [("inlineData" in p) for p in parts] == [True, True, False] and parts[-1]["text"] == "user text"


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print(f"{name}: ok")
