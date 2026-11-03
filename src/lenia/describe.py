"""The generative VLM ("CEO"): rare calls that look at pictures and write (names, reviews). One interface, several
backends, a persistent cache and a hard spend cap, in the style of lenia.judge.

  d = Describer("gemini-flash", cfg["vlms"]["gemini-flash"], out / "describe_cache.sqlite", max_usd=1.0)
  [a] = d.ask_many([Ask(NAMER, [world, t1, t2, t3, t4], {"names": "(none yet)"})])
  a.data -> {"sees": ..., "push": ...}; a.cost, a.usage

Backends
  vertex      Gemini on Vertex AI (google-genai SDK, Application Default Credentials; bills the Google Cloud project,
              so it uses that project's credits). Needs `uv add google-genai` and `gcloud auth application-default login`.
  openrouter  any OpenAI-style chat model on OpenRouter (open weights: Qwen3-VL, Gemma). OPENROUTER_API_KEY in .env.
  fake        offline, deterministic, free (tests and dry runs).
  Another provider (e.g. Sam's API) is one more method with the signature of _vertex / _openrouter: take the system text,
  the user text and PNG bytes; return (text, usage dict, cost in USD, model version).

Prompts live in lenia.prompts.GEN (system text, user template with {fields}, the JSON fields to return); the full
texts, the filled-in values, the picture bytes, the model and its settings key the cache, so editing a prompt starts
fresh answers (tests/test_golden.py fails on any change). Answers must be one JSON object with the prompt's fields;
anything else counts as a failed call (not cached, retried on the next run).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .judge import BudgetError, FatalAPIError, image_hash
from .prompts import GEN

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 524, 529}
FATAL_STATUS = {401, 402, 403}


@dataclass
class Ask:
    prompt_id: str
    images: list                 # [(H, W, 3) uint8], in the order the prompt describes them
    fill: dict = field(default_factory=dict)   # values for the {fields} of the user template
    sample: int = 0


@dataclass
class Answer:
    data: dict | None            # the parsed JSON object; None if the call failed
    raw: str
    cost: float = 0.0
    usage: dict = field(default_factory=dict)
    version: str | None = None
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.data is not None


def png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def render(prompt_id: str, fill: dict) -> tuple[str, str]:
    p = GEN[prompt_id]
    user = p["user"]
    for k, v in fill.items():
        user = user.replace("{" + k + "}", str(v))
    left = re.findall(r"\{[a-z_]+\}", user)
    if left:
        raise ValueError(f"{prompt_id}: unfilled fields {left}")
    return p["system"], user


def parse(text: str, fields: list) -> dict | None:
    """The first JSON object in text (code fences allowed) that has every field as a non-empty string."""
    for m in re.finditer(r"\{.*\}", text or "", re.S):
        try:
            d = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(d, dict) and all(isinstance(d.get(f), str) and d[f].strip() for f in fields):
            return {f: d[f].strip() for f in fields} | {k: v for k, v in d.items() if k not in fields}
    return None


class Cache:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, data TEXT, raw TEXT, cost REAL,"
                         " usage TEXT, version TEXT, model TEXT, prompt_id TEXT, created REAL)")
        self._db.commit()

    def get(self, key: str) -> Answer | None:
        with self._lock:
            r = self._db.execute("SELECT data, raw, cost, usage, version FROM answers WHERE key=?", (key,)).fetchone()
        return None if r is None else Answer(json.loads(r[0]), r[1], r[2] or 0.0, json.loads(r[3] or "{}"), r[4], cached=True)

    def put(self, key: str, a: Answer, model: str, prompt_id: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO answers VALUES (?,?,?,?,?,?,?,?,?)",
                             (key, json.dumps(a.data), a.raw, a.cost, json.dumps(a.usage), a.version, model, prompt_id,
                              time.time()))
            self._db.commit()

    def total_cost(self) -> float:
        with self._lock:
            return float(self._db.execute("SELECT COALESCE(SUM(cost), 0) FROM answers").fetchone()[0])


class Describer:
    def __init__(self, name: str, cfg: dict, cache_path, max_usd: float = 0.0, log=None, cache_only: bool = False):
        self.name, self.cfg = name, cfg
        self.cache_only = cache_only           # answer from the cache only, like LENIA_VLM_CACHE_ONLY but for this describer
        self.backend = cfg["backend"]
        if self.backend not in ("vertex", "openrouter", "fake"):
            raise ValueError(f"unknown VLM backend {self.backend!r}")
        self.model = cfg.get("model", "fake")
        self.cache = Cache(cache_path)
        self.cap = float(max_usd or 0)
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.spent = 0.0
        self._client = None

    def settings(self) -> dict:
        """Everything about the call besides the prompt and pictures that can change an answer."""
        return {k: self.cfg.get(k) for k in ("temperature", "max_output_tokens", "thinking", "extra")}

    def key(self, a: Ask) -> str:
        payload = {"backend": self.backend, "model": self.model, "settings": self.settings(), "prompt": GEN[a.prompt_id],
                   "fill": a.fill, "images": [image_hash(x) for x in a.images], "sample": a.sample}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def ask_many(self, asks: list, desc: str = "describe") -> list:
        keys = [self.key(a) for a in asks]
        out = [self.cache.get(k) for k in keys]
        todo, seen = [], set()
        for i, v in enumerate(out):
            if v is None and keys[i] not in seen:
                seen.add(keys[i])
                todo.append(i)
        est = len(todo) * float(self.cfg.get("est_cost_per_call_usd", 0))
        self.log(f"[{desc} {self.name}] {len(asks)} asks: {len(asks) - len(todo)} cached, {len(todo)} new (est. ${est:.3f})")
        if self.cap and todo:
            so_far = self.cache.total_cost()
            if so_far + est > self.cap:
                raise BudgetError(f"[{desc} {self.name}] spent ${so_far:.2f}; {len(todo)} more calls (~${est:.2f}) would pass "
                                  f"the cap of ${self.cap:.2f}. Nothing was called; re-run with a higher cap.")
        res = {}
        if todo and (self.cache_only or os.environ.get("LENIA_VLM_CACHE_ONLY")):           # answer from the cache only; make no calls
            self.log(f"[{desc} {self.name}] LENIA_VLM_CACHE_ONLY: {len(todo)} uncached asks left unanswered")
            res = {keys[i]: Answer(None, "not cached (LENIA_VLM_CACHE_ONLY)") for i in todo}
            todo = []
        if todo:
            workers = 1 if self.backend == "fake" else int(self.cfg.get("max_workers", 4))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(self._ask, asks[i], keys[i]): i for i in todo}
                for n, fut in enumerate(as_completed(futs), 1):
                    i = futs[fut]
                    a = fut.result()
                    self.spent += a.cost
                    if a.ok:
                        self.cache.put(keys[i], a, self.model, asks[i].prompt_id)
                    elif n <= 3:
                        self.log(f"[{desc} {self.name}] failed call: {a.raw[:300]}")
                    res[keys[i]] = a
        if len(res) >= 3 and not any(a.ok for a in res.values()) and not (self.cache_only or os.environ.get("LENIA_VLM_CACHE_ONLY")):
            raise FatalAPIError(f"[{desc} {self.name}] all {len(res)} new calls failed; first: {next(iter(res.values())).raw[:300]}")
        final = [v if v is not None else res[k] for v, k in zip(out, keys)]
        n_fail = sum(not a.ok for a in final)
        if n_fail:
            self.log(f"[{desc} {self.name}] WARNING: {n_fail} calls failed (not cached; re-run to retry)")
        return final

    def _ask(self, a: Ask, key: str) -> Answer:
        system, user = render(a.prompt_id, a.fill)
        pngs = [png_bytes(x) for x in a.images]
        try:
            if self.backend == "fake":
                text, usage, cost, ver = self._fake(a.prompt_id, pngs)
            elif self.backend == "vertex":
                text, usage, cost, ver = self._vertex(system, user, pngs)
            else:
                text, usage, cost, ver = self._openrouter(system, user, pngs)
        except FatalAPIError:
            raise
        except ImportError as e:                                 # a missing package is not one bad call: stop
            raise FatalAPIError(f"{e}: install the package for the Python you run (e.g. `python3 -m pip install "
                                "google-genai`; `uv add google-genai` installs into .venv only)")
        except Exception as e:                                   # one failed call, not the run
            return Answer(None, f"{type(e).__name__}: {e}")
        data = parse(text, GEN[a.prompt_id]["fields"])
        return Answer(data, text or "", cost, usage, ver)

    # -- backends: (system, user, pngs) -> (text, usage, cost USD, model version)
    def _price(self, n_in: int, n_out: int) -> float:
        return n_in * float(self.cfg.get("usd_per_m_input", 0)) / 1e6 + n_out * float(self.cfg.get("usd_per_m_output", 0)) / 1e6

    def _vertex(self, system: str, user: str, pngs: list):
        if os.environ.get("LENIA_VERTEX_REST"):                 # force the REST path (to test it where the SDK exists)
            return self._vertex_rest(system, user, pngs)
        try:
            from google import genai                           # the SDK where installed (the Mac: extra `vertex`)
            from google.genai import types
        except ImportError:                                    # Colab: plain REST with its preinstalled google-auth
            return self._vertex_rest(system, user, pngs)
        if self._client is None:
            project = self.cfg.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise FatalAPIError("set GOOGLE_CLOUD_PROJECT in .env (or `project:` under this VLM in the config)")
            self._client = genai.Client(vertexai=True, project=project,
                                        location=self.cfg.get("location") or os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
        th = self.cfg.get("thinking") or {}
        tcfg = None
        if "level" in th:
            tcfg = types.ThinkingConfig(thinking_level=th["level"])
        elif "budget" in th:
            tcfg = types.ThinkingConfig(thinking_budget=int(th["budget"]))
        conf = types.GenerateContentConfig(system_instruction=system, temperature=self.cfg.get("temperature"),
                                           max_output_tokens=self.cfg.get("max_output_tokens"),
                                           response_mime_type="application/json", thinking_config=tcfg)
        parts = [types.Part.from_bytes(data=b, mime_type="image/png") for b in pngs] + [user]
        for attempt in range(int(self.cfg.get("max_retries", 5)) + 1):       # 429 (per-minute quota) and 5xx: back off
            try:
                r = self._client.models.generate_content(model=self.model, contents=parts, config=conf)
                break
            except Exception as e:
                code = getattr(e, "code", None)
                if code in FATAL_STATUS or code == 404:
                    raise FatalAPIError(f"Vertex {code}: {e}")
                if code not in RETRY_STATUS or attempt == int(self.cfg.get("max_retries", 5)):
                    raise
                time.sleep(min(60.0, 4 * 2 ** attempt))
        u = r.usage_metadata
        n_in, n_out, n_th = (getattr(u, k, 0) or 0 for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count"))
        usage = {"input": n_in, "output": n_out, "thinking": n_th}
        return r.text, usage, self._price(n_in, n_out + n_th), getattr(r, "model_version", None)

    def _vertex_rest(self, system: str, user: str, pngs: list):
        """Vertex generateContent over HTTPS with Application Default Credentials (google-auth only), for machines without
        the google-genai SDK (Colab, where installing it upgrades the google-auth that Colab pins). Same request as the
        SDK path, so answers share the cache."""
        import base64
        project = self.cfg.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise FatalAPIError("set GOOGLE_CLOUD_PROJECT in .env (or `project:` under this VLM in the config)")
        loc = self.cfg.get("location") or os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        if self._client is None or not self._client.valid:
            import google.auth                                 # preinstalled on Colab; a dependency of google-genai
            import google.auth.transport.requests
            if self._client is None:
                self._client, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            self._client.refresh(google.auth.transport.requests.Request())
        host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
        url = f"https://{host}/v1/projects/{project}/locations/{loc}/publishers/google/models/{self.model}:generateContent"
        gen = {"responseMimeType": "application/json"}
        if self.cfg.get("temperature") is not None:
            gen["temperature"] = self.cfg["temperature"]
        if self.cfg.get("max_output_tokens"):
            gen["maxOutputTokens"] = self.cfg["max_output_tokens"]
        th = self.cfg.get("thinking") or {}
        if "level" in th:
            gen["thinkingConfig"] = {"thinkingLevel": str(th["level"]).upper()}
        elif "budget" in th:
            gen["thinkingConfig"] = {"thinkingBudget": int(th["budget"])}
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"inlineData": {"mimeType": "image/png", "data": base64.b64encode(b).decode()}}
                                                        for b in pngs] + [{"text": user}]}],
                "generationConfig": gen}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._client.token}"}
        if os.environ.get("GOOGLE_CLOUD_QUOTA_PROJECT"):
            headers["x-goog-user-project"] = os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"]
        data, last = None, ""
        for attempt in range(int(self.cfg.get("max_retries", 5)) + 1):
            req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=float(self.cfg.get("timeout_s", 180))) as r:
                    data = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as e:
                last = f"Vertex HTTP {e.code}: {e.read().decode(errors='replace')[:500]}"
                if e.code in FATAL_STATUS or e.code == 404:
                    raise FatalAPIError(last)
                if e.code not in RETRY_STATUS:
                    raise RuntimeError(last)
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
            time.sleep(min(60.0, 4 * 2 ** attempt))
        if data is None:
            raise RuntimeError(last)
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        u = data.get("usageMetadata") or {}
        n_in, n_out, n_th = (int(u.get(k) or 0) for k in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount"))
        return text, {"input": n_in, "output": n_out, "thinking": n_th}, self._price(n_in, n_out + n_th), data.get("modelVersion")

    def _openrouter(self, system: str, user: str, pngs: list):
        import base64
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise FatalAPIError("OPENROUTER_API_KEY is not set (put it in .env)")
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b).decode()}}
                   for b in pngs] + [{"type": "text", "text": user}]
        body = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "usage": {"include": True}, **(self.cfg.get("extra") or {})}
        for k in ("temperature", "max_output_tokens"):
            if self.cfg.get(k) is not None:
                body["max_tokens" if k == "max_output_tokens" else k] = self.cfg[k]
        if self.cfg.get("thinking"):
            body["reasoning"] = self.cfg["thinking"]
        data, last = None, ""
        for attempt in range(int(self.cfg.get("max_retries", 5)) + 1):
            req = urllib.request.Request(OPENROUTER_URL, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                                                  "X-Title": "lenia-vlm"})
            try:
                with urllib.request.urlopen(req, timeout=float(self.cfg.get("timeout_s", 180))) as r:
                    data = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}"
                if e.code in FATAL_STATUS:
                    raise FatalAPIError(last)
                if e.code not in RETRY_STATUS:
                    raise RuntimeError(last)
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
            time.sleep(min(60.0, 2 ** attempt))
        if data is None:
            raise RuntimeError(last)
        if "error" in data:
            raise RuntimeError(json.dumps(data["error"])[:500])
        u = data.get("usage") or {}
        det = u.get("completion_tokens_details") or {}
        usage = {"input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0),
                 "thinking": det.get("reasoning_tokens", 0), "provider": data.get("provider")}
        cost = float(u.get("cost") or 0) or self._price(usage["input"], usage["output"])
        return data["choices"][0]["message"].get("content") or "", usage, cost, data.get("model")

    def _fake(self, prompt_id: str, pngs: list):
        h = hashlib.sha1(b"".join(pngs)).hexdigest()[:6]
        return json.dumps({f: f"fake {f} {h}" for f in GEN[prompt_id]["fields"]}), {"input": 1000, "output": 50, "thinking": 0}, 0.0, "fake"
