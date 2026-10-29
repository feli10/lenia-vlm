"""Pairwise VLM judge with a persistent cache (adapted from vlm-pref-evo's src/vpe/judge.py).

Every query is (prompt, first image, second image, sample). A Verdict holds p_first: the judge's
probability that the FIRST image is preferred. Swapping the order is a separate query, so position
bias is measurable.

Backends
  decisions  OpenRouter Decisions API (e.g. openai/gpt-6-luna-decisions): a probability per option.
  jev        TypeSafe System One (Jev). Image input is experimental ("Dreamer-ready"): needs the
             TypeSafe-Danger-Zone-Images header; PNG/JPEG stills, 280-1568 px per side.
  fake       Offline stand-in for tests: prefers larger, smoother creatures, with a first-slot bias
             and noise. Costs nothing.

Keys come from the environment or the repo's .env (OPENROUTER_API_KEY, TYPESAFE_API_KEY); they are
never logged or written anywhere.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import random
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .prompts import get_prompt, render_prompt

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_USD_PER_TOKEN = 0.042 / 1e6
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 524, 529}
FATAL_STATUS = {401, 402, 403}
KEY_ENV = {"decisions": "OPENROUTER_API_KEY", "jev": "TYPESAFE_API_KEY"}


class BudgetError(RuntimeError):
    pass


class FatalAPIError(RuntimeError):
    pass


@dataclass
class Verdict:
    p_first: float | None  # P(first image preferred); None if the call failed
    choice: str | None     # "first" | "second" | None
    raw: str
    cost: float = 0.0
    cached: bool = False
    snapshot: str | None = None   # the exact model version the provider reports (e.g. jev-1.14-20261014)

    @property
    def ok(self) -> bool:
        return self.p_first is not None


@dataclass
class Query:
    prompt_id: str
    first: np.ndarray      # (H, W, 3) uint8
    second: np.ndarray
    sample: int = 0


@dataclass
class ChoiceQuery:
    image: np.ndarray      # (H, W, 3) uint8
    instructions: str      # "{W}" is replaced by the backend's name for the image
    options: dict          # option id -> description, in the order shown
    sample: int = 0


def load_env(path) -> None:
    """Put KEY=value lines from a .env file into os.environ (existing variables win). Values are not echoed."""
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k.strip() and v and k.strip() not in os.environ:
            os.environ[k.strip()] = v


def image_hash(img: np.ndarray) -> str:
    h = hashlib.sha1(np.ascontiguousarray(img).tobytes())
    h.update(str(img.shape).encode())
    return h.hexdigest()[:20]


def png_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def parse_probs(answer: dict) -> tuple[float | None, str | None]:
    """A Decisions / Jev choice answer -> (P(first), choice)."""
    probs = answer.get("probabilities") or {}
    pf, ps = float(probs.get("first", 0) or 0), float(probs.get("second", 0) or 0)
    choice = answer.get("choice")
    if pf + ps > 0:
        p_first = pf / (pf + ps)
    elif choice in ("first", "second"):
        p_first = 1.0 if choice == "first" else 0.0
    else:
        return None, None
    if choice not in ("first", "second"):
        choice = "first" if p_first >= 0.5 else "second"
    return p_first, choice


# ---------- cache ----------

class Cache:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, p_first REAL, choice TEXT,"
                         " raw TEXT, cost REAL, model TEXT, prompt_id TEXT, created REAL)")
        # model versions live in their own table, so the verdicts table (and older code reading it) is unchanged
        self._db.execute("CREATE TABLE IF NOT EXISTS snapshots (key TEXT PRIMARY KEY, snapshot TEXT)")
        self._db.commit()

    def get(self, key: str) -> Verdict | None:
        with self._lock:
            row = self._db.execute("SELECT p_first, choice, raw, cost FROM verdicts WHERE key=?", (key,)).fetchone()
        return None if row is None else Verdict(row[0], row[1], row[2], row[3] or 0.0, cached=True)

    def put(self, key: str, v: Verdict, model: str, prompt_id: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO verdicts VALUES (?,?,?,?,?,?,?,?)",
                             (key, v.p_first, v.choice, v.raw, v.cost, model, prompt_id, time.time()))
            if v.snapshot:
                self._db.execute("INSERT OR REPLACE INTO snapshots VALUES (?,?)", (key, v.snapshot))
            self._db.commit()

    def snapshots(self) -> dict:
        """Model version -> number of cached answers it gave."""
        with self._lock:
            return dict(self._db.execute("SELECT snapshot, COUNT(*) FROM snapshots GROUP BY snapshot").fetchall())

    def total_cost(self) -> float:
        """Everything this cache has paid for: every call of the run it belongs to, across resumes."""
        with self._lock:
            return float(self._db.execute("SELECT COALESCE(SUM(cost), 0) FROM verdicts").fetchone()[0])


# ---------- judge ----------

class Judge:
    def __init__(self, name: str, jcfg: dict, cache_path, yes: bool = False, log=None, verbose: bool = False):
        self.name, self.cfg = name, jcfg
        self.backend = jcfg["backend"]
        if self.backend not in ("decisions", "jev", "fake"):
            raise ValueError(f"unknown judge backend {self.backend!r}")
        self.model = jcfg.get("model", "fake")
        self.cache = Cache(cache_path)
        self.yes, self.verbose = yes, verbose
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.spent = 0.0
        self.cap = float(jcfg.get("max_run_usd") or 0)       # hard cap on this run's total spend (0 = none)
        self.snapshot = None                                   # the model version seen first in this process

    def _check_cap(self, n: int, est: float, desc: str) -> None:
        """Refuse a batch that would take the run past max_run_usd. Not overridden by yes=True."""
        if not self.cap or not n:
            return
        so_far = self.cache.total_cost()
        if so_far + est > self.cap:
            raise BudgetError(f"[{desc}] this run has spent ${so_far:.2f} (its judge cache); {n} more calls (~${est:.2f}) "
                              f"would pass the cap of ${self.cap:.2f}. Nothing was called. Re-run with a higher --max-usd: "
                              "the run resumes where it stopped.")

    def _note(self, v: "Verdict", desc: str) -> None:
        """Spend, and a warning if the provider's model version changes during a run."""
        self.spent += v.cost
        if v.snapshot:
            if self.snapshot is None:
                self.snapshot = v.snapshot
            elif v.snapshot != self.snapshot:
                self.log(f"[{desc}] WARNING: model version changed mid-run: {self.snapshot} -> {v.snapshot}")
                self.snapshot = v.snapshot

    def _all_failed(self, results: list, desc: str) -> None:
        """A batch where every new call failed (e.g. an unknown model name) stops the run instead of reading as
        a batch of rejections."""
        if len(results) >= 4 and not any(v.ok for v in results):
            raise FatalAPIError(f"[{desc}] all {len(results)} new calls failed; first error: {results[0].raw[:300]}")

    def _key_value(self) -> str | None:
        env = KEY_ENV.get(self.backend)
        return os.environ.get(env) if env else None

    def key(self, q: Query) -> str:
        payload = {"backend": self.backend, "model": self.model, "prompt": get_prompt(q.prompt_id),
                   "a": image_hash(q.first), "b": image_hash(q.second), "sample": q.sample}
        if self.backend == "fake":
            payload["fake"] = [self.cfg.get("fake_position_bias"), self.cfg.get("fake_noise")]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def judge_many(self, queries: list, desc: str = "judge") -> list[Verdict]:
        keys = [self.key(q) for q in queries]
        out = [self.cache.get(k) for k in keys]
        uniq, seen = [], set()
        for i, v in enumerate(out):
            if v is None and keys[i] not in seen:
                seen.add(keys[i])
                uniq.append(i)
        est = len(uniq) * float(self.cfg.get("est_cost_per_call_usd", 0))
        self.log(f"[{desc}] {len(queries)} queries: {len(queries) - len(uniq)} cached, {len(uniq)} new (est. ${est:.3f})")
        self._check_cap(len(uniq), est, desc)
        if est > float(self.cfg.get("budget_usd", 2)) and not self.yes:
            raise BudgetError(f"estimated ${est:.2f} for {len(uniq)} calls exceeds budget_usd={self.cfg.get('budget_usd')}; "
                              "re-run with --yes")
        if uniq and self.backend != "fake" and not self._key_value():
            raise FatalAPIError(f"{KEY_ENV[self.backend]} is not set (put it in .env)")
        results = {}
        if uniq:
            workers = 1 if self.backend == "fake" else int(self.cfg.get("max_workers", 8))
            done, step = 0, max(1, len(uniq) // 10)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(self._call, queries[i]): i for i in uniq}
                for fut in as_completed(futs):
                    i = futs[fut]
                    v = fut.result()           # FatalAPIError propagates and stops the run
                    if v.ok:
                        self.cache.put(keys[i], v, self.model, queries[i].prompt_id)
                    elif self.verbose or done < 3:
                        self.log(f"[{desc}] failed call: {v.raw[:300]}")
                    self._note(v, desc)
                    results[keys[i]] = v
                    done += 1
                    if done % step == 0 or done == len(uniq):
                        self.log(f"[{desc}] {done}/{len(uniq)} done, spent ${self.spent:.4f}")
        self._all_failed(list(results.values()), desc)
        final = [v if v is not None else results[k] for v, k in zip(out, keys)]
        n_fail = sum(not v.ok for v in final)
        if n_fail:
            self.log(f"[{desc}] WARNING: {n_fail} queries failed (not cached; re-run to retry)")
        return final

    # -- multiple choice: one image, a menu of options
    def choose_many(self, queries: list, desc: str = "choose") -> list:
        """queries: [ChoiceQuery]. Returns [dict option id -> probability, or None if the call failed]."""
        keys = []
        for q in queries:
            payload = {"backend": self.backend, "model": self.model, "kind": "choice", "instructions": q.instructions,
                       "options": list(q.options.items()), "img": image_hash(q.image), "sample": q.sample}
            keys.append(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
        out = [self.cache.get(k) for k in keys]
        todo = [i for i, v in enumerate(out) if v is None]
        est = len(todo) * float(self.cfg.get("est_cost_per_call_usd", 0))
        self.log(f"[{desc}] {len(queries)} queries: {len(queries) - len(todo)} cached, {len(todo)} new (est. ${est:.3f})")
        self._check_cap(len(todo), est, desc)
        if est > float(self.cfg.get("budget_usd", 2)) and not self.yes:
            raise BudgetError(f"estimated ${est:.2f} exceeds budget_usd; re-run with --yes")
        if todo and self.backend != "fake" and not self._key_value():
            raise FatalAPIError(f"{KEY_ENV[self.backend]} is not set (put it in .env)")
        res = {}
        if todo:
            workers = 1 if self.backend == "fake" else int(self.cfg.get("max_workers", 8))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(self._choose, queries[i], keys[i]): i for i in todo}
                for n, fut in enumerate(as_completed(futs), 1):
                    i = futs[fut]
                    v = fut.result()
                    if v.ok:
                        self.cache.put(keys[i], v, self.model, "choice")
                    elif n <= 3:
                        self.log(f"[{desc}] failed call: {v.raw[:300]}")
                    self._note(v, desc)
                    res[i] = v
        self._all_failed(list(res.values()), desc)
        final = [res.get(i, v) for i, v in enumerate(out)]
        probs = []
        for v in final:
            try:
                probs.append(json.loads(v.raw)["probs"] if v is not None and v.ok else None)
            except (ValueError, KeyError, TypeError):
                probs.append(None)
        n_fail = sum(p is None for p in probs)
        if n_fail:
            self.log(f"[{desc}] WARNING: {n_fail} queries failed (not cached; re-run to retry)")
        return probs

    def _choose(self, q, key: str) -> "Verdict":
        ids = list(q.options)
        if self.backend == "fake":                 # deterministic, image-dependent, with a first-option bias
            rng = random.Random(key)
            h = int(image_hash(q.image)[:6], 16)
            logits = [rng.gauss(0, 1) + (1.0 if j == 0 else 0) + (1.5 if (h + j) % len(ids) == 0 else 0) for j in range(len(ids))]
            z = [math.exp(x) for x in logits]
            pr = {o: z[j] / sum(z) for j, o in enumerate(ids)}
            return Verdict(1.0, "first", json.dumps({"probs": pr}), 0.0)
        api_key = self._key_value()
        crit = {o: t for o, t in q.options.items()}
        if self.backend == "decisions":
            img = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png_b64(q.image)}}
            body = {"model": self.model, "state": ["World:", img],
                    "questions": {"match": {"type": "choice", "instructions": q.instructions.replace("{W}", "the image"),
                                            "criteria": crit}}}
            data = self._post(DECISIONS_URL, body, {"Authorization": f"Bearer {api_key}", "X-Title": "lenia-vlm"})
            cost_of = lambda d: float((d.get("usage") or {}).get("cost") or 0)
        else:
            rid = str(uuid.UUID(hex=image_hash(q.image).ljust(32, "0")))
            body = {"model": self.model, "state": {"world": rid},
                    "assets": {rid: {"type": "image", "format": "png", "data_base64": png_b64(q.image)}},
                    "questions": {"match": {"type": "choice", "instructions": q.instructions.replace("{W}", "`world`"),
                                            "criteria": crit}}}
            data = self._post(JEV_URL, body, {"Authorization": f"Bearer {api_key}", "TypeSafe-Danger-Zone-Images": "true"})
            cost_of = lambda d: float((d.get("usage") or {}).get("input_tokens") or 0) * JEV_USD_PER_TOKEN
        if "_error" in data:
            return Verdict(None, None, data["_error"])
        ans = (data.get("answers") or {}).get("match") or {}
        probs = {o: float(v or 0) for o, v in (ans.get("probabilities") or {}).items() if o in crit}
        if sum(probs.values()) <= 0:
            if ans.get("choice") in crit:
                probs = {o: float(o == ans["choice"]) for o in crit}
            else:
                return Verdict(None, None, json.dumps(data)[:2000])
        tot = sum(probs.values())
        probs = {o: probs.get(o, 0.0) / tot for o in crit}
        return Verdict(1.0, "first", json.dumps({"probs": probs, "choice": ans.get("choice")}), cost_of(data),
                       snapshot=data.get("model"))

    # -- internals
    def _post(self, url: str, body: dict, headers: dict) -> dict:
        data = json.dumps(body).encode()
        last = ""
        for attempt in range(int(self.cfg.get("max_retries", 6)) + 1):
            req = urllib.request.Request(url, data=data, method="POST",
                                         headers={"Content-Type": "application/json", **headers})
            try:
                with urllib.request.urlopen(req, timeout=float(self.cfg.get("timeout_s", 120))) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}"
                if e.code in FATAL_STATUS:
                    raise FatalAPIError(last)
                if e.code not in RETRY_STATUS:
                    return {"_error": last}
            except Exception as e:                       # network error: retry
                last = f"{type(e).__name__}: {e}"
            time.sleep(min(60.0, 2 ** attempt + random.random()))
        return {"_error": last}

    def _call(self, q: Query) -> Verdict:
        if self.backend == "fake":
            return self._call_fake(q)
        key = self._key_value()
        if self.backend == "decisions":
            p = render_prompt(q.prompt_id, "the first image", "the second image")
            img = lambda a: {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png_b64(a)}}
            body = {"model": self.model, "state": ["First image:", img(q.first), "Second image:", img(q.second)],
                    "questions": {"preference": {"type": "choice", "instructions": p["instructions"],
                                                 "criteria": {"first": p["first"], "second": p["second"]}}}}
            data = self._post(DECISIONS_URL, body, {"Authorization": f"Bearer {key}", "X-Title": "lenia-vlm"})
            cost_of = lambda d: float((d.get("usage") or {}).get("cost") or 0)
        else:   # jev
            p = render_prompt(q.prompt_id, "`first`", "`second`")
            ra, rb = (str(uuid.UUID(hex=image_hash(a).ljust(32, "0"))) for a in (q.first, q.second))
            if ra == rb:
                rb = str(uuid.UUID(hex=hashlib.md5((rb + "b").encode()).hexdigest()))
            body = {"model": self.model, "state": {"first": ra, "second": rb},
                    "assets": {ra: {"type": "image", "format": "png", "data_base64": png_b64(q.first)},
                               rb: {"type": "image", "format": "png", "data_base64": png_b64(q.second)}},
                    "questions": {"preference": {"type": "choice", "instructions": p["instructions"],
                                                 "criteria": {"first": p["first"], "second": p["second"]}}}}
            data = self._post(JEV_URL, body, {"Authorization": f"Bearer {key}", "TypeSafe-Danger-Zone-Images": "true"})
            cost_of = lambda d: float((d.get("usage") or {}).get("input_tokens") or 0) * JEV_USD_PER_TOKEN
        if "_error" in data:
            return Verdict(None, None, data["_error"])
        ans = (data.get("answers") or {}).get("preference") or {}
        pf, choice = parse_probs(ans)
        if pf is None:
            return Verdict(None, None, json.dumps(data)[:2000])
        return Verdict(pf, choice, json.dumps(ans), cost_of(data), snapshot=data.get("model"))

    def _call_fake(self, q: Query) -> Verdict:
        def score(img):
            x = img.astype(np.float32).mean(-1) / 255
            ink = (x < 0.9).mean()
            rough = np.abs(np.diff(x, axis=0)).mean() + np.abs(np.diff(x, axis=1)).mean()
            return 4 * ink - 20 * rough
        rng = random.Random(self.key(q))
        logit = 3.0 * (score(q.first) - score(q.second)) + float(self.cfg.get("fake_position_bias", 0.3))
        logit += rng.gauss(0, float(self.cfg.get("fake_noise", 0.5)))
        pf = 1 / (1 + math.exp(-logit))
        return Verdict(pf, "first" if pf >= 0.5 else "second", json.dumps({"fake_logit": logit}), 0.0)
