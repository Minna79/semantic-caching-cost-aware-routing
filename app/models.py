"""LLM clients: a free offline mock, and OpenRouter with a disk cache and a hard call budget.

Cost-control rules (built for a tight budget):
* Successful responses are cached on disk, so re-running the same prompt never pays twice.
  The original latency is stored and replayed, so benchmarks still reflect real latency.
* CallBudget stops the process after N real API calls.
* HTTP 402 (out of credits) is never retried.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .text import tokenize

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUS = (429, 500, 502, 503, 504)


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str
    error: str | None = None
    from_disk_cache: bool = False


class BudgetExceeded(RuntimeError):
    pass


class CallBudget:
    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.used = 0

    def spend(self) -> None:
        if self.used >= self.max_calls:
            raise BudgetExceeded(f"LLM call budget of {self.max_calls} reached; raise MAX_LLM_CALLS to continue")
        self.used += 1


class DiskCache:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.commit()

    @staticmethod
    def key(model: str, prompt: str, max_tokens: int) -> str:
        blob = json.dumps([model, prompt, max_tokens], ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        row = self.db.execute("SELECT value FROM llm_cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO llm_cache (key, value) VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False))
        )
        self.db.commit()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockLLM:
    """Deterministic, free stand-in. Answers by copying the context sentence that best matches
    the question. For testing the pipeline only: it says nothing about real model quality/cost."""

    _CHUNK = re.compile(r"^\[([a-z]{2}_\d+)\] (.*)$", re.MULTILINE)
    _SENT = re.compile(r"(?<=[.!?؟؛])\s+")

    def complete(self, prompt: str, model: str, max_tokens: int = 300) -> LLMResponse:
        start = time.perf_counter()
        question = prompt.rsplit("Question:", 1)[-1].strip().splitlines()[0] if "Question:" in prompt else ""
        q_tokens = set(tokenize(question))
        best, best_doc, best_overlap = "", "", 0
        for doc, text in self._CHUNK.findall(prompt):
            for sent in self._SENT.split(text):
                overlap = len(q_tokens & set(tokenize(sent)))
                if overlap > best_overlap:
                    best, best_doc, best_overlap = sent.strip(), doc, overlap
        needed = 2 if "small" in model else 1
        answer = f"{best} [{best_doc}]" if best_overlap >= needed else "I don't know."
        return LLMResponse(
            text=answer,
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(answer),
            latency_s=time.perf_counter() - start,
            model=model,
        )


class OpenRouterLLM:
    def __init__(self, api_key: str | None, budget: CallBudget, cache: DiskCache | None = None, max_retries: int = 4):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set (see .env.example)")
        self.api_key, self.budget, self.cache, self.max_retries = api_key, budget, cache, max_retries

    def complete(self, prompt: str, model: str, max_tokens: int = 300) -> LLMResponse:
        key = DiskCache.key(model, prompt, max_tokens)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return LLMResponse(**hit, from_disk_cache=True)

        self.budget.spend()  # a real call is about to happen
        start = time.perf_counter()
        last_error = "no attempt made"
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(
                    OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
                    timeout=90,
                )
            except requests.RequestException as e:
                last_error = f"network error: {type(e).__name__}"
                time.sleep(min(2**attempt, 30))
                continue

            if r.status_code == 402:  # out of credits: waiting never fixes this
                return self._failure(model, start, "HTTP 402 payment required (out of credits)")
            if r.status_code in RETRY_STATUS:
                last_error = f"HTTP {r.status_code}"
                wait = min(int(r.headers.get("Retry-After", 2**attempt * 2)), 60)
                time.sleep(wait)
                continue
            try:
                data = r.json()
            except ValueError:
                return self._failure(model, start, f"HTTP {r.status_code}: non-JSON body")
            if r.status_code != 200 or "error" in data or not data.get("choices"):
                return self._failure(model, start, f"HTTP {r.status_code}: {data.get('error', data)}"[:300])

            text = (data["choices"][0]["message"].get("content") or "").strip()
            usage = data.get("usage") or {}
            resp = LLMResponse(
                text=text,
                input_tokens=int(usage.get("prompt_tokens") or estimate_tokens(prompt)),
                output_tokens=int(usage.get("completion_tokens") or estimate_tokens(text)),
                latency_s=time.perf_counter() - start,
                model=model,
            )
            if not text:
                resp.error = "empty completion"
            elif self.cache is not None:
                self.cache.put(key, {k: getattr(resp, k) for k in ("text", "input_tokens", "output_tokens", "latency_s", "model", "error")})
            return resp
        return self._failure(model, start, f"gave up after retries ({last_error})")

    @staticmethod
    def _failure(model: str, start: float, error: str) -> LLMResponse:
        return LLMResponse("", 0, 0, time.perf_counter() - start, model, error=error)


def get_llm(settings) -> "MockLLM | OpenRouterLLM":
    if settings.llm_backend == "mock":
        return MockLLM()
    if settings.llm_backend == "openrouter":
        return OpenRouterLLM(
            settings.openrouter_api_key, CallBudget(settings.max_llm_calls), DiskCache(settings.llm_cache_path)
        )
    raise ValueError(f"unknown LLM backend: {settings.llm_backend!r}")
