"""Eval data, quality scoring, traffic construction, summaries and the baseline runner."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from .logging_config import RequestLogger, compute_cost
from .rag import RagIndex, generate
from .text import detect_lang, normalize, tokenize

STOPWORDS = set(
    "a an the of in on at to for and or is was were are be been by with as it its that this from which who what "
    "when where how did do does has have had not but into than then also their his her they he she "
    "و در به از که را این با است بود تا برای آن یک هم یا شد می ها های".split()
)
IDK_MARKERS = ("i don t know", "i do not know", "نمی دانم", "نمیدانم")


def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_idk(answer: str) -> bool:
    n = normalize(answer)
    return not n or any(m in n for m in IDK_MARKERS)


def key_recall(expected: str, answer: str) -> float:
    """Share of the expected answer's content words (and numbers) found in the answer.

    Cheap, deterministic and free, but a proxy: it rewards covering the expected facts and
    cannot judge wording or contradictions. 0.0 for empty or 'I don't know' answers."""
    if is_idk(answer):
        return 0.0
    keys = {t for t in tokenize(expected) if t not in STOPWORDS}
    if not keys:
        return 0.0
    got = set(tokenize(answer))

    def present(t: str) -> bool:
        return t in got or t + "s" in got or (t.endswith("s") and t[:-1] in got)

    return sum(present(t) for t in keys) / len(keys)


def build_traffic(items: list[dict], paraphrases: list[dict], repeats: int = 40, seed: int = 0) -> list[dict]:
    """Realistic mixed traffic: every eval question once, its paraphrases, plus exact repeats.

    Each entry carries `source_id` (the eval question it is really asking) so that quality can
    be scored, and later so false cache hits can be detected."""
    rng = random.Random(seed)
    by_id = {it["id"]: it for it in items}
    traffic = [{"query": it["question"], "source_id": it["id"], "kind": "original"} for it in items]
    traffic += [{"query": p["paraphrase"], "source_id": p["source_id"], "kind": "paraphrase"} for p in paraphrases]
    traffic += [
        {"query": by_id[sid]["question"], "source_id": sid, "kind": "repeat"}
        for sid in rng.choices(sorted(by_id), k=repeats)
    ]
    rng.shuffle(traffic)
    return traffic


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    quality = [r["quality"] for r in ok if r["quality"] is not None]
    lat = [r["latency_s"] for r in rows]
    cost = sum(r["cost_usd"] or 0.0 for r in rows)

    def by(key: str) -> dict:
        groups: dict[str, list[float]] = {}
        for r in ok:
            if r["quality"] is not None and r.get(key):
                groups.setdefault(r[key], []).append(r["quality"])
        return {k: round(float(np.mean(v)), 3) for k, v in sorted(groups.items())}

    return {
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "total_cost_usd": round(cost, 6),
        "cost_per_request_usd": round(cost / len(rows), 6) if rows else None,
        "cache_hit_rate": round(sum(bool(r["cache_hit"]) for r in rows) / len(rows), 3) if rows else None,
        "latency_p50_s": round(percentile(lat, 50), 3),
        "latency_p95_s": round(percentile(lat, 95), 3),
        "quality_mean": round(float(np.mean(quality)), 3) if quality else None,
        "retrieval_hit_rate": round(float(np.mean([r["retrieval_hit"] for r in rows])), 3) if rows else None,
        "quality_by_lang": by("lang"),
        "quality_by_kind": by("kind"),
        "total_input_tokens": sum(r["input_tokens"] or 0 for r in rows),
        "total_output_tokens": sum(r["output_tokens"] or 0 for r in rows),
    }


def run_baseline(
    traffic: list[dict],
    items_by_id: dict[str, dict],
    index: RagIndex,
    llm,
    logger: RequestLogger,
    pricing: dict,
    model: str,
    tier: str,
    top_k: int,
    run_id: str,
) -> list[dict]:
    """Always-large-model baseline: retrieve, generate with one model, log everything."""
    logger.delete_run(run_id)
    for req in traffic:
        query, item = req["query"], items_by_id[req["source_id"]]
        t0 = time.perf_counter()
        retrieved = index.retrieve(query, k=top_k)
        retrieval_s = time.perf_counter() - t0
        res = generate(query, retrieved, llm, model, tier)
        failed = bool(res["error"])
        logger.log(
            run_id=run_id, system="baseline", ts=time.time(), query=query, lang=detect_lang(query),
            kind=req["kind"], source_id=req["source_id"], route=tier, model=model,
            cache_hit=0, escalated=0,
            input_tokens=res["input_tokens"], output_tokens=res["output_tokens"],
            cost_usd=compute_cost(pricing, tier, res["input_tokens"], res["output_tokens"]),
            # retrieval time measured now + the model call's own (replayed, if disk-cached) latency
            latency_s=retrieval_s + res["llm_latency_s"],
            quality=None if failed else key_recall(item["expected_answer"], res["answer"]),
            retrieval_hit=int(Path(item["doc"]).stem in res["sources"]),
            top_score=res["top_score"], answer=res["answer"], error=res["error"],
        )
    return logger.fetch(run_id)
