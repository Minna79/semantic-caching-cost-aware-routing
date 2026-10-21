"""Step 1 of the project: instrument the always-large-model baseline.

Every request is logged (tokens, cost from the price table, latency, quality) to SQLite and a
JSON report is written. Traffic = every eval question + its paraphrases + exact repeats, so the
same sequence can later be replayed through the optimised system.

Examples:
  python scripts/run_baseline.py                       # free offline mock, for plumbing only
  python scripts/run_baseline.py --llm openrouter --embedding sentence-transformers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_pricing, load_settings
from app.embeddings import get_embedder
from app.evaluation import build_traffic, load_json, run_baseline, summarize
from app.logging_config import RequestLogger
from app.models import BudgetExceeded, get_llm
from app.rag import RagIndex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["mock", "openrouter"])
    ap.add_argument("--embedding", choices=["hash", "sentence-transformers"])
    ap.add_argument("--model", help="override the large model id")
    ap.add_argument("--limit", type=int, help="only run the first N requests (cheap smoke test)")
    ap.add_argument("--repeats", type=int, default=40, help="number of exact-repeat requests in the traffic")
    ap.add_argument("--run-id", default="baseline")
    args = ap.parse_args()

    overrides = {}
    if args.llm:
        overrides["llm_backend"] = args.llm
    if args.embedding:
        overrides["embedding_backend"] = args.embedding
    if args.model:
        overrides["large_model"] = args.model
    s = load_settings(**overrides)

    items = load_json(s.eval_path)
    paraphrases = load_json(s.eval_path.parent / "paraphrase_pairs.json")
    traffic = build_traffic(items, paraphrases, repeats=args.repeats)
    if args.limit:
        traffic = traffic[: args.limit]

    embedder = get_embedder(s.embedding_backend, s.embedding_model)
    index = RagIndex.build_or_load(s.docs_dir, s.index_dir, embedder)
    print(f"index: {len(index.store)} chunks, corpus_version={index.corpus_version}, embedder={embedder.name}")

    logger = RequestLogger(s.db_path)
    try:
        rows = run_baseline(
            traffic, {it["id"]: it for it in items}, index, get_llm(s), logger,
            load_pricing(s.pricing_path), s.large_model, "large", s.top_k, args.run_id,
        )
    except BudgetExceeded as e:
        print(f"STOPPED: {e}")
        rows = logger.fetch(args.run_id)

    report = {
        "run_id": args.run_id,
        "llm_backend": s.llm_backend,
        "embedding_backend": s.embedding_backend,
        "model": s.large_model,
        "corpus_version": index.corpus_version,
        "summary": summarize(rows),
    }
    if s.llm_backend == "mock" or s.embedding_backend == "hash":
        report["warning"] = "mock LLM and/or offline hash embedder: numbers show the pipeline works, not real quality or cost"
        print("WARNING:", report["warning"])
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    out = s.reports_dir / f"{args.run_id}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
