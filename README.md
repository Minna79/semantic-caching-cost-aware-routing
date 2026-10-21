# Semantic Caching and Cost-Aware LLM Routing

An AI engineering project that reduces LLM inference costs and response latency through semantic caching, intelligent model routing, and answer-quality evaluation in a Retrieval-Augmented Generation (RAG) system.

## Overview

Large Language Models (LLMs) can be expensive and slow when every request requires a new generation. Many applications also receive semantically similar questions that could reuse previously generated answers.

This project explores two optimization techniques:

- **Semantic Caching:** Reuse answers to sufficiently similar queries to avoid unnecessary LLM calls.
- **Cost-Aware Model Routing:** Direct simple queries to a smaller, less expensive model and complex queries to a more capable model, with escalation when answer quality is insufficient.

A benchmarking pipeline and interactive dashboard will be used to evaluate the trade-offs between cost, latency, and answer quality.

## Key Features

- Semantic similarity search for identifying reusable answers
- Configurable similarity threshold for cache-hit decisions
- Cache invalidation using corpus versions and time-to-live (TTL)
- Rule-based query classification and model routing
- Escalation from a small model to a larger model when necessary
- Request logging, token tracking, and cost estimation
- Evaluation against an always-large-model baseline
- Interactive dashboard for comparing system performance

## Architecture

```text
User Query
    |
    v
Query Normalization
    |
    v
Generate Embedding
    |
    v
Semantic Cache Lookup
    |
    +---- Cache Hit ----> Return Cached Answer
    |
    +---- Cache Miss ---> Classify Query
                              |
                    +---------+---------+
                    |                   |
                    v                   v
                Small Model         Large Model
                    |
                    v
              Quality Check
                    |
              +-----+-----+
              |           |
              v           v
           Accept      Escalate
                         to Large Model
                    |
                    v
             Store Cache Entry
                    |
                    v
             Log Request Metrics
                    |
                    v
             Evaluation Dashboard
```

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| FastAPI | REST API and request handling |
| numpy (Qdrant later) | Vector search; a Qdrant backend can be added behind the same interface |
| sentence-transformers | Query embedding generation |
| requests + OpenRouter | LLM calls (LiteLLM can replace this later) |
| SQLite | Request logs and performance metrics |
| Streamlit | Interactive evaluation dashboard |

## How It Works

### 1. Baseline Measurement

Establish a baseline using a large language model for every query without semantic caching. Record request latency, token usage, estimated cost, and answer quality.

### 2. Semantic Caching

Normalize each query, generate its embedding, and search for similar cached queries. If the similarity score exceeds a configurable threshold and the entry is valid, return the cached response.

Otherwise, process the request normally and store the result for future reuse.

Cache entries will include metadata such as the original query, embedding, answer, corpus version, and timestamp.

### 3. Cost-Aware Routing

Classify incoming queries as simple or complex.

- **Simple queries:** Route to a smaller, less expensive model.
- **Complex queries:** Route to a larger, more capable model.
- **Low-confidence answers:** Escalate to the larger model when predefined quality checks indicate that the initial response may be insufficient.

The initial implementation will use a rule-based router, with the option to evaluate an LLM-based classifier later.

### 4. Evaluation and Benchmarking

Compare the optimized system against the baseline using the same evaluation dataset. Include paraphrased and repeated queries to simulate realistic cache reuse.

### 5. Performance Dashboard

Use Streamlit to visualize request volume, cache hit rate, model distribution, estimated cost, latency percentiles, escalation rate, and answer-quality metrics.

## Evaluation Metrics

| Metric | Description |
|---|---|
| Cost per request | Average estimated LLM cost per request |
| Cache hit rate | Percentage of requests served from the cache |
| p50 latency | Median end-to-end response time |
| p95 latency | Response time below which 95% of requests complete |
| Answer quality | Correctness and relevance measured on a fixed evaluation set |
| Escalation rate | Percentage of requests escalated to the larger model |
| False cache-hit rate | Percentage of cache hits that return an inappropriate answer |

## Experimental Design

The evaluation will compare two configurations:

1. **Baseline:** Large-model generation without semantic caching.
2. **Optimized system:** Semantic caching, cost-aware routing, and quality-based escalation.

Both configurations will use the same documents, evaluation questions, and quality criteria.

The cache similarity threshold will be tuned using paraphrase pairs and near-miss pairs. Cache invalidation, corpus-version checks, and TTL will help prevent stale responses from being reused.

Cost estimates will use a configurable model-pricing table and recorded token usage. Embedding and other system overhead will be considered separately when reporting total operating cost.

## Project Status

| Step | Status |
|---|---|
| Documents (15 English + 15 Persian Wikipedia articles) | Done |
| Eval set: 115 questions, 50 paraphrase pairs, 50 near-miss pairs | Done |
| RAG pipeline, request logging, cost table, baseline runner | Done (tested with a mock LLM) |
| Real baseline numbers | Needs a run with real models, see below |
| Semantic cache, threshold tuning, router, escalation, dashboard, comparison | Not started |

## Project Structure

```text
semantic-caching-cost-aware-routing/
├── app/
│   ├── config.py            # Settings from environment variables
│   ├── text.py              # Normalisation, language detection, sentence splitting (EN + FA)
│   ├── embeddings.py        # Embedders: real multilingual model + offline stand-in
│   ├── vectorstore.py       # Exact cosine search with numpy (no server needed)
│   ├── rag.py               # Chunking, index, retrieval, prompt, generation
│   ├── models.py            # Mock LLM and OpenRouter client (disk cache + call budget)
│   ├── logging_config.py    # SQLite request log and cost accounting
│   └── evaluation.py        # Quality scoring, traffic builder, summaries, baseline runner
├── scripts/
│   ├── run_baseline.py      # Run the always-large-model baseline
│   ├── check_key.py         # Check the OpenRouter key and remaining credit
│   └── check_models.py      # Smoke-test the two models
├── config/
│   └── model_pricing.json   # Price table (placeholder values, edit before trusting costs)
├── data/
│   ├── eval_questions.json  # 115 questions (EN + FA) with answers and verbatim evidence quotes
│   ├── paraphrase_pairs.json# 50 pairs that SHOULD hit the cache
│   └── near_miss_pairs.json # 50 pairs that must NOT hit the cache
├── docs/                    # Source documents
├── tests/                   # 34 tests (python -m unittest discover -s tests -t .)
├── fetch_docs.py            # Downloads the Wikipedia documents
├── generate_eval.py         # Original LLM-based question generator (superseded by data/)
└── README.md
```

Planned but not built yet: `app/cache.py`, `app/router.py`, `app/main.py` (FastAPI), `dashboard/app.py` (Streamlit).

## Evaluation Data

Every question in `data/eval_questions.json` was written from the document text and carries one or more
`evidence` quotes that appear verbatim in its source document. `tests/test_data.py` fails if any quote
stops matching, so answers stay traceable to the documents.

- 115 questions: 63 English, 52 Persian; 82 `simple` (single fact lookup) and 33 `complex` (combine facts or explain).
- Paraphrase pairs keep the same answer in different words, and double as the repeat traffic in the benchmark.
- Near-miss pairs are worded almost identically but have different answers (for example Pasargadae vs Chogha Zanbil
  UNESCO dates). One pair is flagged `"hard": true` because it only reverses word order, which most embedders
  cannot separate; report threshold results with and without it.

## Getting Started

### Prerequisites

- Python 3.10 or later
- An OpenRouter API key only if you want real model calls (the mock needs nothing)

### Installation

```bash
git clone https://github.com/Minna79/semantic-caching-cost-aware-routing.git
cd semantic-caching-cost-aware-routing
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit
```

### Running the tests

```bash
python -m unittest discover -s tests -t .
```

### Running the baseline

Plumbing check, completely free (mock LLM and offline embedder, numbers are not meaningful):

```bash
python scripts/run_baseline.py
```

Real run (uses the multilingual embedding model and OpenRouter):

```bash
python scripts/run_baseline.py --llm openrouter --embedding sentence-transformers --limit 20   # small trial first
python scripts/run_baseline.py --llm openrouter --embedding sentence-transformers
```

Results go to `data/runtime/` (gitignored): the request log (`requests.db`) and `reports/baseline.json`.

### Working on a tight budget

- Every successful model response is cached on disk (`data/runtime/llm_cache.sqlite`), so re-running a prompt never costs twice.
- `MAX_LLM_CALLS` (default 300) stops a run after that many real API calls.
- Cost in reports is computed from token counts and `config/model_pricing.json`, not from provider billing, so a free model can
  stand in for a paid one: set `LARGE_MODEL` to a free model id and keep the paid model's list price in the price table.
  State this plainly in any results you publish.
