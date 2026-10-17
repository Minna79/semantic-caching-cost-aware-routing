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
| Qdrant | Vector search and semantic cache |
| sentence-transformers | Query embedding generation |
| LiteLLM | Unified interface for multiple LLM providers |
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

## Project Structure

```text
semantic-caching-cost-aware-routing/
├── app/
│   ├── main.py              # FastAPI application
│   ├── cache.py             # Semantic cache operations
│   ├── router.py            # Query classification and routing
│   ├── models.py            # LLM integration
│   ├── rag.py               # Retrieval and generation pipeline
│   ├── evaluation.py        # Benchmarking and quality evaluation
│   └── logging_config.py    # Request and metrics logging
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── data/
│   └── eval_questions.json  # Evaluation dataset
├── tests/
│   ├── test_cache.py
│   ├── test_router.py
│   └── test_api.py
├── config/
│   └── model_pricing.json   # Configurable model prices
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

*The structure above is the planned organization and may change during implementation.*

## Getting Started

### Prerequisites

- Python 3.10 or later
- API credentials for the selected LLM provider or providers
- Qdrant running locally or through an accessible instance

### Installation

Clone the repository:

```bash
git clone https://github.com/Minna79/semantic-caching-cost-aware-routing.git
cd semantic-caching-cost-aware-routing
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables using `.env.example` as a template. Add the required model-provider credentials and configuration values to your local `.env` file. Never commit API keys or secrets.

### Running the Application

The exact commands will be added once the application entry points and configuration are implemented.

The planned services are:

- FastAPI backend for query processing
- Qdrant for semantic cache storage
- SQLite for request metrics
- Streamlit for the performance dashboard
