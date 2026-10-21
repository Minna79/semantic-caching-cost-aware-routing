"""Central settings, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


@dataclass(frozen=True)
class Settings:
    root: Path
    docs_dir: Path
    runtime_dir: Path  # gitignored: indexes, request logs, LLM response cache
    eval_path: Path
    pricing_path: Path
    llm_backend: str  # "mock" | "openrouter"
    embedding_backend: str  # "hash" (offline stand-in) | "sentence-transformers"
    embedding_model: str
    small_model: str
    large_model: str
    top_k: int
    max_llm_calls: int  # hard cap on real API calls per process
    openrouter_api_key: str | None

    @property
    def index_dir(self) -> Path:
        return self.runtime_dir / "index"

    @property
    def db_path(self) -> Path:
        return self.runtime_dir / "requests.db"

    @property
    def llm_cache_path(self) -> Path:
        return self.runtime_dir / "llm_cache.sqlite"

    @property
    def reports_dir(self) -> Path:
        return self.runtime_dir / "reports"

    def model_for_tier(self, tier: str) -> str:
        return {"small": self.small_model, "large": self.large_model}[tier]


def load_settings(**overrides) -> Settings:
    env = os.environ.get
    runtime = Path(env("RUNTIME_DIR", ROOT / "data" / "runtime"))
    values = dict(
        root=ROOT,
        docs_dir=Path(env("DOCS_DIR", ROOT / "docs")),
        runtime_dir=runtime,
        eval_path=Path(env("EVAL_PATH", ROOT / "data" / "eval_questions.json")),
        pricing_path=Path(env("PRICING_PATH", ROOT / "config" / "model_pricing.json")),
        llm_backend=env("LLM_BACKEND", "mock"),
        embedding_backend=env("EMBEDDING_BACKEND", "hash"),
        embedding_model=env(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        small_model=env("SMALL_MODEL", "anthropic/claude-haiku-5.5"),
        large_model=env("LARGE_MODEL", "anthropic/claude-sonnet-5.5"),
        top_k=int(env("TOP_K", "4")),
        max_llm_calls=int(env("MAX_LLM_CALLS", "300")),
        openrouter_api_key=env("OPENROUTER_API_KEY"),
    )
    values.update(overrides)
    return Settings(**values)


def load_pricing(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["tiers"]
