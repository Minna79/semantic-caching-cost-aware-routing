"""Request logging (SQLite) and cost accounting."""
from __future__ import annotations

import sqlite3
from pathlib import Path

COLUMNS = [
    ("run_id", "TEXT"), ("system", "TEXT"), ("ts", "REAL"), ("query", "TEXT"), ("lang", "TEXT"),
    ("kind", "TEXT"), ("source_id", "TEXT"), ("route", "TEXT"), ("model", "TEXT"),
    ("cache_hit", "INTEGER"), ("escalated", "INTEGER"), ("input_tokens", "INTEGER"),
    ("output_tokens", "INTEGER"), ("cost_usd", "REAL"), ("latency_s", "REAL"),
    ("quality", "REAL"), ("retrieval_hit", "INTEGER"), ("top_score", "REAL"), ("answer", "TEXT"), ("error", "TEXT"),
]
FIELD_NAMES = [c for c, _ in COLUMNS]


def compute_cost(pricing: dict, tier: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD from token counts and the per-tier price table (never from provider billing)."""
    p = pricing[tier]
    return input_tokens / 1e6 * p["input_per_mtok"] + output_tokens / 1e6 * p["output_per_mtok"]


class RequestLogger:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        cols = ", ".join(f"{name} {typ}" for name, typ in COLUMNS)
        self.db.execute(f"CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
        self.db.commit()

    def log(self, **fields) -> None:
        unknown = set(fields) - set(FIELD_NAMES)
        if unknown:
            raise ValueError(f"unknown log fields: {sorted(unknown)}")
        row = {name: fields.get(name) for name in FIELD_NAMES}
        self.db.execute(
            f"INSERT INTO requests ({', '.join(FIELD_NAMES)}) VALUES ({', '.join('?' * len(FIELD_NAMES))})",
            [row[n] for n in FIELD_NAMES],
        )
        self.db.commit()

    def fetch(self, run_id: str) -> list[dict]:
        cur = self.db.execute("SELECT * FROM requests WHERE run_id=? ORDER BY id", (run_id,))
        return [dict(r) for r in cur.fetchall()]

    def delete_run(self, run_id: str) -> None:
        self.db.execute("DELETE FROM requests WHERE run_id=?", (run_id,))
        self.db.commit()
