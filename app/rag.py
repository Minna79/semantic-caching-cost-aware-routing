"""Retrieval-augmented answering: chunk the documents, index them, retrieve, generate."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .text import split_sentences
from .vectorstore import NumpyVectorStore

CITATION = re.compile(r"\[[a-z]{2}_\d+\]")

PROMPT_TEMPLATE = (
    "You answer questions using ONLY the context below. Reply in the same language as the question, "
    "in one or two sentences, and cite the source id in square brackets, for example [en_03]. "
    "If the context does not contain the answer, reply exactly: I don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


@dataclass
class Retrieved:
    score: float
    doc: str
    text: str


def load_docs(docs_dir: Path) -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path(docs_dir).glob("*.txt"))}


def compute_corpus_version(docs_dir: Path) -> str:
    """Hash of all document names and bytes. Changes whenever the corpus changes, so anything
    keyed on it (the index now, cache entries later) is invalidated automatically."""
    h = hashlib.sha256()
    for p in sorted(Path(docs_dir).glob("*.txt")):
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def chunk_text(text: str, max_chars: int = 900) -> list[str]:
    """Sentence-aware chunks of about max_chars, each prefixed with the document title and
    overlapping the previous chunk by one sentence."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    title, body = sentences[0], sentences[1:]
    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for sent in body:
        if current and size + len(sent) > max_chars:
            chunks.append(current)
            current, size = [current[-1]], len(current[-1])  # one-sentence overlap
        current.append(sent)
        size += len(sent) + 1
    if current:
        chunks.append(current)
    return [f"{title}: {' '.join(c)}" for c in chunks]


class RagIndex:
    def __init__(self, embedder, store: NumpyVectorStore, corpus_version: str):
        self.embedder, self.store, self.corpus_version = embedder, store, corpus_version

    @classmethod
    def build(cls, docs_dir: Path, embedder) -> "RagIndex":
        payloads = [
            {"doc": doc, "text": chunk} for doc, text in load_docs(docs_dir).items() for chunk in chunk_text(text)
        ]
        store = NumpyVectorStore(embedder.dim)
        for i in range(0, len(payloads), 64):
            batch = payloads[i : i + 64]
            store.add(embedder.embed([p["text"] for p in batch]), batch)
        return cls(embedder, store, compute_corpus_version(docs_dir))

    @classmethod
    def build_or_load(cls, docs_dir: Path, index_dir: Path, embedder) -> "RagIndex":
        version = compute_corpus_version(docs_dir)
        prefix = Path(index_dir) / f"{embedder.name}_{version}"
        if prefix.with_suffix(".npy").exists() and prefix.with_suffix(".json").exists():
            return cls(embedder, NumpyVectorStore.load(prefix), version)
        index = cls.build(docs_dir, embedder)
        index.store.save(prefix)
        return index

    def retrieve(self, query: str, k: int = 4) -> list[Retrieved]:
        hits = self.store.search(self.embedder.embed([query])[0], k=k)
        return [Retrieved(h.score, h.payload["doc"], h.payload["text"]) for h in hits]


def build_prompt(question: str, retrieved: list[Retrieved]) -> str:
    context = "\n".join(f"[{r.doc}] {' '.join(r.text.split())}" for r in retrieved)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate(question: str, retrieved: list[Retrieved], llm, model: str, tier: str, max_tokens: int = 300) -> dict:
    resp = llm.complete(build_prompt(question, retrieved), model, max_tokens)
    scores = [round(r.score, 4) for r in retrieved]
    return {
        "answer": resp.text,
        "sources": list(dict.fromkeys(r.doc for r in retrieved)),
        "retrieval_scores": scores,
        "top_score": scores[0] if scores else 0.0,
        "model": model,
        "tier": tier,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "llm_latency_s": resp.latency_s,
        "from_disk_cache": resp.from_disk_cache,
        "cited": bool(CITATION.search(resp.text)),
        "error": resp.error,
    }
