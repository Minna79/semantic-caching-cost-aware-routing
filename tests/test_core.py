import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app.embeddings import HashEmbedder
from app.evaluation import build_traffic, is_idk, key_recall, summarize
from app.logging_config import RequestLogger, compute_cost
from app.models import BudgetExceeded, CallBudget, DiskCache, MockLLM, OpenRouterLLM
from app.rag import RagIndex, build_prompt, chunk_text, compute_corpus_version, generate
from app.text import detect_lang, normalize, split_sentences
from app.vectorstore import NumpyVectorStore


class TextTests(unittest.TestCase):
    def test_normalize_english(self):
        self.assertEqual(normalize("How do I reset my Password?!"), "how do i reset my password")

    def test_normalize_persian(self):
        # Arabic ي/ك -> Persian ی/ک, Persian digits -> ASCII, ZWNJ -> space, punctuation dropped
        self.assertEqual(normalize("درياچه ۱۳۷۷؟"), "دریاچه 1377")
        self.assertEqual(normalize("می‌گویند"), "می گویند")

    def test_detect_lang(self):
        self.assertEqual(detect_lang("What is a qanat?"), "en")
        self.assertEqual(detect_lang("قنات چیست؟"), "fa")

    def test_split_sentences_handles_headings(self):
        out = split_sentences("Title\n\nFirst one. Second one?\n== History ==\nThird.")
        self.assertEqual(out, ["Title", "First one.", "Second one?", "History", "Third."])


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = NumpyVectorStore(3)
        self.store.add(np.eye(3), [{"n": "x"}, {"n": "y"}, {"n": "z"}])

    def test_search_orders_by_cosine(self):
        hits = self.store.search(np.array([0.9, 0.1, 0.0]), k=2)
        self.assertEqual([h.payload["n"] for h in hits], ["x", "y"])

    def test_filter_and_delete(self):
        hits = self.store.search(np.array([1.0, 0, 0]), k=3, where=lambda p: p["n"] != "x")
        self.assertNotIn("x", [h.payload["n"] for h in hits])
        self.assertEqual(self.store.delete_where(lambda p: p["n"] == "x"), 1)
        self.assertEqual(len(self.store), 2)

    def test_empty_store(self):
        self.assertEqual(NumpyVectorStore(3).search(np.zeros(3)), [])

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            self.store.save(Path(d) / "idx")
            loaded = NumpyVectorStore.load(Path(d) / "idx")
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded.search(np.array([0, 1.0, 0]), k=1)[0].payload["n"], "y")


class EmbedderTests(unittest.TestCase):
    def test_unit_norm_and_deterministic(self):
        e = HashEmbedder()
        a, b = e.embed(["tardigrade legs", "tardigrade legs"])
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=5)
        np.testing.assert_array_equal(a, b)

    def test_similar_text_scores_higher_than_unrelated(self):
        e = HashEmbedder()
        q, near, far = e.embed(["when was the transistor invented", "the transistor was invented when", "krakatoa eruption deaths"])
        self.assertGreater(float(q @ near), float(q @ far))


class RagTests(unittest.TestCase):
    DOC = "Title\n\n" + " ".join(f"Sentence number {i} about nothing in particular." for i in range(60))

    def test_chunks_are_bounded_titled_and_overlap(self):
        chunks = chunk_text(self.DOC, max_chars=300)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(c.startswith("Title: ") for c in chunks))
        self.assertTrue(all(len(c) < 300 + 100 for c in chunks))
        last_of_first = chunks[0].rstrip().rsplit(". ", 1)[-1]
        self.assertIn(last_of_first.rstrip("."), chunks[1])

    def test_corpus_version_changes_with_content(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.txt").write_text("one", encoding="utf-8")
            v1 = compute_corpus_version(Path(d))
            self.assertEqual(v1, compute_corpus_version(Path(d)))
            (Path(d) / "a.txt").write_text("two", encoding="utf-8")
            self.assertNotEqual(v1, compute_corpus_version(Path(d)))

    def test_index_retrieves_and_mock_generates_cited_answer(self):
        with tempfile.TemporaryDirectory() as d:
            docs = Path(d)
            (docs / "en_00.txt").write_text(
                "Tardigrade\n\nTardigrades have four pairs of legs. They live in moss.", encoding="utf-8")
            (docs / "en_01.txt").write_text(
                "Krakatoa\n\nKrakatoa erupted violently in 1883. Many people died.", encoding="utf-8")
            index = RagIndex.build(docs, HashEmbedder())
            retrieved = index.retrieve("How many pairs of legs do tardigrades have?", k=1)
            self.assertEqual(retrieved[0].doc, "en_00")
            res = generate("How many pairs of legs do tardigrades have?", retrieved, MockLLM(), "large-model", "large")
            self.assertIn("four pairs of legs", res["answer"])
            self.assertTrue(res["cited"])
            self.assertEqual(res["sources"], ["en_00"])

    def test_index_persists_and_reloads(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as idx:
            (Path(d) / "en_00.txt").write_text("T\n\nSome long enough sentence here.", encoding="utf-8")
            a = RagIndex.build_or_load(Path(d), Path(idx), HashEmbedder())
            b = RagIndex.build_or_load(Path(d), Path(idx), HashEmbedder())
            self.assertEqual(len(a.store), len(b.store))
            self.assertTrue(any(Path(idx).iterdir()))

    def test_prompt_contains_context_and_question(self):
        from app.rag import Retrieved
        p = build_prompt("Why?", [Retrieved(0.9, "en_00", "line one\nline two")])
        self.assertIn("[en_00] line one line two", p)
        self.assertTrue(p.rstrip().endswith("Answer:"))
        self.assertIn("Question: Why?", p)


def fake_response(status, body=None, headers=None):
    r = mock.Mock()
    r.status_code, r.headers = status, headers or {}
    r.json.return_value = body if body is not None else {}
    return r


OK_BODY = {"choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


class OpenRouterTests(unittest.TestCase):
    def make(self, max_calls=5, cache=None):
        return OpenRouterLLM("key", CallBudget(max_calls), cache, max_retries=2)

    @mock.patch("app.models.time.sleep")
    @mock.patch("app.models.requests.post")
    def test_success_and_token_usage(self, post, _sleep):
        post.return_value = fake_response(200, OK_BODY)
        r = self.make().complete("hi", "m")
        self.assertEqual((r.text, r.input_tokens, r.output_tokens, r.error), ("hello", 10, 2, None))

    @mock.patch("app.models.time.sleep")
    @mock.patch("app.models.requests.post")
    def test_402_is_not_retried(self, post, _sleep):
        post.return_value = fake_response(402, {"error": "no credits"})
        r = self.make().complete("hi", "m")
        self.assertIn("402", r.error)
        self.assertEqual(post.call_count, 1)

    @mock.patch("app.models.time.sleep")
    @mock.patch("app.models.requests.post")
    def test_429_is_retried_then_succeeds(self, post, _sleep):
        post.side_effect = [fake_response(429, {}, {"Retry-After": "1"}), fake_response(200, OK_BODY)]
        r = self.make().complete("hi", "m")
        self.assertEqual(r.text, "hello")
        self.assertEqual(post.call_count, 2)

    @mock.patch("app.models.time.sleep")
    @mock.patch("app.models.requests.post")
    def test_gives_up_after_retries(self, post, _sleep):
        post.return_value = fake_response(503, {})
        r = self.make().complete("hi", "m")
        self.assertIn("gave up", r.error)
        self.assertEqual(post.call_count, 3)

    @mock.patch("app.models.requests.post")
    def test_budget_stops_real_calls(self, post):
        post.return_value = fake_response(200, OK_BODY)
        llm = self.make(max_calls=2)
        llm.complete("a", "m"); llm.complete("b", "m")
        with self.assertRaises(BudgetExceeded):
            llm.complete("c", "m")
        self.assertEqual(post.call_count, 2)

    @mock.patch("app.models.requests.post")
    def test_disk_cache_means_no_second_call_and_no_budget(self, post):
        post.return_value = fake_response(200, OK_BODY)
        with tempfile.TemporaryDirectory() as d:
            llm = self.make(max_calls=1, cache=DiskCache(Path(d) / "c.sqlite"))
            first = llm.complete("same prompt", "m")
            second = llm.complete("same prompt", "m")  # would exceed the budget if it hit the API
        self.assertFalse(first.from_disk_cache)
        self.assertTrue(second.from_disk_cache)
        self.assertEqual(second.latency_s, first.latency_s)
        self.assertEqual(post.call_count, 1)

    @mock.patch("app.models.requests.post")
    def test_failures_are_not_cached(self, post):
        post.return_value = fake_response(402, {"error": "no credits"})
        with tempfile.TemporaryDirectory() as d:
            cache = DiskCache(Path(d) / "c.sqlite")
            self.make(cache=cache).complete("p", "m")
            self.assertIsNone(cache.get(DiskCache.key("m", "p", 300)))

    def test_requires_key(self):
        with self.assertRaises(ValueError):
            OpenRouterLLM(None, CallBudget(1))


class LoggingAndEvalTests(unittest.TestCase):
    PRICING = {"large": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}}

    def test_cost(self):
        self.assertAlmostEqual(compute_cost(self.PRICING, "large", 1_000_000, 100_000), 3.0 + 1.5)

    def test_logger_roundtrip_and_unknown_field(self):
        log = RequestLogger(":memory:")
        log.log(run_id="r", query="q", cost_usd=0.5, cache_hit=0)
        rows = log.fetch("r")
        self.assertEqual((len(rows), rows[0]["query"], rows[0]["cost_usd"]), (1, "q", 0.5))
        with self.assertRaises(ValueError):
            log.log(run_id="r", bogus=1)
        log.delete_run("r")
        self.assertEqual(log.fetch("r"), [])

    def test_key_recall(self):
        self.assertEqual(key_recall("It was found in 1901.", "Discovered in 1901 off Antikythera"), 0.5)
        self.assertEqual(key_recall("Callimachus.", "It was Callimachus [en_09]"), 1.0)
        self.assertEqual(key_recall("Callimachus.", "I don't know."), 0.0)
        self.assertEqual(key_recall("۷۰٫۷ درجه", "حدود 70 7 درجه"), 1.0)  # Persian digits normalise

    def test_idk(self):
        self.assertTrue(is_idk("I don't know."))
        self.assertTrue(is_idk(""))
        self.assertTrue(is_idk("نمی‌دانم"))
        self.assertFalse(is_idk("Callimachus"))

    def test_traffic_is_deterministic_and_mixed(self):
        items = [{"id": f"q{i}", "question": f"question {i}"} for i in range(10)]
        para = [{"source_id": "q1", "paraphrase": "paraphrased 1"}]
        t1, t2 = build_traffic(items, para, repeats=5, seed=1), build_traffic(items, para, repeats=5, seed=1)
        self.assertEqual(t1, t2)
        self.assertEqual({r["kind"] for r in t1}, {"original", "paraphrase", "repeat"})
        self.assertEqual(len(t1), 16)

    def test_summarize(self):
        rows = [
            {"error": None, "cost_usd": 1.0, "cache_hit": 0, "latency_s": 1.0, "quality": 1.0, "retrieval_hit": 1,
             "lang": "en", "kind": "original", "input_tokens": 10, "output_tokens": 1},
            {"error": None, "cost_usd": 0.0, "cache_hit": 1, "latency_s": 3.0, "quality": 0.5, "retrieval_hit": 0,
             "lang": "fa", "kind": "repeat", "input_tokens": 0, "output_tokens": 0},
            {"error": "HTTP 402", "cost_usd": 0.0, "cache_hit": 0, "latency_s": 2.0, "quality": None, "retrieval_hit": 0,
             "lang": "en", "kind": "original", "input_tokens": 0, "output_tokens": 0},
        ]
        s = summarize(rows)
        self.assertEqual((s["requests"], s["errors"], s["total_cost_usd"]), (3, 1, 1.0))
        self.assertEqual(s["quality_mean"], 0.75)
        self.assertEqual(s["latency_p50_s"], 2.0)
        self.assertEqual(s["quality_by_lang"], {"en": 1.0, "fa": 0.5})


if __name__ == "__main__":
    unittest.main()
