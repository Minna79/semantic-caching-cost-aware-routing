"""Guards the eval data: every answer must be traceable to a verbatim quote in its document."""
import collections
import unittest

from app.config import load_settings
from app.evaluation import load_json

S = load_settings()
ITEMS = load_json(S.eval_path)
PARA = load_json(S.eval_path.parent / "paraphrase_pairs.json")
NEAR = load_json(S.eval_path.parent / "near_miss_pairs.json")


class EvalData(unittest.TestCase):
    def test_sizes(self):
        self.assertGreaterEqual(len(ITEMS), 100)
        self.assertEqual(len(PARA), 50)
        self.assertEqual(len(NEAR), 50)

    def test_ids_and_questions_unique(self):
        self.assertEqual(len({i["id"] for i in ITEMS}), len(ITEMS))
        self.assertEqual(len({i["question"] for i in ITEMS}), len(ITEMS))

    def test_every_evidence_quote_is_in_its_document(self):
        for it in ITEMS:
            text = (S.docs_dir / it["doc"]).read_text(encoding="utf-8")
            self.assertTrue(it["evidence"], it["id"])
            for quote in it["evidence"]:
                self.assertIn(quote, text, f'{it["id"]}: {quote!r}')

    def test_both_languages_and_difficulties(self):
        counts = collections.Counter((i["lang"], i["difficulty"]) for i in ITEMS)
        for lang in ("en", "fa"):
            for diff in ("simple", "complex"):
                self.assertGreaterEqual(counts[(lang, diff)], 10, (lang, diff))

    def test_pairs_reference_real_questions_and_differ(self):
        by_id = {i["id"]: i for i in ITEMS}
        for p in PARA:
            self.assertEqual(by_id[p["source_id"]]["question"], p["question"])
            self.assertNotEqual(p["question"], p["paraphrase"])
        for n in NEAR:
            self.assertNotEqual(n["a"], n["b"])
        self.assertEqual(len({(n["a"], n["b"]) for n in NEAR}), len(NEAR))


if __name__ == "__main__":
    unittest.main()
