import unittest

from src.evaluate import ndcg_at_k, recall_at_k
from src.rerank import rerank


class RerankTest(unittest.TestCase):
    def test_rerank_changes_order_but_not_candidates(self):
        candidates = {"a": 2.0, "b": 1.0}
        pairs_seen = []

        def predict(pairs):
            pairs_seen.extend(pairs)
            return [0.1, 0.9]

        out = rerank("query", candidates, {"a": "text a", "b": "text b"}, predict)
        self.assertEqual(pairs_seen, [("query", "text a"), ("query", "text b")])
        self.assertEqual(set(out), set(candidates))
        self.assertEqual(recall_at_k({"q": out}, {"q": {"b": 1}}, 1), 1.0)
        self.assertGreater(ndcg_at_k({"q": out}, {"q": {"b": 1}}, 1),
                           ndcg_at_k({"q": candidates}, {"q": {"b": 1}}, 1))

    def test_empty_and_invalid_predictions(self):
        self.assertEqual(rerank("q", {}, {}, lambda _: self.fail("unexpected call")), {})
        with self.assertRaises(ValueError):
            rerank("q", {"a": 1.0}, {"a": "x"}, lambda _: [])
        with self.assertRaises(ValueError):
            rerank("q", {"a": 1.0}, {"a": "x"}, lambda _: [float("nan")])


if __name__ == "__main__":
    unittest.main()
