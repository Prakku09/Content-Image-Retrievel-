import unittest

import numpy as np

from retrieval.evaluation import evaluate_all, evaluate_retrieval


class EvaluationMetricTests(unittest.TestCase):
    def test_average_precision_and_at_ten_metrics_use_full_ranking(self):
        ranking = np.arange(20)
        labels = ["cat"] * 2 + ["dog"] * 8 + ["cat"] * 10

        metrics = evaluate_retrieval(ranking, labels, "cat", k=10)

        expected_ap = (
            2.0
            + sum((index + 3.0) / (index + 11.0) for index in range(10))
        ) / 12.0
        self.assertAlmostEqual(metrics["ap"], expected_ap)
        self.assertEqual(metrics["precision_k"], 0.2)
        self.assertAlmostEqual(metrics["recall_k"], 2.0 / 12.0)
        self.assertAlmostEqual(
            metrics["f1_k"],
            2 * 0.2 * (2.0 / 12.0) / (0.2 + (2.0 / 12.0)),
        )

    def test_single_query_metric_rejects_non_permutation_rankings(self):
        labels = ["cat", "dog", "cat"]
        invalid_rankings = (
            np.asarray([0, 0, 2]),
            np.asarray([0, 1, 3]),
            np.asarray([[0], [1], [2]]),
            np.asarray([0.0, 1.0, 2.0]),
        )

        for ranking in invalid_rankings:
            with self.subTest(ranking=ranking):
                with self.assertRaises(ValueError):
                    evaluate_retrieval(ranking, labels, "cat", k=2)

    def test_aggregate_metrics_reject_duplicate_or_missing_gallery_indices(self):
        with self.assertRaisesRegex(ValueError, "permutation"):
            evaluate_all(
                [np.asarray([0, 0, 2])],
                ["cat", "dog", "cat"],
                ["cat"],
                [0.0],
                k=2,
            )


if __name__ == "__main__":
    unittest.main()
