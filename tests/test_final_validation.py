import unittest

import numpy as np

from output.statistical_evaluation_audit_v2.validate_saved_run import (
    centered_stratified_bootstrap_mean_test,
    paired_mean_bootstrap_ci,
)


class FinalValidationTests(unittest.TestCase):
    def test_centered_bootstrap_mean_test_is_reproducible(self):
        differences = np.array([0.03, 0.04, 0.02, 0.06, 0.05, 0.04])
        categories = np.array(["a", "a", "a", "b", "b", "b"])
        first = centered_stratified_bootstrap_mean_test(
            differences, categories, replicates=2000, seed=91
        )
        second = centered_stratified_bootstrap_mean_test(
            differences, categories, replicates=2000, seed=91
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first["null_hypothesis"],
            "equally category-weighted mean paired AP difference is zero",
        )

    def test_centered_bootstrap_rejects_unequal_stratum_sizes(self):
        with self.assertRaises(ValueError):
            centered_stratified_bootstrap_mean_test(
                [0.1, 0.2, 0.3], ["a", "a", "b"], replicates=100, seed=4
            )

    def test_paired_bootstrap_ci_is_reproducible(self):
        differences = np.array([0.1, 0.2, -0.1, 0.0])
        categories = np.array(["a", "a", "b", "b"])
        first = paired_mean_bootstrap_ci(
            differences, categories, replicates=500, seed=12
        )
        second = paired_mean_bootstrap_ci(
            differences, categories, replicates=500, seed=12
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
