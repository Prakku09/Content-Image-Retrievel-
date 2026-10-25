import unittest

import numpy as np

from statistical_evaluation import holm_adjust, paired_bootstrap_ci, paired_permutation_test


class StatisticalEvaluationTests(unittest.TestCase):
    def test_holm_adjustment_preserves_order_and_controls_step_down(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03, 0.002])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.008])

    def test_bootstrap_is_reproducible_and_contains_positive_shift(self):
        categories = np.repeat(['a', 'b'], 10)
        differences = np.concatenate([
            np.linspace(0.01, 0.10, 10),
            np.linspace(0.02, 0.11, 10),
        ])
        first = paired_bootstrap_ci(differences, categories, 1000, seed=11)
        second = paired_bootstrap_ci(differences, categories, 1000, seed=11)
        self.assertEqual(first, second)
        self.assertLess(0, first[0])

    def test_paired_permutation_is_reproducible_and_two_sided(self):
        differences = np.full(100, 0.05)
        first = paired_permutation_test(differences, 10000, seed=19)
        second = paired_permutation_test(differences, 10000, seed=19)
        self.assertEqual(first, second)
        self.assertLess(first, 0.01)

    def test_bootstrap_rejects_unbalanced_category_strata(self):
        differences = np.array([0.1, 0.2, 0.3])
        categories = np.array(['a', 'a', 'b'])
        with self.assertRaises(ValueError):
            paired_bootstrap_ci(differences, categories, 100, seed=1)


if __name__ == '__main__':
    unittest.main()
