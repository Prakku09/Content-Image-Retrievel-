import unittest

import numpy as np

from retrieval.retrieval import (
    validate_fusion_weights,
    weighted_euclidean_distances,
)


class WeightedRetrievalTests(unittest.TestCase):
    def test_mean_weight_control_uses_direct_m5_distance(self):
        query_color = np.array([0.0, 0.0])
        gallery_color = np.array([[1.0, 0.0], [0.0, 2.0]])
        query_texture = np.array([0.0])
        gallery_texture = np.array([[0.0], [1.0]])
        query_weights = np.array([[0.8, 0.2], [0.4, 0.6]])
        mean_weights = validate_fusion_weights(np.mean(query_weights, axis=0))

        distances = weighted_euclidean_distances(
            query_color,
            gallery_color,
            query_texture,
            gallery_texture,
            mean_weights,
        )

        expected = np.sqrt(
            mean_weights[0] * np.array([1.0, 4.0])
            + mean_weights[1] * np.array([0.0, 1.0])
        )
        np.testing.assert_allclose(distances, expected)
        self.assertAlmostEqual(float(np.sum(mean_weights)), 1.0, places=12)

    def test_weight_validation_accepts_float_tolerance(self):
        weights = validate_fusion_weights([0.3, 0.7000005], atol=1e-6)
        self.assertEqual(weights.shape, (2,))

    def test_weight_validation_preserves_model_output_precision(self):
        weights = np.array([0.6, 0.4], dtype=np.float32)
        validated = validate_fusion_weights(weights)
        self.assertEqual(validated.dtype, np.float32)

    def test_weight_validation_rejects_non_normalized_weights(self):
        with self.assertRaises(ValueError):
            validate_fusion_weights([0.4, 0.5])

    def test_weight_validation_rejects_negative_or_non_finite_weights(self):
        for weights in ([-0.1, 1.1], [np.nan, np.nan]):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    validate_fusion_weights(weights)

    def test_weighted_distance_rejects_mismatched_feature_dimensions(self):
        with self.assertRaises(ValueError):
            weighted_euclidean_distances(
                np.array([0.0]),
                np.array([[0.0, 1.0]]),
                np.array([0.0]),
                np.array([[0.0]]),
                np.array([0.5, 0.5]),
            )


if __name__ == '__main__':
    unittest.main()
