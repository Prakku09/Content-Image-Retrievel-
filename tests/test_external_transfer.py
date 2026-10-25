import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config
from external_dataset import read_oxford_pet_splits
from external_transfer import (
    DEFAULT_RUN_ROOT,
    FEATURE_NAMES,
    _stratified_bootstrap_ci,
    _validate_duplicate_review,
    load_frozen_model,
    normalize_with_corel_bounds,
)
from model.mlp_model import AdaptiveWeightMLP
from retrieval.evaluation import evaluate_retrieval


class ExternalTransferTests(unittest.TestCase):
    def test_oxford_parser_uses_official_identity_and_breed_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            annotation_root = root / "annotations"
            image_root.mkdir()
            annotation_root.mkdir()
            trainval = []
            test = []
            for class_id in range(1, 38):
                if class_id <= 12:
                    species_id = 1
                    breed_id = class_id
                else:
                    species_id = 2
                    breed_id = class_id - 12
                breed = f"breed{class_id}"
                for split_rows, suffix in ((trainval, "1"), (test, "2")):
                    stem = f"{breed}_{suffix}"
                    Image.new("RGB", (8, 8), (class_id, 0, 0)).save(
                        image_root / f"{stem}.jpg"
                    )
                    split_rows.append(
                        f"{stem} {class_id} {species_id} {breed_id}"
                    )
            (annotation_root / "trainval.txt").write_text(
                "\n".join(trainval), encoding="utf-8"
            )
            (annotation_root / "test.txt").write_text(
                "\n".join(test), encoding="utf-8"
            )

            records = read_oxford_pet_splits(root)
            records_again = read_oxford_pet_splits(root)

            self.assertEqual(len(records), 74)
            self.assertEqual(len({record.label for record in records}), 37)
            self.assertEqual(
                {record.split for record in records}, {"gallery", "query"}
            )
            self.assertEqual(
                len({record.identity for record in records}), len(records)
            )
            self.assertEqual(
                [(record.identity, record.split) for record in records],
                [(record.identity, record.split) for record in records_again],
            )

    def test_feature_order_and_corel_scaler_are_preserved_without_clipping(self):
        self.assertEqual(len(FEATURE_NAMES), config.TOTAL_FEATURE_DIM)
        self.assertEqual(FEATURE_NAMES[:3], ("h_mean", "h_std", "h_skew"))
        self.assertEqual(FEATURE_NAMES[9:], (
            "glcm_contrast", "glcm_correlation", "glcm_energy",
            "glcm_homogeneity",
        ))
        parameters = {
            "color_min": [0.0] * 9,
            "color_max": [1.0] * 9,
            "texture_min": [0.0] * 4,
            "texture_max": [1.0] * 4,
        }
        color = np.full((1, 9), 2.0)
        texture = np.full((1, 4), -1.0)

        normalized_color, normalized_texture = normalize_with_corel_bounds(
            color, texture, parameters
        )

        np.testing.assert_array_equal(normalized_color, color)
        np.testing.assert_array_equal(normalized_texture, texture)

    def test_frozen_checkpoint_loads_with_strict_architecture(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "checkpoint.pt"
            model = AdaptiveWeightMLP(input_dim=config.TOTAL_FEATURE_DIM)
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

            loaded_model, checkpoint = load_frozen_model(checkpoint_path)

            self.assertFalse(loaded_model.training)
            self.assertIn("model_state_dict", checkpoint)
            with self.assertRaises(ValueError):
                load_frozen_model(checkpoint_path, expected_input_dim=12)

    def test_external_outputs_are_separate_from_corel_outputs(self):
        self.assertNotEqual(DEFAULT_RUN_ROOT.resolve(), config.PART2_OUTPUT_DIR.resolve())
        self.assertIn("external_transfer_oxford_pet", str(DEFAULT_RUN_ROOT))

    def test_existing_metric_function_calculates_full_gallery_ap(self):
        result = evaluate_retrieval(
            np.asarray([0, 1, 2, 3]),
            ["cat", "dog", "cat", "dog"],
            "cat",
            k=2,
        )

        self.assertAlmostEqual(result["ap"], (1.0 + 2.0 / 3.0) / 2.0)
        self.assertAlmostEqual(result["precision_k"], 0.5)
        self.assertAlmostEqual(result["recall_k"], 0.5)
        self.assertAlmostEqual(result["f1_k"], 0.5)

    def test_category_stratified_bootstrap_accepts_unequal_query_counts(self):
        differences = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
        categories = np.asarray(["a", "a", "b", "b", "b"])

        first = _stratified_bootstrap_ci(
            differences, categories, replicates=500, seed=9
        )
        second = _stratified_bootstrap_ci(
            differences, categories, replicates=500, seed=9
        )

        self.assertEqual(first, second)
        self.assertLess(first[0], float(np.mean(differences)))
        self.assertGreater(first[1], float(np.mean(differences)))

    def test_duplicate_review_requires_exact_candidate_coverage(self):
        candidate = {
            "gallery_path": "images/a.jpg",
            "query_path": "images/b.jpg",
            "hamming_distance": 2,
        }
        report = {"near_duplicate_candidates_cross_split": [candidate]}
        review = {
            "dataset_fingerprint_sha256": "fingerprint",
            "pairs": [candidate],
            "excluded_gallery_paths": ["images/a.jpg"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            review_path = Path(temporary) / "review.json"
            review_path.write_text(json.dumps(review), encoding="utf-8")

            loaded, excluded = _validate_duplicate_review(
                review_path, report, "fingerprint"
            )
            self.assertEqual(loaded, review)
            self.assertEqual(excluded, {"images/a.jpg"})

            with self.assertRaisesRegex(ValueError, "different dataset"):
                _validate_duplicate_review(review_path, report, "other")


if __name__ == "__main__":
    unittest.main()
