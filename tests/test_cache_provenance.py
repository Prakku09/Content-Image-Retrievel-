import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from main import (
    _feature_source_fingerprints,
    _guard_legacy_output_overwrites,
)
import main
from model.mlp_trainer import _model_source_fingerprint


class CacheProvenanceTests(unittest.TestCase):
    def test_feature_cache_fingerprint_includes_current_extraction_sources(self):
        fingerprints = _feature_source_fingerprints()
        self.assertEqual(
            set(fingerprints),
            {"feature_extractor.py", "normalization.py"},
        )
        self.assertTrue(all(len(value) == 64 for value in fingerprints.values()))
        for path in (
            Path(__file__).resolve().parents[1]
            / "features"
            / "feature_extractor.py",
            Path(__file__).resolve().parents[1]
            / "features"
            / "normalization.py",
        ):
            self.assertTrue(path.is_file())

    def test_checkpoint_provenance_fingerprints_model_and_training_code(self):
        fingerprint = _model_source_fingerprint()
        self.assertEqual(len(fingerprint), 64)

    def test_legacy_pipeline_refuses_to_overwrite_existing_feature_cache(self):
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "features.pkl"
            cache_path.touch()
            with patch.object(main.config, "FEATURES_PKL_PATH", cache_path):
                with self.assertRaisesRegex(FileExistsError, "--allow-overwrite-results"):
                    _guard_legacy_output_overwrites("part1")
                _guard_legacy_output_overwrites("part1", allow_overwrite=True)


if __name__ == "__main__":
    unittest.main()
