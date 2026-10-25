import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from external_dataset import (
    dataset_fingerprint,
    detect_duplicates,
    load_metadata,
)


class ExternalDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.metadata = self.root / "metadata.csv"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_metadata(self, rows):
        with self.metadata.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(
                output, fieldnames=["image_path", "label", "split", "identity"]
            )
            writer.writeheader()
            writer.writerows(rows)

    def _image(self, relative_path, color):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 12), color).save(path)
        return path

    def test_load_metadata_validates_labels_and_split_identity(self):
        self._image("a.jpg", (255, 0, 0))
        self._image("b.jpg", (0, 255, 0))
        self._write_metadata([
            {"image_path": "a.jpg", "label": "cat", "split": "gallery", "identity": "a"},
            {"image_path": "b.jpg", "label": "cat", "split": "query", "identity": "b"},
        ])

        records = load_metadata(self.root, self.metadata)

        self.assertEqual([record.split for record in records], ["gallery", "query"])
        self.assertEqual({record.label for record in records}, {"cat"})

    def test_load_metadata_rejects_identity_leakage_and_escaping_paths(self):
        self._image("a.jpg", (255, 0, 0))
        self._image("b.jpg", (0, 255, 0))
        self._write_metadata([
            {"image_path": "a.jpg", "label": "cat", "split": "gallery", "identity": "same"},
            {"image_path": "b.jpg", "label": "cat", "split": "query", "identity": "same"},
        ])
        with self.assertRaisesRegex(ValueError, "both gallery and query"):
            load_metadata(self.root, self.metadata)

        self._write_metadata([
            {"image_path": "..\\outside.jpg", "label": "cat", "split": "gallery", "identity": "a"},
            {"image_path": "b.jpg", "label": "cat", "split": "query", "identity": "b"},
        ])
        with self.assertRaisesRegex(ValueError, "escapes dataset root"):
            load_metadata(self.root, self.metadata)

    def test_load_metadata_rejects_conflicting_labels_for_one_identity(self):
        self._image("a.jpg", (255, 0, 0))
        self._image("b.jpg", (0, 255, 0))
        self._image("c.jpg", (0, 0, 255))
        self._write_metadata([
            {"image_path": "a.jpg", "label": "cat", "split": "gallery", "identity": "same"},
            {"image_path": "b.jpg", "label": "dog", "split": "gallery", "identity": "same"},
            {"image_path": "c.jpg", "label": "cat", "split": "query", "identity": "query"},
        ])

        with self.assertRaisesRegex(ValueError, "maps to multiple labels"):
            load_metadata(self.root, self.metadata)

    def test_load_metadata_rejects_missing_images_and_unrepresented_query_labels(self):
        self._image("a.jpg", (255, 0, 0))
        self._write_metadata([
            {"image_path": "a.jpg", "label": "cat", "split": "gallery", "identity": "a"},
            {"image_path": "missing.jpg", "label": "dog", "split": "query", "identity": "b"},
        ])
        with self.assertRaises(FileNotFoundError):
            load_metadata(self.root, self.metadata)

        self._image("b.jpg", (0, 255, 0))
        self._write_metadata([
            {"image_path": "a.jpg", "label": "cat", "split": "gallery", "identity": "a"},
            {"image_path": "b.jpg", "label": "dog", "split": "query", "identity": "b"},
        ])
        with self.assertRaisesRegex(ValueError, "represented in the gallery"):
            load_metadata(self.root, self.metadata)

    def test_duplicate_detection_and_fingerprint_are_reproducible(self):
        self._image("gallery/a.jpg", (100, 20, 30))
        self._image("query/b.jpg", (100, 20, 30))
        self._write_metadata([
            {"image_path": "gallery/a.jpg", "label": "cat", "split": "gallery", "identity": "a"},
            {"image_path": "query/b.jpg", "label": "cat", "split": "query", "identity": "b"},
        ])
        records = load_metadata(self.root, self.metadata)

        duplicates = detect_duplicates(records, near_distance=0)
        first = dataset_fingerprint(records, duplicates["content_hashes"], [self.metadata])
        second = dataset_fingerprint(records, duplicates["content_hashes"], [self.metadata])

        self.assertEqual(len(duplicates["exact_cross_split_pairs"]), 1)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
