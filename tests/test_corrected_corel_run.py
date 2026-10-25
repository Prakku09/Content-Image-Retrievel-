import unittest

from pathlib import Path

from corrected_corel_run import (
    _group_disjoint_query_split,
    _group_disjoint_split,
    _group_id_map,
)


class CorrectedCorelRunTests(unittest.TestCase):
    def test_duplicate_groups_are_kept_in_one_mlp_partition(self):
        gallery_paths = [f"horses/{index:03d}.jpg" for index in range(90)]
        labels = ["horses"] * len(gallery_paths)
        duplicate_groups = _group_id_map(
            [],
            [{
                "left_path": gallery_paths[0],
                "right_path": gallery_paths[1],
                "distance": 4,
            }],
        )

        train, validation = _group_disjoint_split(
            gallery_paths, labels, duplicate_groups, seed=42
        )
        train_paths = {gallery_paths[index] for index in train}
        validation_paths = {gallery_paths[index] for index in validation}

        self.assertFalse(train_paths & validation_paths)
        self.assertEqual(train_paths | validation_paths, set(gallery_paths))
        self.assertEqual(
            duplicate_groups[gallery_paths[0]],
            duplicate_groups[gallery_paths[1]],
        )
        self.assertFalse(
            {duplicate_groups.get(path, path) for path in train_paths}
            & {
                duplicate_groups.get(path, path)
                for path in validation_paths
            }
        )

    def test_group_disjoint_split_is_reproducible(self):
        paths = [f"beaches/{index:03d}.jpg" for index in range(90)]
        labels = ["beaches"] * len(paths)
        first = _group_disjoint_split(paths, labels, {}, seed=91)
        second = _group_disjoint_split(paths, labels, {}, seed=91)
        self.assertEqual(first, second)
        self.assertEqual((len(first[0]), len(first[1])), (70, 20))

    def test_query_split_keeps_near_duplicate_groups_together(self):
        paths = [
            Path(f"horses/{index:03d}.jpg")
            for index in range(100)
        ]
        groups = _group_id_map(
            [],
            [{
                "left_path": paths[0],
                "right_path": paths[1],
                "distance": 4,
            }],
        )

        gallery, queries = _group_disjoint_query_split(paths, groups, seed=42)

        self.assertEqual(len(gallery), 90)
        self.assertEqual(len(queries), 10)
        pair_partitions = {
            "query" if path in queries else "gallery"
            for path in paths[:2]
        }
        self.assertEqual(len(pair_partitions), 1)
        self.assertEqual(set(gallery) | set(queries), set(paths))


if __name__ == "__main__":
    unittest.main()
