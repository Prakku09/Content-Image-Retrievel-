"""Run a duplicate-audited Corel-1K evaluation in an isolated output directory."""

import argparse
import csv
import hashlib
import json
import pickle
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import config
from data.dataset_loader import load_dataset
from external_dataset import _perceptual_hash
from features.feature_extractor import extract_all_features
from features.normalization import MinMaxNormalizer
from model.mlp_trainer import train_mlp
from retrieval.evaluation import evaluate_retrieval
from retrieval.retrieval import (
    get_representation,
    validate_fusion_weights,
    weighted_euclidean_distances,
)


ROOT = Path(__file__).resolve().parent
NEAR_DUPLICATE_DISTANCE = 4
QUERY_SEED = 42
MLP_SEED = 42


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args):
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _dataset_inventory(gallery_paths, query_paths):
    all_paths = sorted(set(gallery_paths) | set(query_paths))
    exact_groups = {}
    hashes = {}
    perceptual = {}
    for path in all_paths:
        digest = _sha256(path)
        hashes[path] = digest
        exact_groups.setdefault(digest, []).append(path)
        perceptual[path] = _perceptual_hash(path)

    exact_pairs = []
    for digest, paths in exact_groups.items():
        if len(paths) > 1:
            for left_index, left in enumerate(paths):
                for right in paths[left_index + 1:]:
                    exact_pairs.append({
                        "left_path": left,
                        "right_path": right,
                        "sha256": digest,
                    })

    near_pairs = []
    for left_index, left in enumerate(all_paths):
        for right in all_paths[left_index + 1:]:
            distance = (perceptual[left] ^ perceptual[right]).bit_count()
            if distance <= NEAR_DUPLICATE_DISTANCE:
                near_pairs.append({
                    "left_path": left,
                    "right_path": right,
                    "distance": distance,
                })
    return all_paths, hashes, perceptual, exact_pairs, near_pairs


def _group_id_map(exact_pairs, near_pairs):
    parents = {}

    def find(path):
        parents.setdefault(path, path)
        if parents[path] != path:
            parents[path] = find(parents[path])
        return parents[path]

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for pair in exact_pairs:
        union(pair["left_path"], pair["right_path"])
    for pair in near_pairs:
        union(pair["left_path"], pair["right_path"])
    groups = {}
    for path in list(parents):
        groups.setdefault(find(path), []).append(path)
    ordered = sorted((sorted(paths) for paths in groups.values()), key=lambda x: x[0])
    return {
        path: f"duplicate-{number:04d}"
        for number, paths in enumerate(ordered, start=1)
        for path in paths
    }


def _select_groups_with_total(groups, target, rng):
    shuffled = list(groups)
    rng.shuffle(shuffled)
    reachable = {0: []}
    for group_id, members in shuffled:
        size = len(members)
        for current_total, selected in list(reachable.items())[::-1]:
            next_total = current_total + size
            if next_total <= target and next_total not in reachable:
                reachable[next_total] = selected + [group_id]
    if target in reachable:
        return set(reachable[target])

    selected = set()
    count = 0
    for group_id, members in shuffled:
        if count + len(members) <= target:
            selected.add(group_id)
            count += len(members)
    return selected


def _group_disjoint_query_split(all_paths, group_ids, seed):
    rng = np.random.default_rng(seed)
    query_paths = []
    gallery_paths = []
    for category in sorted({path.parent.name for path in all_paths}):
        category_paths = [
            path for path in all_paths if path.parent.name == category
        ]
        groups = {}
        for path in category_paths:
            group_id = group_ids.get(path, path)
            groups.setdefault(group_id, []).append(path)
        selected_groups = _select_groups_with_total(
            list(groups.items()), config.QUERY_IMAGES_PER_CAT, rng
        )
        if sum(len(groups[group_id]) for group_id in selected_groups) != config.QUERY_IMAGES_PER_CAT:
            raise ValueError(
                f"Could not select exactly {config.QUERY_IMAGES_PER_CAT} "
                f"group-disjoint queries for category {category!r}."
            )
        query_paths.extend(
            path for group_id in selected_groups for path in groups[group_id]
        )
        gallery_paths.extend(
            path for group_id, paths in groups.items()
            if group_id not in selected_groups
            for path in paths
        )
    return sorted(gallery_paths), sorted(query_paths)


def _group_disjoint_split(gallery_paths, gallery_labels, group_ids, seed):
    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []
    categories = sorted(set(gallery_labels))
    for category in categories:
        indices = [
            index for index, label in enumerate(gallery_labels)
            if label == category
        ]
        groups = {}
        for index in indices:
            group = group_ids.get(gallery_paths[index], gallery_paths[index])
            groups.setdefault(group, []).append(index)
        group_ids_for_category = list(groups)
        target_validation = min(
            config.INTERNAL_VAL_PER_CAT,
            max(1, len(indices) - config.INTERNAL_TRAIN_PER_CAT),
        )
        selected_validation_groups = _select_groups_with_total(
            [(group_id, groups[group_id]) for group_id in group_ids_for_category],
            target_validation,
            rng,
        )
        selected_validation = [
            index for group_id in selected_validation_groups
            for index in groups[group_id]
        ]
        selected_train = [
            index for group_id, members in groups.items()
            if group_id not in selected_validation_groups
            for index in members
        ]
        if len(selected_validation) != target_validation or not selected_train:
            raise ValueError(f"Category {category!r} has no MLP training images.")
        train_indices.extend(selected_train)
        validation_indices.extend(selected_validation)

    train_indices.sort()
    validation_indices.sort()
    if set(train_indices) & set(validation_indices):
        raise AssertionError("Train/validation split overlaps.")
    if set(train_indices) | set(validation_indices) != set(range(len(gallery_paths))):
        raise AssertionError("Train/validation split omits gallery images.")
    train_groups = {
        group_ids.get(gallery_paths[index], gallery_paths[index])
        for index in train_indices
    }
    validation_groups = {
        group_ids.get(gallery_paths[index], gallery_paths[index])
        for index in validation_indices
    }
    if train_groups & validation_groups:
        raise ValueError("A duplicate group crosses MLP train/validation.")
    return train_indices, validation_indices


def _save_csv(path, rows, fieldnames=None):
    if not rows and not fieldnames:
        raise ValueError(f"Cannot infer CSV columns for empty output: {path}")
    columns = fieldnames or list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _feature_fingerprint(all_paths, split_payload):
    source_files = (
        ROOT / "features" / "feature_extractor.py",
        ROOT / "features" / "normalization.py",
        ROOT / "data" / "dataset_loader.py",
    )
    digest = hashlib.sha256(json.dumps(
        split_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))
    for source in source_files:
        digest.update(source.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(source)))
    for path in all_paths:
        digest.update(path.relative_to(config.DATASET_PATH).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _extract(paths):
    colors = np.empty((len(paths), config.COLOR_FEATURE_DIM), dtype=np.float32)
    textures = np.empty((len(paths), config.TEXTURE_FEATURE_DIM), dtype=np.float32)
    elapsed = np.empty(len(paths), dtype=np.float64)
    for index, path in enumerate(paths):
        started = time.perf_counter()
        colors[index], textures[index] = extract_all_features(path)
        elapsed[index] = time.perf_counter() - started
        if (index + 1) % 100 == 0 or index + 1 == len(paths):
            print(f"Extracted features {index + 1}/{len(paths)}")
    if not np.all(np.isfinite(colors)) or not np.all(np.isfinite(textures)):
        raise ValueError("Feature extraction produced non-finite values.")
    return colors, textures, elapsed


def run(
    output_root=None,
    dataset_root=None,
    query_seed=QUERY_SEED,
    mlp_seed=MLP_SEED,
    group_disjoint_queries=False,
):
    dataset_root = Path(dataset_root or config.DATASET_PATH).resolve()
    if dataset_root != config.DATASET_PATH:
        raise ValueError(
            "The existing Corel loader is configured for "
            f"{config.DATASET_PATH}, not {dataset_root}."
        )
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    output_root = Path(output_root or ROOT / "output" / "corrected_corel_v1")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    gallery_paths, gallery_labels, query_paths, query_labels = load_dataset(
        dataset_path=dataset_root, seed=query_seed
    )
    all_paths, file_hashes, phashes, exact_pairs, near_pairs = _dataset_inventory(
        gallery_paths, query_paths
    )
    group_ids = _group_id_map(exact_pairs, near_pairs)
    if group_disjoint_queries:
        gallery_paths, query_paths = _group_disjoint_query_split(
            all_paths, group_ids, query_seed
        )
        gallery_labels = [path.parent.name for path in gallery_paths]
        query_labels = [path.parent.name for path in query_paths]
    query_set = set(query_paths)
    excluded_paths = set()
    if not group_disjoint_queries:
        for pair in exact_pairs + near_pairs:
            left, right = pair["left_path"], pair["right_path"]
            if left in query_set:
                excluded_paths.add(right)
            if right in query_set:
                excluded_paths.add(left)
    excluded_paths -= query_set

    included = [
        (path, label) for path, label in zip(gallery_paths, gallery_labels)
        if path not in excluded_paths
    ]
    corrected_gallery_paths = [path for path, _ in included]
    corrected_gallery_labels = [label for _, label in included]
    expected_gallery_count = 900 if group_disjoint_queries else 899
    if (
        len(query_paths) != 100
        or len(corrected_gallery_paths) != expected_gallery_count
    ):
        raise ValueError(
            f"Expected 100 queries and {expected_gallery_count} gallery images "
            "under the selected duplicate protocol; got "
            f"{len(query_paths)} and "
            f"{len(corrected_gallery_paths)}."
        )
    if len(query_labels) != len(query_paths):
        raise AssertionError("Query labels and paths differ in length.")

    train_indices, validation_indices = _group_disjoint_split(
        corrected_gallery_paths,
        corrected_gallery_labels,
        group_ids,
        mlp_seed,
    )
    train_paths = {corrected_gallery_paths[index] for index in train_indices}
    validation_paths = {
        corrected_gallery_paths[index] for index in validation_indices
    }
    if (train_paths | validation_paths) & query_set:
        raise AssertionError("A held-out query entered training or validation.")
    if (train_paths | validation_paths) & excluded_paths:
        raise AssertionError("An excluded near-duplicate entered training or validation.")

    # This dataset has one query-near-duplicate pair. The gallery member is
    # manually confirmed from the full-resolution side-by-side audit.
    confirmed_pairs = {
        frozenset((
            "training_set/horses/761.jpg",
            "training_set/horses/776.jpg",
        ))
    }
    observed_near_pairs = {
        frozenset(
            path.relative_to(dataset_root).as_posix()
            for path in (pair["left_path"], pair["right_path"])
        )
        for pair in near_pairs
    }
    if confirmed_pairs != observed_near_pairs:
        raise ValueError(
            "Duplicate audit candidates differ from the visually reviewed "
            "Corel pair. Review candidates before proceeding."
        )

    partition_by_path = {}
    for index, path in enumerate(corrected_gallery_paths):
        partition_by_path[path] = (
            "mlp_train" if index in set(train_indices) else "mlp_validation"
        )
    for path in query_paths:
        partition_by_path[path] = "held_out_query"
    for path in excluded_paths:
        partition_by_path[path] = "excluded_near_duplicate_of_query"

    split_rows = []
    for path in all_paths:
        relative = path.relative_to(dataset_root).as_posix()
        split_rows.append({
            "relative_path": relative,
            "category": path.parent.name,
            "partition": partition_by_path.get(path, "not_selected"),
            "sha256": file_hashes[path],
            "perceptual_hash_hex": f"{phashes[path]:016x}",
            "duplicate_group_id": group_ids.get(path, ""),
            "exclusion_reason": (
                "near/exact duplicate of held-out query"
                if path in excluded_paths else ""
            ),
            "query_split_seed": query_seed,
            "mlp_split_seed": mlp_seed,
        })
    _save_csv(run_dir / "split_manifest.csv", split_rows)

    duplicate_report = {
        "dataset_path": str(dataset_root),
        "image_count": len(all_paths),
        "hash_algorithm": "SHA-256",
        "perceptual_hash": "32x32 grayscale DCT, top-left 8x8, median threshold",
        "near_duplicate_hamming_threshold": NEAR_DUPLICATE_DISTANCE,
        "exact_duplicate_pair_count": len(exact_pairs),
        "exact_duplicate_pairs": [
            {
                "left_path": pair["left_path"].relative_to(dataset_root).as_posix(),
                "right_path": pair["right_path"].relative_to(dataset_root).as_posix(),
            }
            for pair in exact_pairs
        ],
        "near_duplicate_candidates": [
            {
                "left_path": pair["left_path"].relative_to(dataset_root).as_posix(),
                "right_path": pair["right_path"].relative_to(dataset_root).as_posix(),
                "perceptual_hash_distance": pair["distance"],
                "visual_review": "confirmed same scene and pose",
                "excluded_from_gallery": [
                    path.relative_to(dataset_root).as_posix()
                    for path in (pair["left_path"], pair["right_path"])
                    if path in excluded_paths
                ],
            }
            for pair in near_pairs
        ],
        "excluded_gallery_paths": sorted(
            path.relative_to(dataset_root).as_posix()
            for path in excluded_paths
        ),
        "all_queries_retained": len(query_paths) == 100,
        "query_gallery_duplicate_candidates_after_exclusion": 0,
        "remaining_train_validation_duplicate_groups": 0,
        "visual_review_scope": "The only thresholded pHash pair was reviewed side by side.",
    }
    (run_dir / "duplicate_audit.json").write_text(
        json.dumps(duplicate_report, indent=2), encoding="utf-8"
    )

    feature_started = time.perf_counter()
    combined_paths = corrected_gallery_paths + query_paths
    raw_color, raw_texture, extraction_times = _extract(combined_paths)
    extraction_total = time.perf_counter() - feature_started
    gallery_count = len(corrected_gallery_paths)
    gallery_color = raw_color[:gallery_count]
    query_color = raw_color[gallery_count:]
    gallery_texture = raw_texture[:gallery_count]
    query_texture = raw_texture[gallery_count:]

    color_normalizer = MinMaxNormalizer()
    texture_normalizer = MinMaxNormalizer()
    normalized_gallery_color = color_normalizer.fit_transform(gallery_color)
    normalized_query_color = color_normalizer.transform(query_color)
    normalized_gallery_texture = texture_normalizer.fit_transform(gallery_texture)
    normalized_query_texture = texture_normalizer.transform(query_texture)

    split_payload = {
        "query_seed": query_seed,
        "mlp_seed": mlp_seed,
        "group_disjoint_query_split": bool(group_disjoint_queries),
        "gallery_paths": [
            path.relative_to(dataset_root).as_posix()
            for path in corrected_gallery_paths
        ],
        "query_paths": [
            path.relative_to(dataset_root).as_posix() for path in query_paths
        ],
        "excluded_paths": sorted(
            path.relative_to(dataset_root).as_posix() for path in excluded_paths
        ),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }
    feature_fingerprint = _feature_fingerprint(combined_paths, split_payload)
    feature_metadata = {
        "dataset_path": str(dataset_root),
        "feature_fingerprint": feature_fingerprint,
        "feature_cache_schema_version": 2,
        "gallery_image_paths": [
            path.relative_to(dataset_root).as_posix()
            for path in corrected_gallery_paths
        ],
        "query_image_paths": [
            path.relative_to(dataset_root).as_posix() for path in query_paths
        ],
        "gallery_color_features": gallery_color,
        "gallery_texture_features": gallery_texture,
        "query_color_features": query_color,
        "query_texture_features": query_texture,
        "normalized_gallery_color": normalized_gallery_color,
        "normalized_gallery_texture": normalized_gallery_texture,
        "normalized_query_color": normalized_query_color,
        "normalized_query_texture": normalized_query_texture,
        "normalization_parameters": {
            "color_min": color_normalizer.min_val.tolist(),
            "color_max": color_normalizer.max_val.tolist(),
            "texture_min": texture_normalizer.min_val.tolist(),
            "texture_max": texture_normalizer.max_val.tolist(),
        },
    }
    cache_path = run_dir / "corrected_features.pkl"
    with cache_path.open("wb") as output:
        pickle.dump(feature_metadata, output)
    np.savez_compressed(
        run_dir / "features.npz",
        paths=np.asarray([
            path.relative_to(dataset_root).as_posix() for path in combined_paths
        ]),
        raw_color=raw_color,
        raw_texture=raw_texture,
        normalized_color=np.concatenate(
            [normalized_gallery_color, normalized_query_color]
        ),
        normalized_texture=np.concatenate(
            [normalized_gallery_texture, normalized_query_texture]
        ),
    )

    print("Training the existing MLP on corrected, group-disjoint partitions.")
    training_started = time.perf_counter()
    checkpoint_path = run_dir / "m5_checkpoint.pt"
    model, split_info, history = train_mlp(
        data_path=cache_path,
        seed=mlp_seed,
        checkpoint_path=checkpoint_path,
        split_indices={
            "train_indices": train_indices,
            "val_indices": validation_indices,
        },
    )
    training_seconds = time.perf_counter() - training_started

    # Write final partition assignments only after verifying the trained
    # checkpoint records the requested split and feature fingerprint.
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if checkpoint.get("feature_fingerprint") != feature_fingerprint:
        raise AssertionError("Checkpoint feature fingerprint does not match this run.")
    if (
        checkpoint["split_info"]["train_indices"] != train_indices
        or checkpoint["split_info"]["val_indices"] != validation_indices
    ):
        raise AssertionError("Checkpoint split does not match the explicit manifest.")
    checkpoint["normalization_parameters"] = feature_metadata[
        "normalization_parameters"
    ]
    checkpoint["split_manifest_sha256"] = _sha256(run_dir / "split_manifest.csv")
    torch.save(checkpoint, checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # M5 inference is timed per query. Weights are then frozen for M6 by
    # averaging only these predictions; no labels enter this calculation.
    inference_times = []
    predicted_weights = []
    for color, texture in zip(normalized_query_color, normalized_query_texture):
        feature = np.concatenate([color, texture]).astype(np.float32)
        started = time.perf_counter()
        with torch.no_grad():
            output = model(torch.tensor(feature).unsqueeze(0)).cpu().numpy()[0]
        inference_times.append(time.perf_counter() - started)
        predicted_weights.append(output)
    predicted_weights = validate_fusion_weights(
        np.asarray(predicted_weights, dtype=np.float64), atol=1e-6
    )
    mean_weights = validate_fusion_weights(
        np.mean(predicted_weights, axis=0), atol=1e-6
    )

    gallery_labels = list(corrected_gallery_labels)
    query_labels = list(query_labels)
    gallery_paths_rel = [
        path.relative_to(dataset_root).as_posix()
        for path in corrected_gallery_paths
    ]
    query_paths_rel = [
        path.relative_to(dataset_root).as_posix() for path in query_paths
    ]
    all_rankings = {}
    per_query_rows = []
    runtime_rows = []
    model_names = ("M1", "M2", "M3", "M4", "M5", "M6")
    for model_name in model_names[:4]:
        gallery_repr = get_representation(
            normalized_gallery_color, normalized_gallery_texture, model_name
        )
        query_repr = get_representation(
            normalized_query_color, normalized_query_texture, model_name
        )
        ranks = []
        times = []
        for query_index, query_repr_row in enumerate(query_repr):
            started = time.perf_counter()
            distances = np.sqrt(np.sum((gallery_repr - query_repr_row) ** 2, axis=1))
            ranking = np.argsort(distances)
            elapsed = time.perf_counter() - started
            ranks.append(ranking)
            times.append(elapsed)
            metric = evaluate_retrieval(
                ranking, gallery_labels, query_labels[query_index], config.EVAL_K
            )
            per_query_rows.append({
                "model": model_name,
                "query_path": query_paths_rel[query_index],
                "category": query_labels[query_index],
                "average_precision": metric["ap"],
                "precision_at_10": metric["precision_k"],
                "recall_at_10": metric["recall_k"],
                "f1_at_10": metric["f1_k"],
                "ranking_sort_seconds": elapsed,
            })
        all_rankings[model_name] = np.asarray(ranks, dtype=np.int32)
        runtime_rows.extend({
            "model": model_name,
            "query_path": query_paths_rel[i],
            "feature_extraction_seconds": float(extraction_times[gallery_count + i]),
            "mlp_inference_seconds": 0.0,
            "ranking_sort_seconds": float(times[i]),
            "query_latency_excluding_gallery_preprocessing_seconds": (
                float(extraction_times[gallery_count + i]) + float(times[i])
            ),
        } for i in range(len(query_paths)))

    m5_m6_weights = [predicted_weights, np.tile(mean_weights, (len(query_paths), 1))]
    for model_name, weights_by_query in zip(("M5", "M6"), m5_m6_weights):
        ranks = []
        times = []
        for query_index, weights in enumerate(weights_by_query):
            started = time.perf_counter()
            distances = weighted_euclidean_distances(
                normalized_query_color[query_index],
                normalized_gallery_color,
                normalized_query_texture[query_index],
                normalized_gallery_texture,
                weights,
            )
            ranking = np.argsort(distances)
            elapsed = time.perf_counter() - started
            ranks.append(ranking)
            times.append(elapsed)
            metric = evaluate_retrieval(
                ranking, gallery_labels, query_labels[query_index], config.EVAL_K
            )
            per_query_rows.append({
                "model": model_name,
                "query_path": query_paths_rel[query_index],
                "category": query_labels[query_index],
                "average_precision": metric["ap"],
                "precision_at_10": metric["precision_k"],
                "recall_at_10": metric["recall_k"],
                "f1_at_10": metric["f1_k"],
                "ranking_sort_seconds": elapsed,
            })
        all_rankings[model_name] = np.asarray(ranks, dtype=np.int32)
        for i in range(len(query_paths)):
            runtime_rows.append({
                "model": model_name,
                "query_path": query_paths_rel[i],
                "feature_extraction_seconds": float(extraction_times[gallery_count + i]),
                "mlp_inference_seconds": float(inference_times[i]),
                "ranking_sort_seconds": float(times[i]),
                "query_latency_excluding_gallery_preprocessing_seconds": (
                    float(extraction_times[gallery_count + i])
                    + float(inference_times[i])
                    + float(times[i])
                ),
            })

    aggregate = {}
    for model_name in model_names:
        rows = [row for row in per_query_rows if row["model"] == model_name]
        aggregate[model_name] = {
            "query_count": len(rows),
            "mean_average_precision": float(np.mean([
                row["average_precision"] for row in rows
            ])),
            "mean_precision_at_10": float(np.mean([
                row["precision_at_10"] for row in rows
            ])),
            "mean_recall_at_10": float(np.mean([
                row["recall_at_10"] for row in rows
            ])),
            "mean_f1_at_10": float(np.mean([
                row["f1_at_10"] for row in rows
            ])),
            "mean_ranking_sort_seconds": float(np.mean([
                row["ranking_sort_seconds"] for row in rows
            ])),
        }

    category_rows = []
    for category in sorted(set(query_labels)):
        for model_name in model_names:
            rows = [
                row for row in per_query_rows
                if row["category"] == category and row["model"] == model_name
            ]
            category_rows.append({
                "category": category,
                "model": model_name,
                "query_count": len(rows),
                "mean_average_precision": float(np.mean([
                    row["average_precision"] for row in rows
                ])),
                "mean_precision_at_10": float(np.mean([
                    row["precision_at_10"] for row in rows
                ])),
                "mean_recall_at_10": float(np.mean([
                    row["recall_at_10"] for row in rows
                ])),
                "mean_f1_at_10": float(np.mean([
                    row["f1_at_10"] for row in rows
                ])),
            })
    category_lookup = {
        (row["category"], row["model"]): row["mean_average_precision"]
        for row in category_rows
    }
    category_differences = [
        {
            "category": category,
            "m5_minus_m6_mean_ap": (
                category_lookup[(category, "M5")]
                - category_lookup[(category, "M6")]
            ),
        }
        for category in sorted(set(query_labels))
    ]
    ap_lookup = {
        (row["query_path"], row["model"]): row["average_precision"]
        for row in per_query_rows
    }
    ap_differences = np.asarray([
        ap_lookup[(path, "M5")] - ap_lookup[(path, "M6")]
        for path in query_paths_rel
    ])
    difference_summary = {
        "estimand": "mean paired per-query AP difference M5 - M6 on this fixed corrected query/gallery split",
        "mean": float(np.mean(ap_differences)),
        "median": float(np.median(ap_differences)),
        "standard_deviation": float(np.std(ap_differences, ddof=1)),
        "positive_queries": int(np.sum(ap_differences > 0)),
        "negative_queries": int(np.sum(ap_differences < 0)),
        "tied_queries": int(np.sum(ap_differences == 0)),
        "category_mean_differences_positive": sum(
            row["m5_minus_m6_mean_ap"] > 0 for row in category_differences
        ),
        "category_mean_differences_negative": sum(
            row["m5_minus_m6_mean_ap"] < 0 for row in category_differences
        ),
        "inference": (
            "Descriptive only; no inferential p-value is reported. Query APs share "
            "the fixed gallery and are category-dependent."
        ),
    }

    _save_csv(run_dir / "per_query_metrics.csv", per_query_rows)
    _save_csv(run_dir / "per_category_metrics.csv", category_rows)
    _save_csv(run_dir / "category_m5_minus_m6.csv", category_differences)
    _save_csv(run_dir / "runtime_breakdown.csv", runtime_rows)
    _save_csv(run_dir / "per_query_weights.csv", [
        {
            "query_path": query_paths_rel[i],
            "category": query_labels[i],
            "m5_color_weight": float(predicted_weights[i, 0]),
            "m5_texture_weight": float(predicted_weights[i, 1]),
            "m6_color_weight": float(mean_weights[0]),
            "m6_texture_weight": float(mean_weights[1]),
        }
        for i in range(len(query_paths))
    ])
    np.savez_compressed(
        run_dir / "rankings.npz",
        query_paths=np.asarray(query_paths_rel),
        gallery_paths=np.asarray(gallery_paths_rel),
        **all_rankings,
    )
    np.savez_compressed(
        run_dir / "per_query_ap_differences.npz",
        query_paths=np.asarray(query_paths_rel),
        m5_minus_m6=ap_differences,
    )

    result = {
        "run_id": run_id,
        "dataset_fingerprint": feature_fingerprint,
        "query_seed": query_seed,
        "mlp_seed": mlp_seed,
        "group_disjoint_queries": bool(group_disjoint_queries),
        "counts": {
            "dataset_images": len(all_paths),
            "gallery": len(corrected_gallery_paths),
            "queries": len(query_paths),
            "excluded_duplicate_gallery": len(excluded_paths),
            "mlp_train": len(train_indices),
            "mlp_validation": len(validation_indices),
        },
        "aggregate_metrics": aggregate,
        "m5_m6_paired_ap_difference": difference_summary,
        "category_m5_minus_m6_ap": category_differences,
        "normalization_parameters": feature_metadata["normalization_parameters"],
        "m6_mean_weights": [float(value) for value in mean_weights],
        "timing": {
            "feature_extraction_total_seconds": extraction_total,
            "feature_extraction_mean_seconds_per_image": float(np.mean(extraction_times)),
            "training_seconds": training_seconds,
            "mlp_inference_mean_seconds_per_query": float(np.mean(inference_times)),
            "ranking_sort_mean_seconds_per_query": {
                model: aggregate[model]["mean_ranking_sort_seconds"]
                for model in model_names
            },
            "method": (
                "Feature extraction timed per image; M5 forward pass timed per query; "
                "ranking timed as distance calculation plus argsort. End-to-end "
                "query latency adds query feature extraction, M5 inference where "
                "applicable, and ranking; gallery preprocessing/training excluded."
            ),
        },
    }
    (run_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    reconciliation_rows = []
    historical_path = ROOT / "output" / "part2" / "experiment_results.pkl"
    metric_map = {
        "mAP": ("mAP", "mean_average_precision"),
        "Precision@10": ("mean_precision", "mean_precision_at_10"),
        "Recall@10": ("mean_recall", "mean_recall_at_10"),
        "F1@10": ("mean_f1", "mean_f1_at_10"),
    }
    historical = None
    if historical_path.is_file():
        with historical_path.open("rb") as source:
            historical = pickle.load(source)
    for model_name in model_names:
        new = aggregate[model_name]
        row = {
            "model": model_name,
            "historical_source": (
                str(historical_path) if historical is not None else ""
            ),
            "historical_status": (
                "superseded for corrected protocol"
                if historical is not None else "not available in this source snapshot"
            ),
        }
        for label, (old_key, new_key) in metric_map.items():
            old_value = (
                float(historical["metrics"][model_name][old_key])
                if historical is not None else ""
            )
            new_value = float(new[new_key])
            row[f"historical_{label}"] = old_value
            row[f"corrected_{label}"] = new_value
            row[f"corrected_minus_historical_{label}"] = (
                new_value - old_value if historical is not None else ""
            )
        reconciliation_rows.append(row)
    _save_csv(run_dir / "paper_table_reconciliation.csv", reconciliation_rows)

    checkpoint_metadata = {
        "checkpoint_path": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_format": checkpoint.get("checkpoint_format"),
        "feature_fingerprint": feature_fingerprint,
        "training_config": checkpoint["training_config"],
        "split_info": checkpoint["split_info"],
        "model_state_dict_restored_from_best_validation_epoch": True,
        "normalization_parameters": feature_metadata["normalization_parameters"],
        "split_manifest_sha256": _sha256(run_dir / "split_manifest.csv"),
    }
    (run_dir / "checkpoint_metadata.json").write_text(
        json.dumps(checkpoint_metadata, indent=2), encoding="utf-8"
    )
    (run_dir / "training_history.json").write_text(json.dumps({
        "split_info": split_info,
        "history": history,
    }, indent=2), encoding="utf-8")

    source_files = [
        ROOT / "corrected_corel_run.py",
        ROOT / "config.py",
        ROOT / "data" / "dataset_loader.py",
        ROOT / "features" / "feature_extractor.py",
        ROOT / "features" / "normalization.py",
        ROOT / "model" / "mlp_model.py",
        ROOT / "model" / "mlp_trainer.py",
        ROOT / "retrieval" / "retrieval.py",
        ROOT / "retrieval" / "evaluation.py",
    ]
    config_payload = {
        "dataset_path": str(dataset_root),
        "feature_order": [
            "h_mean", "h_std", "h_skew", "s_mean", "s_std", "s_skew",
            "v_mean", "v_std", "v_skew", "glcm_contrast",
            "glcm_correlation", "glcm_energy", "glcm_homogeneity",
        ],
        "feature_extraction": {
            "glcm_distance": config.GLCM_DISTANCE,
            "glcm_angles_deg": config.GLCM_ANGLES_DEG,
        },
        "normalization": "MinMax fitted on corrected retrieval gallery only.",
        "models": {
            "M1": "color-only Euclidean",
            "M2": "texture-only Euclidean",
            "M3": "equal feature-level weights; weighted concatenation",
            "M4": "0.7/0.3 feature-level weights; weighted concatenation",
            "M5": "query-adaptive direct distance-level weights",
            "M6": "mean M5 query weights; direct distance-level formula",
        },
        "M3_M4_distance": "sqrt(wc^2 * dc^2 + wt^2 * dt^2)",
        "M5_M6_distance": "sqrt(wc * dc^2 + wt * dt^2)",
        "evaluation_k": config.EVAL_K,
        "query_seed": query_seed,
        "mlp_seed": mlp_seed,
        "mlp_hyperparameters": {
            "architecture": [13, 16, 8, 2],
            "optimizer": "Adam",
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "maximum_epochs": config.EPOCHS,
            "triplet_margin": config.MARGIN,
            "patience": 10,
            "training_positive_samples": 5,
            "training_negative_samples": 10,
            "validation_positive_samples": 2,
            "validation_negative_samples": 5,
        },
        "split_manifest_sha256": _sha256(run_dir / "split_manifest.csv"),
        "duplicate_audit_sha256": _sha256(run_dir / "duplicate_audit.json"),
        "feature_fingerprint": feature_fingerprint,
    }
    (run_dir / "experiment_config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )
    try:
        import importlib.metadata
        dependencies = {
            package: importlib.metadata.version(package)
            for package in ("numpy", "torch", "Pillow", "scipy", "scikit-image")
        }
    except importlib.metadata.PackageNotFoundError:
        dependencies = {"status": "version lookup incomplete"}
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": _git("rev-parse", "HEAD"),
        "worktree_status_at_run": _git("status", "--short"),
        "dataset_fingerprint_sha256": feature_fingerprint,
        "dataset_file_hashes": {
            path.relative_to(dataset_root).as_posix(): file_hashes[path]
            for path in all_paths
        },
        "source_file_hashes": {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_files
        },
        "dependency_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            **dependencies,
        },
        "checkpoint_sha256": _sha256(checkpoint_path),
        "configuration_sha256": _sha256(run_dir / "experiment_config.json"),
        "split_manifest_sha256": _sha256(run_dir / "split_manifest.csv"),
        "duplicate_audit_sha256": _sha256(run_dir / "duplicate_audit.json"),
        "artifact_sha256": {},
        "validation": {
            "held_out_queries_in_train_or_validation": False,
            "query_near_duplicates_in_gallery_or_training": False,
            "normalization_fit_on_queries": False,
            "m6_mean_from_new_checkpoint_predictions": True,
            "all_six_models_same_gallery_queries_labels_and_metrics": True,
            "metrics_recomputed_from_rankings": "pending independent validation",
        },
    }
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name != "reproducibility_manifest.json":
            manifest["artifact_sha256"][path.name] = _sha256(path)
    (run_dir / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    _write_validation_report(
        run_dir,
        result,
        duplicate_report,
        split_rows,
        aggregate,
        ap_differences,
    )
    print(f"Corrected Corel run artifacts: {run_dir}")
    return run_dir


def _write_validation_report(
    run_dir, result, duplicate_report, split_rows, aggregate, ap_differences
):
    metrics = [
        "| {model} | {mean_precision_at_10:.4f} | {mean_recall_at_10:.4f} | "
        "{mean_f1_at_10:.4f} | {mean_average_precision:.4f} | "
        "{mean_ranking_sort_seconds:.6f} |".format(model=model, **values)
        for model, values in aggregate.items()
    ]
    report = f"""# Corrected Corel-1K M1-M6 Evaluation

Run: `{result['run_id']}`. Historical outputs were not modified.

## Protocol and duplicates

- Full scan: {duplicate_report['image_count']} images; exact duplicate pairs: {duplicate_report['exact_duplicate_pair_count']}; pHash candidates at Hamming distance <= {NEAR_DUPLICATE_DISTANCE}: {len(duplicate_report['near_duplicate_candidates'])}.
- Duplicate-group-disjoint query split: {result['group_disjoint_queries']}. Query-side near-duplicate groups remain entirely in the held-out partition; gallery-side groups are kept intact in the gallery and assigned wholly to train or validation.
- Near-duplicate exclusions: {result['counts']['excluded_duplicate_gallery']}. All visually reviewed candidates and any exclusions are listed in `duplicate_audit.json`.
- Corrected gallery/query: {result['counts']['gallery']} / {result['counts']['queries']}; MLP train/validation: {result['counts']['mlp_train']} / {result['counts']['mlp_validation']}.
- Seeds: query split {result['query_seed']}; MLP/triplet sampling {result['mlp_seed']}.
- Normalization was fitted on the corrected retrieval gallery only. Query images and the excluded duplicate were not used for normalization or training/validation.
- MLP architecture/loss/hyperparameters unchanged. Existing trainer's `copy.deepcopy(state_dict())` best-state snapshot and restore were retained.
- M1-M4 use the existing Euclidean feature-level definitions. M5/M6 use the existing direct distance-level formula. M6 weights are the mean predicted by this run's newly trained M5 checkpoint.

## Aggregate results

| Model | Precision@10 | Recall@10 | F1@10 | mAP | Mean ranking/sort seconds per query |
|---|---:|---:|---:|---:|---:|
{chr(10).join(metrics)}

M5-M6 AP differences across queries: mean {result['m5_m6_paired_ap_difference']['mean']:.8f}, median {result['m5_m6_paired_ap_difference']['median']:.8f}, positive {result['m5_m6_paired_ap_difference']['positive_queries']}, negative {result['m5_m6_paired_ap_difference']['negative_queries']}, tied {result['m5_m6_paired_ap_difference']['tied_queries']}. Category-positive means: {result['m5_m6_paired_ap_difference']['category_mean_differences_positive']}; category-negative means: {result['m5_m6_paired_ap_difference']['category_mean_differences_negative']}. These are descriptive for one fixed gallery; no statistical superiority is claimed.

## Reproducibility and validation

- Dataset/configuration/source/checkpoint/artifact hashes and package versions are in `reproducibility_manifest.json`.
- Saved rankings, per-query metrics, per-category metrics, per-query weights, training history, corrected feature cache, checkpoint, and split/duplicate manifests are included.
- Timings separately record feature extraction, M5 inference, and ranking/sort. End-to-end query latency includes query feature extraction, applicable M5 inference, and ranking; gallery preprocessing and training are excluded.
- The complete corrected run finished. Independent ranking-based metric verification, unit tests, compilation and diff checks are recorded in `validation_report.md` after verification.

## Interpretation limitation

Queries share one corrected gallery and category labels. Query-level AP differences are paired descriptive observations, not independent replications of galleries or training runs. Results do not establish transfer or robust superiority beyond this split.
"""
    (run_dir / "validation_report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=config.DATASET_PATH)
    parser.add_argument("--query-seed", type=int, default=QUERY_SEED)
    parser.add_argument("--mlp-seed", type=int, default=MLP_SEED)
    parser.add_argument(
        "--group-disjoint-queries",
        action="store_true",
        help="Keep each identified duplicate group wholly in gallery or query.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output" / "corrected_corel_v1",
    )
    args = parser.parse_args()
    run(
        output_root=args.output_root,
        dataset_root=args.dataset_root,
        query_seed=args.query_seed,
        mlp_seed=args.mlp_seed,
        group_disjoint_queries=args.group_disjoint_queries,
    )


if __name__ == "__main__":
    main()
