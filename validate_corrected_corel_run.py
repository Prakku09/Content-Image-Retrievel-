"""Independently validate a completed corrected Corel-1K run."""

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from retrieval.evaluation import evaluate_retrieval


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_run(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads(
        (run_dir / "reproducibility_manifest.json").read_text(encoding="utf-8")
    )
    config = json.loads((run_dir / "experiment_config.json").read_text())
    summary = json.loads((run_dir / "results.json").read_text())
    duplicate_report = json.loads((run_dir / "duplicate_audit.json").read_text())
    checkpoint = torch.load(
        run_dir / "m5_checkpoint.pt", map_location="cpu", weights_only=True
    )
    with (run_dir / "corrected_features.pkl").open("rb") as source:
        features = pickle.load(source)

    with (run_dir / "split_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        split_rows = list(csv.DictReader(source))
    partition_by_path = {
        row["relative_path"]: row["partition"] for row in split_rows
    }
    query_paths = {
        path for path, partition in partition_by_path.items()
        if partition == "held_out_query"
    }
    excluded_paths = {
        path for path, partition in partition_by_path.items()
        if partition == "excluded_near_duplicate_of_query"
    }
    train_paths = {
        path for path, partition in partition_by_path.items()
        if partition == "mlp_train"
    }
    validation_paths = {
        path for path, partition in partition_by_path.items()
        if partition == "mlp_validation"
    }
    corrected_gallery_paths = set(features["gallery_image_paths"])
    corrected_query_paths = set(features["query_image_paths"])
    counts = summary["counts"]
    query_count = counts["queries"]
    gallery_count = counts["gallery"]
    training_count = counts["mlp_train"]
    validation_count = counts["mlp_validation"]
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    require(len(query_paths) == query_count, "Held-out query count differs from results.")
    require(len(corrected_gallery_paths) == gallery_count,
            "Gallery count differs from results.")
    require(len(train_paths) == training_count,
            "MLP train count differs from results.")
    require(len(validation_paths) == validation_count,
            "MLP validation count differs from results.")
    require(query_paths == corrected_query_paths, "Query paths mismatch feature artifact.")
    require(
        corrected_gallery_paths == train_paths | validation_paths,
        "Corrected gallery does not equal train plus validation.",
    )
    require(not (train_paths & validation_paths), "Train/validation paths overlap.")
    require(not (query_paths & (train_paths | validation_paths)),
            "Held-out query entered train or validation.")
    require(not (excluded_paths & (train_paths | validation_paths)),
            "Excluded duplicate entered train/validation.")
    if summary.get("group_disjoint_queries", False):
        require(not excluded_paths,
                "Group-disjoint split unexpectedly excluded duplicate images.")
        group_partitions = {}
        for row in split_rows:
            group_id = row["duplicate_group_id"]
            if group_id:
                group_partitions.setdefault(group_id, set()).add(row["partition"])
        require(
            all(len(partitions) == 1 for partitions in group_partitions.values()),
            "An identified duplicate group crosses split partitions.",
        )
    else:
        require(
            "training_set/horses/776.jpg" in query_paths,
            "Identified held-out query was not retained.",
        )
        require(
            "training_set/horses/761.jpg" in excluded_paths,
            "Identified duplicate gallery image was not excluded.",
        )
    require(
        not duplicate_report["exact_duplicate_pairs"],
        "Exact duplicate pairs were found.",
    )
    require(
        len(duplicate_report["near_duplicate_candidates"]) == 1,
        "Unexpected number of near duplicate candidates.",
    )
    require(
        checkpoint["feature_fingerprint"] == summary["dataset_fingerprint"],
        "Checkpoint feature fingerprint mismatch.",
    )
    checkpoint_metadata = json.loads(
        (run_dir / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    require(
        checkpoint["training_config"]["split_fingerprint"]
        == checkpoint_metadata["training_config"]["split_fingerprint"],
        "Training split fingerprint mismatch in checkpoint metadata.",
    )

    train_indices = checkpoint["split_info"]["train_indices"]
    validation_indices = checkpoint["split_info"]["val_indices"]
    require(len(train_indices) == training_count,
            "Checkpoint train index count mismatch.")
    require(len(validation_indices) == validation_count,
            "Checkpoint validation count mismatch.")
    require(
        set(train_indices).isdisjoint(validation_indices),
        "Checkpoint train/validation indices overlap.",
    )
    require(
        set(train_indices) | set(validation_indices)
        == set(range(len(features["gallery_image_paths"]))),
        "Checkpoint split is not a complete corrected gallery partition.",
    )
    gallery_paths_ordered = features["gallery_image_paths"]
    require(
        {gallery_paths_ordered[index] for index in train_indices} == train_paths,
        "Checkpoint training indices do not map to training paths.",
    )
    require(
        {gallery_paths_ordered[index] for index in validation_indices}
        == validation_paths,
        "Checkpoint validation indices do not map to validation paths.",
    )

    gallery_color = features["gallery_color_features"]
    gallery_texture = features["gallery_texture_features"]
    normalization = features["normalization_parameters"]
    require(
        np.allclose(np.min(gallery_color, axis=0), normalization["color_min"]),
        "Color scaler minima do not match corrected gallery.",
    )
    require(
        np.allclose(np.max(gallery_color, axis=0), normalization["color_max"]),
        "Color scaler maxima do not match corrected gallery.",
    )
    require(
        np.allclose(np.min(gallery_texture, axis=0), normalization["texture_min"]),
        "Texture scaler minima do not match corrected gallery.",
    )
    require(
        np.allclose(np.max(gallery_texture, axis=0), normalization["texture_max"]),
        "Texture scaler maxima do not match corrected gallery.",
    )

    with np.load(run_dir / "rankings.npz", allow_pickle=False) as rankings:
        gallery_from_rankings = rankings["gallery_paths"].tolist()
        queries_from_rankings = rankings["query_paths"].tolist()
        model_rankings = {
            name: rankings[name]
            for name in ("M1", "M2", "M3", "M4", "M5", "M6")
        }
    with (run_dir / "per_query_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        metric_rows = list(csv.DictReader(source))
    with (run_dir / "per_query_weights.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        weight_rows = list(csv.DictReader(source))
    gallery_labels = [Path(path).parent.name for path in gallery_from_rankings]
    query_labels = [Path(path).parent.name for path in queries_from_rankings]
    require(len(gallery_from_rankings) == gallery_count,
            "Ranking gallery size mismatch.")
    require(len(queries_from_rankings) == query_count,
            "Ranking query size mismatch.")
    require(
        queries_from_rankings == features["query_image_paths"],
        "Ranking query ordering mismatch.",
    )
    require(
        gallery_from_rankings == features["gallery_image_paths"],
        "Ranking gallery ordering mismatch.",
    )
    require(len(metric_rows) == 6 * query_count,
            "Unexpected number of per-query metric rows.")
    require(len(weight_rows) == query_count,
            "Per-query weight count mismatch.")
    reproduced = {}
    maximum_error = 0.0
    for model_name, rankings in model_rankings.items():
        require(
            rankings.shape == (query_count, gallery_count),
            f"{model_name} ranking shape mismatch.",
        )
        rows = [row for row in metric_rows if row["model"] == model_name]
        require(len(rows) == 100, f"{model_name} metric row count mismatch.")
        values = []
        for index, ranking in enumerate(rankings):
            metric = evaluate_retrieval(
                ranking, gallery_labels, query_labels[index], k=10
            )
            row = rows[index]
            for key, actual in (
                ("average_precision", metric["ap"]),
                ("precision_at_10", metric["precision_k"]),
                ("recall_at_10", metric["recall_k"]),
                ("f1_at_10", metric["f1_k"]),
            ):
                error = abs(float(row[key]) - actual)
                maximum_error = max(maximum_error, error)
                if error > 1e-12:
                    errors.append(
                        f"{model_name} query {index} {key}: error {error}"
                    )
            values.append(metric)
        reproduced[model_name] = {
            "query_count": len(values),
            "mAP": float(np.mean([item["ap"] for item in values])),
            "precision_at_10": float(np.mean([
                item["precision_k"] for item in values
            ])),
            "recall_at_10": float(np.mean([
                item["recall_k"] for item in values
            ])),
            "f1_at_10": float(np.mean([item["f1_k"] for item in values])),
        }
        saved = summary["aggregate_metrics"][model_name]
        for metric_name, saved_key in (
            ("mAP", "mean_average_precision"),
            ("precision_at_10", "mean_precision_at_10"),
            ("recall_at_10", "mean_recall_at_10"),
            ("f1_at_10", "mean_f1_at_10"),
        ):
            if abs(reproduced[model_name][metric_name] - saved[saved_key]) > 1e-12:
                errors.append(
                    f"{model_name} aggregate {metric_name} differs from results.json."
                )

    weights = np.asarray([
        [float(row["m5_color_weight"]), float(row["m5_texture_weight"])]
        for row in weight_rows
    ])
    require(np.allclose(weights.sum(axis=1), 1.0, atol=1e-6, rtol=0),
            "M5 predicted weights are not normalized.")
    mean_weights = weights.mean(axis=0)
    require(
        np.allclose(mean_weights, summary["m6_mean_weights"], atol=1e-12, rtol=0),
        "M6 weights do not equal newly predicted M5 query-weight mean.",
    )

    hashed = manifest.get("artifact_sha256", {})
    for name, expected in hashed.items():
        path = run_dir / name
        if not path.is_file():
            errors.append(f"Manifest artifact is missing: {name}")
        elif _sha256(path) != expected:
            errors.append(f"Artifact hash mismatch: {name}")

    validation = {
        "status": "passed" if not errors else "failed",
        "query_count": len(query_paths),
        "gallery_count": len(corrected_gallery_paths),
        "training_count": len(train_paths),
        "validation_count": len(validation_paths),
        "models_checked": list(model_rankings),
        "per_query_metric_rows_checked": len(metric_rows),
        "maximum_metric_absolute_error": maximum_error,
        "aggregate_metrics_recomputed": reproduced,
        "m6_weights_match_new_m5_mean": bool(
            np.allclose(mean_weights, summary["m6_mean_weights"], atol=1e-12)
        ),
        "all_manifest_artifact_hashes_verified": not any(
            error.startswith("Artifact hash mismatch")
            or error.startswith("Manifest artifact is missing")
            for error in errors
        ),
        "errors": errors,
    }
    (run_dir / "independent_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    if errors:
        raise AssertionError("; ".join(errors))
    return validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = validate_run(args.run_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
