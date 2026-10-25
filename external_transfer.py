"""Frozen-checkpoint Oxford-IIIT Pet transfer evaluation for Corel-1K CBIR."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

import config
from external_dataset import (
    dataset_fingerprint,
    detect_duplicates,
    read_oxford_pet_splits,
    _perceptual_hash,
)
from features.feature_extractor import extract_all_features
from model.mlp_model import AdaptiveWeightMLP
from retrieval.evaluation import evaluate_retrieval
from retrieval.retrieval import validate_fusion_weights, weighted_euclidean_distances
from statistical_evaluation import (
    holm_adjust,
    paired_permutation_test,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path(
    r"C:\Users\777Pr\Downloads\Oxford-IIIT-Pet\dataset"
)
DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "output" / "external_transfer_oxford_pet"
DEFAULT_DUPLICATE_REVIEW = (
    DEFAULT_RUN_ROOT / "oxford_pet_duplicate_review_v1.json"
)
COREL_RUN_ROOT = (
    REPOSITORY_ROOT / "output" / "statistical_evaluation_v1"
    / "run_20261024T193844Z"
)
DEFAULT_CHECKPOINT = COREL_RUN_ROOT / "seed_42" / "m5_checkpoint.pt"
DEFAULT_SCALER = COREL_RUN_ROOT / "seed_42" / "seed_features.pkl"
DEFAULT_COREL_WEIGHTS = COREL_RUN_ROOT / "per_query_weights.csv"

FEATURE_NAMES = (
    "h_mean", "h_std", "h_skew",
    "s_mean", "s_std", "s_skew",
    "v_mean", "v_std", "v_skew",
    "glcm_contrast", "glcm_correlation", "glcm_energy", "glcm_homogeneity",
)
BOOTSTRAP_REPLICATES = 10000
PERMUTATION_REPLICATES = 100000
ALPHA = 0.05
NEAR_DUPLICATE_DISTANCE = 4


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state():
    def git(*arguments):
        result = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "head_commit": git("rev-parse", "HEAD"),
        "worktree_status": git("status", "--short"),
    }


def load_frozen_model(checkpoint_path, expected_input_dim=config.TOTAL_FEATURE_DIM):
    """Load and validate the Corel-trained MLP without entering training code."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a metadata dictionary.")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint has no model_state_dict.")
    if tuple(state.get("fc1.weight", torch.empty(0)).shape) != (16, expected_input_dim):
        raise ValueError(
            "Frozen checkpoint input dimension does not match extracted features."
        )
    if tuple(state.get("fc3.weight", torch.empty(0)).shape) != (2, 8):
        raise ValueError("Frozen checkpoint output architecture is incompatible.")
    model = AdaptiveWeightMLP(input_dim=expected_input_dim)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def normalize_with_corel_bounds(color_features, texture_features, parameters):
    """Apply the original Corel gallery bounds without clipping external values."""
    color_min = np.asarray(parameters["color_min"], dtype=np.float64)
    color_max = np.asarray(parameters["color_max"], dtype=np.float64)
    texture_min = np.asarray(parameters["texture_min"], dtype=np.float64)
    texture_max = np.asarray(parameters["texture_max"], dtype=np.float64)
    if color_min.shape != (9,) or color_max.shape != (9,):
        raise ValueError("Corel color scaler must have nine feature bounds.")
    if texture_min.shape != (4,) or texture_max.shape != (4,):
        raise ValueError("Corel texture scaler must have four feature bounds.")
    bounds = (color_min, color_max, texture_min, texture_max)
    if any(not np.all(np.isfinite(values)) for values in bounds):
        raise ValueError("Corel scaler bounds must be finite.")
    if np.any(color_max < color_min) or np.any(texture_max < texture_min):
        raise ValueError("Corel scaler maximum bounds cannot be below minima.")
    color_range = np.where(color_max - color_min < 1e-8, 1.0, color_max - color_min)
    texture_range = np.where(
        texture_max - texture_min < 1e-8, 1.0, texture_max - texture_min
    )
    return (
        (color_features - color_min) / color_range,
        (texture_features - texture_min) / texture_range,
    )


def _load_corel_control_weights(weights_path, scaler_artifact):
    """Derive M6 only from the saved seed-42 held-out Corel predictions."""
    with Path(weights_path).open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row.get("seed") == "42"]
    expected_paths = set(scaler_artifact["query_image_paths"])
    if len(rows) != len(expected_paths) or len(rows) != 100:
        raise ValueError("Expected exactly 100 seed-42 Corel query weight records.")
    if {row["query_path"] for row in rows} != expected_paths:
        raise ValueError("Saved M5 weights do not match the seed-42 held-out queries.")
    predicted = np.asarray([
        [float(row["m5_color_weight"]), float(row["m5_texture_weight"])]
        for row in rows
    ], dtype=np.float64)
    validate_fusion_weights(predicted)
    mean_weights = validate_fusion_weights(np.mean(predicted, axis=0))
    saved_control = np.asarray([
        [float(row["m6_color_weight"]), float(row["m6_texture_weight"])]
        for row in rows
    ], dtype=np.float64)
    if not np.allclose(saved_control, mean_weights, atol=2e-6, rtol=0):
        raise ValueError("Saved M6 controls disagree with the mean Corel M5 weights.")
    return mean_weights


def _extract_raw_features(records):
    colors = np.empty((len(records), 9), dtype=np.float32)
    textures = np.empty((len(records), 4), dtype=np.float32)
    for index, record in enumerate(records):
        color, texture = extract_all_features(record.image_path)
        if color.shape != (9,) or texture.shape != (4,):
            raise ValueError(f"Unexpected feature shape for {record.relative_path}.")
        if not np.all(np.isfinite(color)) or not np.all(np.isfinite(texture)):
            raise ValueError(f"Non-finite features for {record.relative_path}.")
        colors[index] = color
        textures[index] = texture
        if (index + 1) % 250 == 0 or index + 1 == len(records):
            print(f"Extracted features for {index + 1}/{len(records)} images.")
    return colors, textures


def _corel_duplicate_candidates(records, corel_root):
    corel_images = sorted(
        path for path in Path(corel_root).rglob("*")
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )
    if not corel_images:
        raise FileNotFoundError(f"No Corel images found under {corel_root}.")
    corel_hashes = {}
    corel_phashes = {}
    for index, path in enumerate(corel_images):
        corel_hashes[path] = _sha256(path)
        corel_phashes[path] = _perceptual_hash(path)
        if (index + 1) % 250 == 0 or index + 1 == len(corel_images):
            print(f"Fingerprinting Corel images {index + 1}/{len(corel_images)}.")

    external_hashes = {}
    exact_pairs = []
    near_pairs = []
    corel_by_digest = {}
    for path, digest in corel_hashes.items():
        corel_by_digest.setdefault(digest, []).append(path)
    for index, record in enumerate(records):
        digest = _sha256(record.image_path)
        external_hashes[record.relative_path] = digest
        for corel_path in corel_by_digest.get(digest, ()):
            exact_pairs.append({
                "external_path": record.relative_path,
                "corel_path": corel_path.relative_to(corel_root).as_posix(),
                "sha256": digest,
            })
        external_hash = _perceptual_hash(record.image_path)
        for corel_path in corel_images:
            if digest == corel_hashes[corel_path]:
                continue
            distance = (
                external_hash ^ corel_phashes[corel_path]
            ).bit_count()
            if distance <= NEAR_DUPLICATE_DISTANCE:
                near_pairs.append({
                    "external_path": record.relative_path,
                    "corel_path": corel_path.relative_to(corel_root).as_posix(),
                    "hamming_distance": distance,
                })
        if (index + 1) % 250 == 0 or index + 1 == len(records):
            print(f"Comparing external images with Corel {index + 1}/{len(records)}.")
    return {
        "corel_image_count": len(corel_images),
        "exact_cross_dataset_pairs": exact_pairs,
        "near_duplicate_candidates_cross_dataset": near_pairs,
        "near_hash_threshold": NEAR_DUPLICATE_DISTANCE,
    }


def _validate_duplicate_review(review_path, duplicate_report, fingerprint):
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if review.get("dataset_fingerprint_sha256") != fingerprint:
        raise ValueError("Duplicate review manifest is for a different dataset.")
    candidate_pairs = {
        (row["gallery_path"], row["query_path"], row["hamming_distance"])
        for row in duplicate_report["near_duplicate_candidates_cross_split"]
    }
    reviewed_pairs = {
        (row["gallery_path"], row["query_path"], row["hamming_distance"])
        for row in review.get("pairs", [])
    }
    if reviewed_pairs != candidate_pairs:
        raise ValueError(
            "Duplicate review does not cover exactly the current cross-split candidates."
        )
    excluded = set(review.get("excluded_gallery_paths", []))
    candidate_gallery_paths = {pair[0] for pair in candidate_pairs}
    if excluded != candidate_gallery_paths:
        raise ValueError(
            "Reviewed exclusions must contain each candidate gallery image exactly once."
        )
    return review, excluded


def _load_verified_feature_cache(cache_run_dir, fingerprint, records,
                                 checkpoint_path, scaler_path):
    """Reuse raw features only when dataset and extraction provenance match."""
    cache_run_dir = Path(cache_run_dir).resolve()
    manifest_path = cache_run_dir / "reproducibility_manifest.json"
    feature_path = cache_run_dir / "features.npz"
    if not manifest_path.is_file() or not feature_path.is_file():
        raise FileNotFoundError("Feature cache run lacks its manifest or NPZ data.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset", {}).get("fingerprint_sha256") != fingerprint:
        raise ValueError("Feature cache dataset fingerprint does not match.")
    if manifest.get("output_files", {}).get("features.npz") != _sha256(feature_path):
        raise ValueError("Feature cache hash does not match its manifest.")
    expected_artifact_hashes = {
        "checkpoint_sha256": _sha256(checkpoint_path),
        "scaler_artifact_sha256": _sha256(scaler_path),
    }
    for key, expected in expected_artifact_hashes.items():
        if manifest.get("corel_transfer_artifacts", {}).get(key) != expected:
            raise ValueError(f"Feature cache {key} does not match current artifacts.")
    source_hashes = manifest.get("source", {}).get("files_sha256", {})
    for path in (
        REPOSITORY_ROOT / "features" / "feature_extractor.py",
        REPOSITORY_ROOT / "config.py",
    ):
        key = str(path.relative_to(REPOSITORY_ROOT))
        if source_hashes.get(key) != _sha256(path):
            raise ValueError(f"Feature cache source hash changed: {key}.")
    with np.load(feature_path, allow_pickle=False) as cached:
        expected_paths = np.asarray([record.relative_path for record in records])
        relative_paths = cached["relative_paths"]
        raw_color = cached["raw_color"]
        raw_texture = cached["raw_texture"]
    if not np.array_equal(relative_paths, expected_paths):
        raise ValueError("Feature cache image ordering does not match the split manifest.")
    if raw_color.shape != (len(records), 9) or raw_texture.shape != (len(records), 4):
        raise ValueError("Feature cache dimensions do not match the current dataset.")
    if not np.all(np.isfinite(raw_color)) or not np.all(np.isfinite(raw_texture)):
        raise ValueError("Feature cache contains non-finite values.")
    return raw_color.astype(np.float32), raw_texture.astype(np.float32)


def _compatibility_summary(raw_color, raw_texture, normalized_color, normalized_texture, parameters):
    normalized = np.concatenate([normalized_color, normalized_texture], axis=1)
    bounds_min = np.concatenate([
        np.asarray(parameters["color_min"], dtype=float),
        np.asarray(parameters["texture_min"], dtype=float),
    ])
    bounds_max = np.concatenate([
        np.asarray(parameters["color_max"], dtype=float),
        np.asarray(parameters["texture_max"], dtype=float),
    ])
    rows = []
    for index, name in enumerate(FEATURE_NAMES):
        values = normalized[:, index]
        rows.append({
            "feature": name,
            "corel_min": float(bounds_min[index]),
            "corel_max": float(bounds_max[index]),
            "external_raw_min": float(np.min(
                raw_color[:, index] if index < 9 else raw_texture[:, index - 9]
            )),
            "external_raw_max": float(np.max(
                raw_color[:, index] if index < 9 else raw_texture[:, index - 9]
            )),
            "normalized_min": float(np.min(values)),
            "normalized_max": float(np.max(values)),
            "normalized_mean": float(np.mean(values)),
            "normalized_std": float(np.std(values)),
            "fraction_below_corel_min": float(np.mean(values < 0)),
            "fraction_above_corel_max": float(np.mean(values > 1)),
        })
    return rows


def _summarize(metrics):
    summary = {}
    for model in ("M5", "M6", "M7"):
        model_rows = [row for row in metrics if row["model"] == model]
        summary[model] = {
            "query_count": len(model_rows),
            "mAP": float(np.mean([row["average_precision"] for row in model_rows])),
            "precision_at_10": float(np.mean([row["precision_at_10"] for row in model_rows])),
            "recall_at_10": float(np.mean([row["recall_at_10"] for row in model_rows])),
            "f1_at_10": float(np.mean([row["f1_at_10"] for row in model_rows])),
            "mean_ranking_time_seconds": float(np.mean([
                row["ranking_time_seconds"] for row in model_rows
            ])),
        }
    return summary


def _stratified_bootstrap_ci(differences, categories, replicates, seed):
    """Bootstrap paired query differences within each breed at its observed size."""
    differences = np.asarray(differences, dtype=float)
    categories = np.asarray(categories)
    strata = [
        differences[categories == category]
        for category in np.unique(categories)
    ]
    if not strata or any(values.size == 0 for values in strata):
        raise ValueError("Each bootstrap breed stratum must contain query pairs.")
    rng = np.random.default_rng(seed)
    estimates = np.zeros(replicates, dtype=float)
    total_queries = len(differences)
    for values in strata:
        samples = rng.choice(
            values, size=(replicates, len(values)), replace=True
        )
        estimates += samples.mean(axis=1) * (len(values) / total_queries)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return [float(low), float(high)]


def _paired_analysis(metrics, query_categories):
    results = {}
    comparisons = (("M5", "M6", "primary"), ("M5", "M7", "secondary"))
    raw_secondary = []
    for left, right, role in comparisons:
        left_rows = [row for row in metrics if row["model"] == left]
        right_rows = [row for row in metrics if row["model"] == right]
        left_rows.sort(key=lambda row: row["query_path"])
        right_rows.sort(key=lambda row: row["query_path"])
        if [row["query_path"] for row in left_rows] != [
            row["query_path"] for row in right_rows
        ]:
            raise ValueError(f"{left}/{right} query IDs do not align.")
        differences = np.asarray([
            a["average_precision"] - b["average_precision"]
            for a, b in zip(left_rows, right_rows)
        ], dtype=float)
        categories = np.asarray([
            row["category"] for row in left_rows
        ])
        ci = _stratified_bootstrap_ci(
            differences, categories, BOOTSTRAP_REPLICATES, seed=42
        )
        permutation_p = paired_permutation_test(
            differences, PERMUTATION_REPLICATES, seed=31415
        )
        try:
            wilcoxon_result = wilcoxon(
                differences, zero_method="wilcox", alternative="two-sided",
                method="auto",
            )
            wilcoxon_p = float(wilcoxon_result.pvalue)
            wilcoxon_statistic = float(wilcoxon_result.statistic)
        except ValueError:
            wilcoxon_p = 1.0
            wilcoxon_statistic = 0.0
        results[f"{left}_vs_{right}"] = {
            "role": role,
            "n_paired_queries": len(differences),
            "mean_difference_m5_minus_control": float(np.mean(differences)),
            "median_difference_m5_minus_control": float(np.median(differences)),
            "standard_deviation_of_differences": float(np.std(differences, ddof=1)),
            "paired_cohen_dz": (
                float(np.mean(differences) / np.std(differences, ddof=1))
                if np.std(differences, ddof=1) > 0 else None
            ),
            "positive_negative_tied_queries": {
                "positive": int(np.count_nonzero(differences > 0)),
                "negative": int(np.count_nonzero(differences < 0)),
                "tied": int(np.count_nonzero(differences == 0)),
            },
            "paired_bootstrap_95_percent_ci_mean_difference": ci,
            "paired_sign_flip_permutation_p_raw": float(permutation_p),
            "wilcoxon_signed_rank_statistic": wilcoxon_statistic,
            "wilcoxon_p_raw": wilcoxon_p,
        }
        if role == "secondary":
            raw_secondary.append(permutation_p)
    corrected = holm_adjust(raw_secondary)
    results["M5_vs_M7"]["paired_sign_flip_permutation_p_holm_secondary"] = corrected[0]
    return results


def _category_tables(metrics, run_dir):
    categories = sorted({row["category"] for row in metrics})
    summary_rows = []
    comparison_rows = []
    for category in categories:
        category_rows = [row for row in metrics if row["category"] == category]
        for model in ("M5", "M6", "M7"):
            model_rows = [row for row in category_rows if row["model"] == model]
            summary_rows.append({
                "category": category,
                "model": model,
                "query_count": len(model_rows),
                "mAP": float(np.mean([row["average_precision"] for row in model_rows])),
                "precision_at_10": float(np.mean([row["precision_at_10"] for row in model_rows])),
                "recall_at_10": float(np.mean([row["recall_at_10"] for row in model_rows])),
                "f1_at_10": float(np.mean([row["f1_at_10"] for row in model_rows])),
            })
        for control in ("M6", "M7"):
            m5_rows = {
                row["query_path"]: row
                for row in category_rows if row["model"] == "M5"
            }
            control_rows = {
                row["query_path"]: row
                for row in category_rows if row["model"] == control
            }
            paths = sorted(m5_rows)
            differences = np.asarray([
                m5_rows[path]["average_precision"]
                - control_rows[path]["average_precision"]
                for path in paths
            ])
            rng = np.random.default_rng(42)
            bootstrap = np.empty(10000)
            for index in range(len(bootstrap)):
                bootstrap[index] = np.mean(
                    rng.choice(differences, size=len(differences), replace=True)
                )
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            comparison_rows.append({
                "category": category,
                "comparison": f"M5-M{control[-1]}",
                "query_count": len(paths),
                "mean_paired_ap_difference": float(np.mean(differences)),
                "median_paired_ap_difference": float(np.median(differences)),
                "bootstrap_95_percent_ci_low": float(low),
                "bootstrap_95_percent_ci_high": float(high),
                "positive_queries": int(np.count_nonzero(differences > 0)),
                "negative_queries": int(np.count_nonzero(differences < 0)),
                "tied_queries": int(np.count_nonzero(differences == 0)),
            })
    _write_csv(run_dir / "category_metrics.csv", summary_rows)
    _write_csv(run_dir / "category_paired_differences.csv", comparison_rows)
    return summary_rows, comparison_rows


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(run_dir, dataset_fingerprint_value, record_counts, summary,
                  paired, mean_weights, weight_summary, compatibility,
                  duplicates, cross_duplicates, source_revision,
                  duplicate_review, feature_source):
    primary = paired["M5_vs_M6"]
    significant = (
        primary["paired_sign_flip_permutation_p_raw"] < ALPHA
        and primary["paired_bootstrap_95_percent_ci_mean_difference"][0] > 0
    )
    lines = [
        "# Frozen Corel-to-Oxford-IIIT Pet Transfer Evaluation",
        "",
        "## Dataset and protocol",
        "",
        "- Dataset: Oxford-IIIT Pet, 37 breeds; official trainval split used as gallery and official test split used as queries.",
        "- Source: https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "- License: dataset page states CC BY-SA 4.0; copyright remains with original image owners. Respect the original source-site terms documented by the dataset.",
        "- Relevance: same breed label, parsed from official class IDs/image stems; every query breed is represented in the gallery.",
        f"- Gallery images used: {record_counts['gallery']}; official trainval images before duplicate removal: {record_counts['official_gallery']}; queries: {record_counts['query']}; total: {record_counts['total']}; breeds: {record_counts['categories']}.",
        f"- Split is the published trainval/test split (not randomly re-split). The {len(duplicate_review['excluded_gallery_paths'])} visually confirmed trainval/test duplicate gallery files were excluded; all {record_counts['query']} official test queries were retained. No model training, fine-tuning, or hyperparameter selection was performed.",
        f"- Dataset fingerprint: `{dataset_fingerprint_value}`.",
        "",
        "## Frozen model and compatibility",
        "",
        "- M5: frozen Corel seed-42 checkpoint; 13 input features, 16/8 hidden layers, 2 Softmax outputs.",
        "- Feature order: 9 HSV moments (H/S/V mean, standard deviation, skewness), then GLCM contrast, correlation, energy, homogeneity.",
        "- External raw features were transformed with the exact Corel seed-42 gallery min/max bounds. Values outside [0, 1] were not clipped.",
        f"- M6 fixed weights are the mean of the 100 saved Corel seed-42 M5 query predictions: colour `{mean_weights[0]:.10f}`, texture `{mean_weights[1]:.10f}`.",
        "- M7 fixed weights: colour 0.5, texture 0.5. All models use the same M5/M6/M7 distance-level formula and gallery/query pairs.",
        f"- Exact duplicate pairs within Oxford across gallery/query: {len(duplicates['exact_cross_split_pairs'])}.",
        f"- Exact duplicate pairs between Oxford and configured Corel-1K: {len(cross_duplicates['exact_cross_dataset_pairs'])}.",
        f"- Perceptual-hash candidates across Oxford splits: {len(duplicates['near_duplicate_candidates_cross_split'])}; all {len(duplicate_review['pairs'])} were visually confirmed and their gallery copies excluded. Cross-dataset candidates: {len(cross_duplicates['near_duplicate_candidates_cross_dataset'])}.",
        f"- Feature source: {feature_source}. Cache reuse is allowed only after dataset, checkpoint, scaler, extraction-source, ordering, shape and content-hash checks.",
        "- Retrieval time measures ranking and sorting only; feature extraction and one-time M5 inference are excluded equally from the timed ranking stage.",
        "",
        "## Aggregate retrieval results",
        "",
        "| Model | mAP | Precision@10 | Recall@10 | F1@10 | Mean ranking ms/query |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("M5", "M6", "M7"):
        row = summary[model]
        lines.append(
            f"| {model} | {row['mAP']:.6f} | {row['precision_at_10']:.6f} | "
            f"{row['recall_at_10']:.6f} | {row['f1_at_10']:.6f} | "
            f"{1000*row['mean_ranking_time_seconds']:.3f} |"
        )
    lines += [
        "",
        "## Prespecified statistical comparison",
        "",
        f"- Primary comparison: paired per-query AP difference M5−M6, n={primary['n_paired_queries']}.",
        f"- Mean paired difference: {primary['mean_difference_m5_minus_control']:.6f}; median: {primary['median_difference_m5_minus_control']:.6f}; paired Cohen's dz: {primary['paired_cohen_dz']}.",
        f"- Positive/negative/tied queries: {primary['positive_negative_tied_queries']['positive']}/{primary['positive_negative_tied_queries']['negative']}/{primary['positive_negative_tied_queries']['tied']}.",
        f"- Category-stratified paired bootstrap 95% CI for mean difference: {primary['paired_bootstrap_95_percent_ci_mean_difference']}.",
        f"- Two-sided paired sign-flip permutation p: {primary['paired_sign_flip_permutation_p_raw']:.6g}; Wilcoxon signed-rank p: {primary['wilcoxon_p_raw']:.6g}.",
        "- The sign-flip test tests a sharp paired null under within-query exchangeability (and is not an exact test of a weak zero-mean null absent symmetry). The bootstrap resamples paired query differences within each breed while keeping the observed breed sample sizes fixed. The Wilcoxon test uses nonzero paired differences and tests a symmetric location shift; it is secondary and can have limited power under skewness/ties.",
        f"- Secondary comparison M5−M7 sign-flip p: {paired['M5_vs_M7']['paired_sign_flip_permutation_p_raw']:.6g}; Holm-adjusted across the predefined secondary comparison set: {paired['M5_vs_M7']['paired_sign_flip_permutation_p_holm_secondary']:.6g}.",
        f"- {'The primary comparison supports higher M5 mean AP over M6 under the stated paired analysis.' if significant else 'The primary analysis does not provide sufficient evidence to claim M5 improves over M6.'}",
        "- Inference is conditional on this single decontaminated official gallery/query split and the frozen Corel checkpoint; it does not establish generalization across independent dataset samples or training seeds.",
        "",
        "## M5 predicted weights and input distribution",
        "",
        f"- Colour weight mean/SD/min/max: {weight_summary['color_mean']:.6f}/{weight_summary['color_std']:.6f}/{weight_summary['color_min']:.6f}/{weight_summary['color_max']:.6f}.",
        f"- Texture weight mean/SD/min/max: {weight_summary['texture_mean']:.6f}/{weight_summary['texture_std']:.6f}/{weight_summary['texture_min']:.6f}/{weight_summary['texture_max']:.6f}.",
        f"- Fraction of all normalized feature values below 0: {compatibility['fraction_values_below_zero']:.4%}; above 1: {compatibility['fraction_values_above_one']:.4%}.",
        "- Feature-wise raw and normalized ranges are saved in `feature_compatibility.csv`; out-of-range values indicate transfer distribution shift and were passed to the frozen model without clipping.",
        "",
        "## Reproducibility and limitations",
        "",
        f"- Source revision: `{source_revision['head_commit']}`; worktree dirty at execution: `{bool(source_revision['worktree_status'])}`.",
        f"- Bootstrap replicates: {BOOTSTRAP_REPLICATES}; permutation replicates: {PERMUTATION_REPLICATES}; seeds: bootstrap 42, permutation 31415.",
        "- No evaluation metrics or rankings were reused. Raw feature reuse, if used, was validated against the dataset fingerprint, extraction source, checkpoint, scaler, array shape/order, and cache hash. Checkpoint, scaler, source, annotation, and dataset hashes are in `reproducibility_manifest.json`.",
        "- Oxford-IIIT Pet is a breed-classification retrieval benchmark, not a broad semantic image retrieval benchmark. Same-breed relevance does not imply fine-grained visual similarity for every image.",
        "- The official dataset notes images are copyrighted by original owners and asks users to respect the source websites' terms. Dataset use rights should be checked for the intended paper distribution.",
        "",
        "## Output files",
        "",
        "- `per_query_metrics.csv`: AP, Precision@10, Recall@10, F1@10, and ranking time for every model/query.",
        "- `per_query_weights.csv`: query-specific M5 weights and fixed M6/M7 weights.",
        "- `category_metrics.csv`, `category_paired_differences.csv`: category-level results and paired uncertainty.",
        "- `feature_compatibility.csv`, `split_manifest.csv`, duplicate audit JSON, `features.npz`, `rankings.npz`, `results.json`, and `reproducibility_manifest.json`.",
        "",
    ]
    (run_dir / "transfer_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_transfer(dataset_root=DEFAULT_DATASET_ROOT, output_root=DEFAULT_RUN_ROOT,
                 checkpoint_path=DEFAULT_CHECKPOINT, scaler_path=DEFAULT_SCALER,
                 corel_weights_path=DEFAULT_COREL_WEIGHTS,
                 corel_dataset_root=config.DATASET_PATH,
                 duplicate_review_path=DEFAULT_DUPLICATE_REVIEW,
                 reuse_feature_cache=None):
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    scaler_path = Path(scaler_path).resolve()
    corel_weights_path = Path(corel_weights_path).resolve()
    corel_dataset_root = Path(corel_dataset_root).resolve()
    duplicate_review_path = Path(duplicate_review_path).resolve()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    records = read_oxford_pet_splits(dataset_root)
    labels = sorted({record.label for record in records})
    if len(records) != 7349 or len(labels) != 37:
        raise ValueError("Unexpected Oxford-IIIT Pet dataset size or breed count.")
    official_gallery_records = [
        record for record in records if record.split == "gallery"
    ]
    query_records = [record for record in records if record.split == "query"]

    duplicate_report = detect_duplicates(records, NEAR_DUPLICATE_DISTANCE)
    cross_duplicate_report = _corel_duplicate_candidates(
        records, corel_dataset_root
    )
    (run_dir / "oxford_duplicate_audit.json").write_text(
        json.dumps(duplicate_report, indent=2), encoding="utf-8"
    )
    (run_dir / "corel_duplicate_audit.json").write_text(
        json.dumps(cross_duplicate_report, indent=2), encoding="utf-8"
    )
    if duplicate_report["exact_cross_split_pairs"]:
        raise ValueError(
            "Exact duplicate images cross the Oxford gallery/query split; refusing evaluation."
        )
    if cross_duplicate_report["exact_cross_dataset_pairs"]:
        raise ValueError(
            "Exact Oxford/Corel image overlap found; independent transfer is not valid."
        )

    annotation_sources = [
        dataset_root / "annotations" / name
        for name in ("trainval.txt", "test.txt", "list.txt")
    ]
    fingerprint = dataset_fingerprint(
        records, duplicate_report["content_hashes"], annotation_sources
    )
    duplicate_review, excluded_gallery_paths = _validate_duplicate_review(
        duplicate_review_path, duplicate_report, fingerprint
    )
    if not excluded_gallery_paths.issubset(
        {record.relative_path for record in official_gallery_records}
    ):
        raise ValueError("Duplicate review references a non-gallery image.")
    gallery_records = [
        record for record in official_gallery_records
        if record.relative_path not in excluded_gallery_paths
    ]
    if not gallery_records:
        raise ValueError("Duplicate review removed every gallery image.")
    if any(
        not any(record.label == query.label for record in gallery_records)
        for query in query_records
    ):
        raise ValueError("Duplicate removal left a query breed without gallery items.")
    remaining_candidates = [
        row for row in duplicate_report["near_duplicate_candidates_cross_split"]
        if row["gallery_path"] not in excluded_gallery_paths
    ]
    if remaining_candidates:
        raise ValueError("Reviewed gallery still contains near-duplicate candidates.")
    (run_dir / "duplicate_review_manifest.json").write_text(
        json.dumps(duplicate_review, indent=2), encoding="utf-8"
    )

    try:
        import pickle
        scaler_artifact = pickle.loads(scaler_path.read_bytes())
    except (OSError, pickle.UnpicklingError) as error:
        raise RuntimeError(f"Could not load trusted Corel scaler artifact: {error}") from error
    if not isinstance(scaler_artifact, dict):
        raise ValueError("Corel feature artifact must contain a dictionary.")
    model, checkpoint = load_frozen_model(checkpoint_path)
    if checkpoint.get("feature_fingerprint") != scaler_artifact.get("feature_fingerprint"):
        raise ValueError("Checkpoint and Corel scaler feature fingerprints differ.")
    mean_weights = _load_corel_control_weights(
        corel_weights_path, scaler_artifact
    )
    if reuse_feature_cache is not None:
        raw_color, raw_texture = _load_verified_feature_cache(
            reuse_feature_cache,
            fingerprint,
            records,
            checkpoint_path,
            scaler_path,
        )
        feature_source = f"validated raw feature cache {Path(reuse_feature_cache).resolve()}"
    else:
        raw_color, raw_texture = _extract_raw_features(records)
        feature_source = "fresh feature extraction"
    normalized_color, normalized_texture = normalize_with_corel_bounds(
        raw_color, raw_texture, scaler_artifact["normalization_parameters"]
    )
    if not np.all(np.isfinite(normalized_color)) or not np.all(
        np.isfinite(normalized_texture)
    ):
        raise ValueError("Normalized external features contain non-finite values.")

    record_index = {record.relative_path: index for index, record in enumerate(records)}
    gallery_indices = np.asarray([
        record_index[record.relative_path] for record in gallery_records
    ])
    query_indices = np.asarray([
        record_index[record.relative_path] for record in query_records
    ])
    gallery_color = normalized_color[gallery_indices]
    gallery_texture = normalized_texture[gallery_indices]
    query_color = normalized_color[query_indices]
    query_texture = normalized_texture[query_indices]
    gallery_labels = [record.label for record in gallery_records]
    query_labels = [record.label for record in query_records]
    query_paths = [record.relative_path for record in query_records]

    model_input = np.concatenate([query_color, query_texture], axis=1).astype(np.float32)
    with torch.inference_mode():
        predicted_weights = model(torch.from_numpy(model_input)).cpu().numpy()
    validate_fusion_weights(predicted_weights, atol=1e-6)
    fixed_weights = {
        "M6": mean_weights,
        "M7": np.asarray([0.5, 0.5], dtype=np.float64),
    }
    rankings = {name: [] for name in ("M5", "M6", "M7")}
    metrics = []
    weight_rows = []
    for query_index, (query_record, query_label) in enumerate(
        zip(query_records, query_labels)
    ):
        query_weights = {
            "M5": predicted_weights[query_index],
            "M6": fixed_weights["M6"],
            "M7": fixed_weights["M7"],
        }
        weight_rows.append({
            "query_path": query_record.relative_path,
            "category": query_label,
            "m5_color_weight": float(query_weights["M5"][0]),
            "m5_texture_weight": float(query_weights["M5"][1]),
            "m6_color_weight": float(query_weights["M6"][0]),
            "m6_texture_weight": float(query_weights["M6"][1]),
            "m7_color_weight": float(query_weights["M7"][0]),
            "m7_texture_weight": float(query_weights["M7"][1]),
        })
        for model_name in ("M5", "M6", "M7"):
            start = time.perf_counter()
            distances = weighted_euclidean_distances(
                query_color[query_index],
                gallery_color,
                query_texture[query_index],
                gallery_texture,
                query_weights[model_name],
            )
            ranking = np.argsort(distances, kind="stable")
            elapsed = time.perf_counter() - start
            rankings[model_name].append(ranking)
            query_metrics = evaluate_retrieval(
                ranking, gallery_labels, query_label, k=config.EVAL_K
            )
            metrics.append({
                "query_path": query_record.relative_path,
                "category": query_label,
                "model": model_name,
                "average_precision": float(query_metrics["ap"]),
                "precision_at_10": float(query_metrics["precision_k"]),
                "recall_at_10": float(query_metrics["recall_k"]),
                "f1_at_10": float(query_metrics["f1_k"]),
                "ranking_time_seconds": float(elapsed),
            })
        if (query_index + 1) % 250 == 0 or query_index + 1 == len(query_records):
            print(f"Ranked query {query_index + 1}/{len(query_records)}.")

    summary = _summarize(metrics)
    paired = _paired_analysis(metrics, query_labels)
    category_summary, category_differences = _category_tables(metrics, run_dir)
    _write_csv(run_dir / "per_query_metrics.csv", metrics)
    _write_csv(run_dir / "per_query_weights.csv", weight_rows)
    _write_csv(run_dir / "feature_compatibility.csv", _compatibility_summary(
        raw_color, raw_texture, normalized_color, normalized_texture,
        scaler_artifact["normalization_parameters"],
    ))
    manifest_rows = []
    for record in records:
        manifest_rows.append({
            "relative_path": record.relative_path,
            "identity": record.identity,
            "label": record.label,
            "split": record.split,
            "sha256": duplicate_report["content_hashes"][record.relative_path],
        })
    included_gallery_paths = {
        record.relative_path for record in gallery_records
    }
    for row in manifest_rows:
        row["included_in_gallery"] = (
            row["relative_path"] in included_gallery_paths
            if row["split"] == "gallery" else False
        )
        row["duplicate_excluded_gallery"] = (
            row["relative_path"] in excluded_gallery_paths
        )
    _write_csv(run_dir / "split_manifest.csv", manifest_rows)
    np.savez_compressed(
        run_dir / "features.npz",
        relative_paths=np.asarray([record.relative_path for record in records]),
        raw_color=raw_color,
        raw_texture=raw_texture,
        normalized_color=normalized_color,
        normalized_texture=normalized_texture,
    )
    np.savez_compressed(
        run_dir / "rankings.npz",
        query_paths=np.asarray(query_paths),
        gallery_paths=np.asarray([
            record.relative_path for record in gallery_records
        ]),
        **{
            model_name: np.asarray(model_rankings, dtype=np.int32)
            for model_name, model_rankings in rankings.items()
        },
    )
    normalized_all = np.concatenate([normalized_color, normalized_texture], axis=1)
    compatibility = {
        "fraction_values_below_zero": float(np.mean(normalized_all < 0)),
        "fraction_values_above_one": float(np.mean(normalized_all > 1)),
    }
    weights = np.asarray(predicted_weights, dtype=float)
    weight_summary = {
        "color_mean": float(np.mean(weights[:, 0])),
        "color_std": float(np.std(weights[:, 0], ddof=1)),
        "color_min": float(np.min(weights[:, 0])),
        "color_max": float(np.max(weights[:, 0])),
        "texture_mean": float(np.mean(weights[:, 1])),
        "texture_std": float(np.std(weights[:, 1], ddof=1)),
        "texture_min": float(np.min(weights[:, 1])),
        "texture_max": float(np.max(weights[:, 1])),
    }

    source_revision = _git_state()
    source_paths = [
        Path(__file__),
        REPOSITORY_ROOT / "external_dataset.py",
        REPOSITORY_ROOT / "features" / "feature_extractor.py",
        REPOSITORY_ROOT / "features" / "normalization.py",
        REPOSITORY_ROOT / "retrieval" / "retrieval.py",
        REPOSITORY_ROOT / "retrieval" / "evaluation.py",
        REPOSITORY_ROOT / "model" / "mlp_model.py",
        REPOSITORY_ROOT / "config.py",
        REPOSITORY_ROOT / "statistical_evaluation.py",
        duplicate_review_path,
    ]
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Oxford-IIIT Pet",
            "source_url": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
            "image_archive_url": "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
            "annotation_archive_url": "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz",
            "license": "CC BY-SA 4.0 according to dataset page; image copyrights remain with original owners",
            "dataset_root": str(dataset_root),
            "fingerprint_sha256": fingerprint,
            "image_count": len(records),
            "gallery_count": len(gallery_records),
            "official_gallery_count": len(official_gallery_records),
            "excluded_duplicate_gallery_count": len(excluded_gallery_paths),
            "excluded_duplicate_gallery_paths": sorted(excluded_gallery_paths),
            "query_count": len(query_records),
            "category_count": len(labels),
            "relevance_definition": "same breed",
            "split_source": "official trainval.txt as gallery and test.txt as query",
            "split_seed": None,
            "manual_duplicate_review_manifest": str(duplicate_review_path),
            "duplicate_review_sha256": _sha256(duplicate_review_path),
        },
        "experiment": {
            "models": ["M5", "M6", "M7"],
            "primary_comparison": "paired per-query AP: M5 minus M6",
            "secondary_comparison": "paired per-query AP: M5 minus M7",
            "alpha": ALPHA,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": 42,
            "permutation_replicates": PERMUTATION_REPLICATES,
            "permutation_seed": 31415,
            "distance_formula": "sqrt(w_color * squared_color_distance + w_texture * squared_texture_distance)",
            "evaluation_k": config.EVAL_K,
            "timing": "ranking and stable sort only; feature extraction and M5 inference excluded",
            "normalization": "frozen seed-42 Corel gallery min/max bounds; no clipping or refitting",
            "feature_order": FEATURE_NAMES,
            "glcm_distance": config.GLCM_DISTANCE,
            "glcm_angles_degrees": config.GLCM_ANGLES_DEG,
            "mlp_retrained": False,
            "external_query_labels_used_for_selection": False,
            "feature_source": feature_source,
            "evaluation_results_reused": False,
        },
        "corel_transfer_artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "scaler_artifact": str(scaler_path),
            "scaler_artifact_sha256": _sha256(scaler_path),
            "corel_query_weights": str(corel_weights_path),
            "corel_query_weights_sha256": _sha256(corel_weights_path),
            "frozen_m6_mean_weights": [float(x) for x in mean_weights],
        },
        "source": {
            **source_revision,
            "files_sha256": {
                str(path.relative_to(REPOSITORY_ROOT)): _sha256(path)
                for path in source_paths
            },
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "output_files": {},
    }
    (run_dir / "results.json").write_text(json.dumps({
        "summary": summary,
        "paired_tests": paired,
        "weight_summary": weight_summary,
        "feature_distribution_summary": compatibility,
        "protocol": {
            "official_gallery_count": len(official_gallery_records),
            "gallery_count_after_duplicate_removal": len(gallery_records),
            "excluded_duplicate_gallery_paths": sorted(excluded_gallery_paths),
            "query_count": len(query_records),
            "feature_source": feature_source,
        },
        "category_summary": category_summary,
        "category_paired_differences": category_differences,
    }, indent=2), encoding="utf-8")
    _write_report(
        run_dir, fingerprint,
        {
            "gallery": len(gallery_records),
            "official_gallery": len(official_gallery_records),
            "query": len(query_records),
            "total": len(records),
            "categories": len(labels),
        },
        summary,
        paired,
        mean_weights,
        weight_summary,
        compatibility,
        duplicate_report,
        cross_duplicate_report,
        source_revision,
        duplicate_review,
        feature_source,
    )
    for path in sorted(run_dir.iterdir()):
        if path.is_file():
            manifest["output_files"][path.name] = _sha256(path)
    (run_dir / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Completed frozen transfer evaluation: {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER)
    parser.add_argument("--corel-weights", type=Path, default=DEFAULT_COREL_WEIGHTS)
    parser.add_argument("--corel-dataset-root", type=Path, default=config.DATASET_PATH)
    parser.add_argument(
        "--duplicate-review", type=Path, default=DEFAULT_DUPLICATE_REVIEW
    )
    parser.add_argument(
        "--reuse-feature-cache", type=Path,
        help="Reuse raw features from a run directory after provenance verification.",
    )
    args = parser.parse_args()
    run_transfer(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        checkpoint_path=args.checkpoint,
        scaler_path=args.scaler,
        corel_weights_path=args.corel_weights,
        corel_dataset_root=args.corel_dataset_root,
        duplicate_review_path=args.duplicate_review,
        reuse_feature_cache=args.reuse_feature_cache,
    )


if __name__ == "__main__":
    main()
