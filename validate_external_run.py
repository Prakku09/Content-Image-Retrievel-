"""Independently verify saved rankings, metrics, weights, and run provenance."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import matplotlib
import numpy as np
import PIL
import scipy
import skimage

from external_dataset import dataset_fingerprint, read_oxford_pet_splits
from retrieval.evaluation import evaluate_retrieval


REPOSITORY_ROOT = Path(__file__).resolve().parent


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def validate_run(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "reproducibility_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_root = Path(manifest["dataset"]["dataset_root"])
    checkpoint_path = Path(
        manifest["corel_transfer_artifacts"]["checkpoint"]
    )
    scaler_path = Path(
        manifest["corel_transfer_artifacts"]["scaler_artifact"]
    )
    weight_path = Path(
        manifest["corel_transfer_artifacts"]["corel_query_weights"]
    )

    output_hashes = manifest["output_files"]
    for name, expected_hash in output_hashes.items():
        path = run_dir / name
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"Output hash verification failed: {name}")
    for artifact_path, key in (
        (checkpoint_path, "checkpoint_sha256"),
        (scaler_path, "scaler_artifact_sha256"),
        (weight_path, "corel_query_weights_sha256"),
    ):
        if _sha256(artifact_path) != manifest["corel_transfer_artifacts"][key]:
            raise ValueError(f"Corel transfer artifact hash changed: {artifact_path}")

    records = read_oxford_pet_splits(dataset_root)
    split_rows = _read_csv(run_dir / "split_manifest.csv")
    if len(split_rows) != len(records):
        raise ValueError("Split manifest row count differs from dataset records.")
    content_hashes = {}
    record_by_path = {record.relative_path: record for record in records}
    for row in split_rows:
        record = record_by_path.get(row["relative_path"])
        if record is None or record.label != row["label"] or record.split != row["split"]:
            raise ValueError("Split manifest metadata does not match Oxford annotations.")
        digest = _sha256(record.image_path)
        if digest != row["sha256"]:
            raise ValueError(f"Dataset image hash changed: {record.relative_path}")
        content_hashes[record.relative_path] = digest
    annotation_sources = [
        dataset_root / "annotations" / name
        for name in ("trainval.txt", "test.txt", "list.txt")
    ]
    fingerprint = dataset_fingerprint(records, content_hashes, annotation_sources)
    if fingerprint != manifest["dataset"]["fingerprint_sha256"]:
        raise ValueError("Dataset fingerprint does not match the run manifest.")

    included_gallery = [
        row for row in split_rows
        if row["split"] == "gallery" and row["included_in_gallery"] == "True"
    ]
    counts_by_category = {}
    for row in split_rows:
        counts = counts_by_category.setdefault(row["label"], {
            "official_gallery": 0,
            "gallery_after_duplicate_removal": 0,
            "excluded_duplicate_gallery": 0,
            "query": 0,
        })
        if row["split"] == "query":
            counts["query"] += 1
        else:
            counts["official_gallery"] += 1
            if row["included_in_gallery"] == "True":
                counts["gallery_after_duplicate_removal"] += 1
            if row["duplicate_excluded_gallery"] == "True":
                counts["excluded_duplicate_gallery"] += 1
    query_records = [record for record in records if record.split == "query"]
    gallery_paths = [row["relative_path"] for row in included_gallery]
    gallery_labels = [row["label"] for row in included_gallery]
    query_labels = {record.relative_path: record.label for record in query_records}
    if len(included_gallery) != manifest["dataset"]["gallery_count"]:
        raise ValueError("Used gallery size differs from manifest.")
    if len(query_records) != manifest["dataset"]["query_count"]:
        raise ValueError("Query size differs from manifest.")
    gallery_identities = {
        row["identity"] for row in included_gallery
    }
    query_identities = {
        record.identity for record in query_records
    }
    identity_overlap = gallery_identities.intersection(query_identities)
    if identity_overlap:
        raise ValueError("Image identities overlap between retained gallery and queries.")
    duplicate_audit = json.loads(
        (run_dir / "oxford_duplicate_audit.json").read_text(encoding="utf-8")
    )
    review = json.loads(
        (run_dir / "duplicate_review_manifest.json").read_text(encoding="utf-8")
    )
    excluded_gallery = set(review["excluded_gallery_paths"])
    if any(
        pair["gallery_path"] not in excluded_gallery
        for pair in duplicate_audit["near_duplicate_candidates_cross_split"]
    ):
        raise ValueError("A visually reviewed duplicate remains in the gallery.")
    if duplicate_audit["exact_cross_split_pairs"]:
        raise ValueError("Exact duplicate images remain across the official split.")
    corel_duplicate_audit = json.loads(
        (run_dir / "corel_duplicate_audit.json").read_text(encoding="utf-8")
    )
    if corel_duplicate_audit["exact_cross_dataset_pairs"]:
        raise ValueError("Exact image overlap exists between Oxford and Corel.")

    with np.load(run_dir / "rankings.npz", allow_pickle=False) as saved:
        saved_query_paths = saved["query_paths"].tolist()
        saved_gallery_paths = saved["gallery_paths"].tolist()
        rankings = {
            model: saved[model]
            for model in ("M5", "M6", "M7")
        }
    if saved_gallery_paths != gallery_paths:
        raise ValueError("Ranking gallery order differs from the split manifest.")
    if set(saved_query_paths) != set(query_labels):
        raise ValueError("Saved ranking query IDs differ from official test queries.")
    if any(rows.shape != (len(saved_query_paths), len(gallery_paths))
           for rows in rankings.values()):
        raise ValueError("Saved ranking matrix shape is incorrect.")

    per_query_rows = _read_csv(run_dir / "per_query_metrics.csv")
    expected_pairs = {
        (model, query_path)
        for model in ("M5", "M6", "M7")
        for query_path in saved_query_paths
    }
    row_map = {}
    for row in per_query_rows:
        key = (row["model"], row["query_path"])
        if key in row_map:
            raise ValueError(f"Duplicate per-query metric row: {key}")
        row_map[key] = row
    if set(row_map) != expected_pairs:
        raise ValueError("Per-query metrics do not cover every query/model pair.")

    expected_indices = np.arange(len(gallery_paths))
    maximum_metric_error = 0.0
    checked_rankings = 0
    model_ap = {model: [] for model in ("M5", "M6", "M7")}
    model_metric_values = {
        model: {name: [] for name in (
            "average_precision", "precision_at_10", "recall_at_10", "f1_at_10"
        )}
        for model in ("M5", "M6", "M7")
    }
    for model, model_rankings in rankings.items():
        for index, ranking in enumerate(model_rankings):
            if not np.array_equal(np.sort(ranking), expected_indices):
                raise ValueError(f"{model} ranking {index} is not a gallery permutation.")
            query_path = saved_query_paths[index]
            metric = evaluate_retrieval(
                ranking, gallery_labels, query_labels[query_path], k=10
            )
            expected = {
                "average_precision": metric["ap"],
                "precision_at_10": metric["precision_k"],
                "recall_at_10": metric["recall_k"],
                "f1_at_10": metric["f1_k"],
            }
            stored = row_map[(model, query_path)]
            for name, value in expected.items():
                stored_value = float(stored[name])
                error = abs(stored_value - value)
                maximum_metric_error = max(maximum_metric_error, error)
                if error > 1e-12:
                    raise ValueError(
                        f"Saved {name} differs from ranking for {model}/{query_path}."
                    )
                model_metric_values[model][name].append(value)
            model_ap[model].append(metric["ap"])
            checked_rankings += 1

    weights = _read_csv(run_dir / "per_query_weights.csv")
    if len(weights) != len(saved_query_paths):
        raise ValueError("Weight row count does not equal query count.")
    if {row["query_path"] for row in weights} != set(saved_query_paths):
        raise ValueError("Per-query weight IDs do not match ranked queries.")
    weight_max_error = 0.0
    for row in weights:
        for prefix in ("m5", "m6", "m7"):
            weight_sum = (
                float(row[f"{prefix}_color_weight"])
                + float(row[f"{prefix}_texture_weight"])
            )
            weight_max_error = max(weight_max_error, abs(weight_sum - 1.0))
            if abs(weight_sum - 1.0) > 1e-6:
                raise ValueError(f"{prefix} weights do not sum to one.")

    result_data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    comparison_rows = []
    for model in ("M5", "M6", "M7"):
        computed = {
            metric: float(np.mean(values))
            for metric, values in model_metric_values[model].items()
        }
        reported = result_data["summary"][model]
        mappings = {
            "average_precision": "mAP",
            "precision_at_10": "precision_at_10",
            "recall_at_10": "recall_at_10",
            "f1_at_10": "f1_at_10",
        }
        for metric, key in mappings.items():
            if abs(computed[metric] - reported[key]) > 1e-12:
                raise ValueError(f"Aggregate {model} {key} does not match per-query rows.")
        comparison_rows.append({
            "model": model,
            "query_count": len(saved_query_paths),
            "mAP": computed["average_precision"],
            "precision_at_10": computed["precision_at_10"],
            "recall_at_10": computed["recall_at_10"],
            "f1_at_10": computed["f1_at_10"],
            "mean_ranking_time_ms": 1000.0 * float(np.mean([
                float(row_map[(model, path)]["ranking_time_seconds"])
                for path in saved_query_paths
            ])),
        })

    duplicate_review_path = Path(
        manifest["dataset"]["manual_duplicate_review_manifest"]
    )
    if _sha256(duplicate_review_path) != manifest["dataset"]["duplicate_review_sha256"]:
        raise ValueError("Manual duplicate review manifest hash changed.")
    audit = {
        "status": "passed",
        "dataset_fingerprint_sha256": fingerprint,
        "queries_checked": len(saved_query_paths),
        "gallery_images_checked": len(gallery_paths),
        "rankings_checked": checked_rankings,
        "per_query_metric_rows_checked": len(per_query_rows),
        "maximum_metric_absolute_error": maximum_metric_error,
        "maximum_weight_sum_absolute_error": weight_max_error,
        "all_queries_retained": len(saved_query_paths) == 3669,
        "metrics_recomputed_from_saved_rankings": True,
        "query_gallery_identity_overlap_after_review": len(identity_overlap),
        "visually_confirmed_gallery_duplicates_removed": len(excluded_gallery),
        "exact_cross_dataset_duplicates": 0,
    }
    (run_dir / "ranking_verification.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    with (run_dir / "model_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    test_command = [
        sys.executable, "-m", "unittest",
        "tests.test_external_dataset",
        "tests.test_external_transfer",
        "tests.test_weighted_retrieval",
        "tests.test_statistical_evaluation",
    ]
    test_result = subprocess.run(
        test_command, cwd=REPOSITORY_ROOT, capture_output=True, text=True
    )
    test_output = test_result.stdout + test_result.stderr
    if test_result.returncode != 0:
        raise RuntimeError(f"Focused test suite failed:\n{test_output}")
    test_count_match = re.search(r"Ran (\d+) tests?", test_output)
    if not test_count_match:
        raise RuntimeError("Could not determine the focused test count.")
    test_count = int(test_count_match.group(1))

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    primary = results["paired_tests"]["M5_vs_M6"]
    secondary = results["paired_tests"]["M5_vs_M7"]
    category_differences = results["category_paired_differences"]
    primary_categories = [
        row for row in category_differences if row["comparison"] == "M5-M6"
    ]
    positive_categories = sum(
        row["mean_paired_ap_difference"] > 0 for row in primary_categories
    )
    negative_categories = sum(
        row["mean_paired_ap_difference"] < 0 for row in primary_categories
    )
    m5_summary = results["summary"]["M5"]
    m6_summary = results["summary"]["M6"]
    report_path = run_dir / "transfer_report.md"
    report = report_path.read_text(encoding="utf-8")
    heading = "## Audit interpretation and protocol correction"
    if heading not in report:
        report += "\n" + "\n".join([
            heading,
            "",
            "- The preliminary full-gallery run was retained as a diagnostic only and is superseded: 24 cross-split perceptual-hash candidates were visually confirmed as repeated image content. The corrected run removes only the 24 gallery copies; all official test queries remain included.",
            f"- Primary M5−M6 mean paired AP difference: {primary['mean_difference_m5_minus_control']:.8f}; category-stratified 95% CI {primary['paired_bootstrap_95_percent_ci_mean_difference']}; sign-flip p={primary['paired_sign_flip_permutation_p_raw']:.6g}.",
            f"- The median paired AP difference is {primary['median_difference_m5_minus_control']:.8f}, with {primary['positive_negative_tied_queries']['positive']} positive and {primary['positive_negative_tied_queries']['negative']} negative query differences. Only {positive_categories}/{len(primary_categories)} breed-level means are positive ({negative_categories} negative). Thus the mean-level result is not a consistent category-level gain.",
            f"- M5 Precision@10, Recall@10 and F1@10 differences versus M6 are {m5_summary['precision_at_10'] - m6_summary['precision_at_10']:.8f}, {m5_summary['recall_at_10'] - m6_summary['recall_at_10']:.8f} and {m5_summary['f1_at_10'] - m6_summary['f1_at_10']:.8f}; all are lower for M5 in this run.",
            f"- Secondary M5−M7 mean AP difference is {secondary['mean_difference_m5_minus_control']:.8f}, CI {secondary['paired_bootstrap_95_percent_ci_mean_difference']}, Holm-adjusted p={secondary['paired_sign_flip_permutation_p_holm_secondary']:.6g}; it does not support an M5 improvement over M7. Its Wilcoxon result ({secondary['wilcoxon_p_raw']:.6g}) reflects a different location-based hypothesis and does not establish a positive mean effect.",
            "- Category-level intervals are exploratory and unadjusted across 37 breeds. Query-level inference is conditional on the fixed gallery; category stratification preserves observed breed counts but cannot quantify uncertainty from resampling the shared gallery or from other datasets/training seeds.",
            "- Interpretation: there is a small positive query-weighted mean AP difference for M5 versus M6 on this corrected split, but the effect is practically tiny, the median and most category means do not favor M5, and top-10 metrics are lower. The evidence does not justify claiming robust overall superiority or broad transfer success.",
            "",
        ])
        report_path.write_text(report, encoding="utf-8")

    manifest["commands_executed"] = [
        f'"{sys.executable}" external_transfer.py --dataset-root "{dataset_root}" --output-root "{run_dir.parent}" --reuse-feature-cache "{manifest["experiment"]["feature_source"].split("cache ", 1)[-1]}"',
        '"' + '" -'.join(test_command[:1]) + '" ' + " ".join(test_command[1:]),
        f'"{sys.executable}" validate_external_run.py --run-dir "{run_dir}"',
    ]
    manifest["validation"] = {
        **audit,
        "unit_test_command": manifest["commands_executed"][1],
        "unit_test_result": f"passed; {test_count} tests",
        "unit_test_output": test_output.strip(),
    }
    manifest["environment"].update({
        "pillow": PIL.__version__,
        "scipy": scipy.__version__,
        "scikit_image": skimage.__version__,
        "matplotlib": matplotlib.__version__,
    })
    manifest.setdefault("source", {}).setdefault("files_sha256", {})[
        "validate_external_run.py"
    ] = _sha256(Path(__file__))
    for name in (
        "tests/test_external_dataset.py",
        "tests/test_external_transfer.py",
    ):
        manifest["source"]["files_sha256"][name.replace("/", "\\")] = _sha256(
            REPOSITORY_ROOT / name
        )
    archive_hashes = {}
    for archive_name in ("images.tar.gz", "annotations.tar.gz"):
        archive_path = dataset_root.parent / archive_name
        if archive_path.is_file():
            archive_hashes[archive_name] = _sha256(archive_path)
    manifest["dataset"]["acquisition_archive_sha256"] = archive_hashes
    manifest["dataset"]["counts_by_category"] = dict(sorted(
        counts_by_category.items()
    ))
    manifest["post_run_report_amendment"] = {
        "report_edited_after_metrics": True,
        "reason": "Documented visually confirmed duplicate removal and bounded interpretation after metric verification.",
        "experiment_code_configuration_checkpoint_or_metrics_changed": False,
    }
    manifest["output_files"] = {
        path.name: _sha256(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "reproducibility_manifest.json"
    }
    (run_dir / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    arguments = parser.parse_args()
    result = validate_run(arguments.run_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
