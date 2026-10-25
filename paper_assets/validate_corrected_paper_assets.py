"""Check paper tables/figures against a validated corrected CBIR run."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


MODELS = ("M1", "M2", "M3", "M4", "M5", "M6")
METRIC_FIELDS = {
    "Precision@10": "mean_precision_at_10",
    "Recall@10": "mean_recall_at_10",
    "F1@10": "mean_f1_at_10",
    "mAP": "mean_average_precision",
}
REQUIRED_ASSETS = (
    "fig1_corrected_pipeline.png",
    "fig1_corrected_pipeline.svg",
    "fig2_corrected_grouped_metrics.png",
    "fig2_corrected_grouped_metrics.svg",
    "fig3_corrected_weight_distributions.png",
    "fig3_corrected_weight_distributions.svg",
    "fig4_corrected_category_weights.png",
    "fig4_corrected_category_weights.svg",
    "fig5_corrected_qualitative_example.png",
    "fig5_corrected_qualitative_example.svg",
    "fig6_corrected_map.png",
    "fig6_corrected_map.svg",
    "fig7_corrected_category_ap_difference.png",
    "fig7_corrected_category_ap_difference.svg",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(run_dir, asset_dir):
    run_dir = Path(run_dir).resolve()
    asset_dir = Path(asset_dir).resolve()
    result = json.loads((run_dir / "results.json").read_text())
    validation = json.loads((run_dir / "independent_validation.json").read_text())
    asset_manifest = json.loads((asset_dir / "asset_manifest.json").read_text())
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    require(validation.get("status") == "passed",
            "Source corrected run has no passed independent validation.")
    require(validation.get("maximum_metric_absolute_error") == 0.0,
            "Source ranking-to-metric validation was not exact.")
    require(asset_manifest.get("source_run_id") == result.get("run_id"),
            "Asset manifest run ID differs from source results.")
    require(
        asset_manifest.get("source_results_sha256")
        == sha256(run_dir / "results.json"),
        "Asset source-results hash mismatch.",
    )
    require(
        asset_manifest.get("source_validation_sha256")
        == sha256(run_dir / "independent_validation.json"),
        "Asset source-validation hash mismatch.",
    )
    for filename, expected_hash in asset_manifest.get(
        "output_files_sha256", {}
    ).items():
        path = asset_dir / filename
        require(path.is_file(), f"Generated output is missing: {filename}.")
        if path.is_file():
            require(
                sha256(path) == expected_hash,
                f"Generated output hash mismatch: {filename}.",
            )
    table_path = asset_dir / "table_III_corrected_metrics.csv"
    with table_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    require(tuple(row["model"] for row in rows) == MODELS,
            "Table rows or model order do not match M1-M6.")
    for row in rows:
        model = row["model"]
        for label, key in METRIC_FIELDS.items():
            error = abs(
                float(row[label]) - result["aggregate_metrics"][model][key]
            )
            require(error <= 1e-12,
                    f"Table {model} {label} differs by {error}.")
            plot_value = asset_manifest["figure_data"][
                "fig2_corrected_grouped_metrics"
            ]["values"][model][label]
            require(abs(plot_value - float(row[label])) <= 1e-12,
                    f"Figure 2 source value differs from table: {model} {label}.")
        map_value = asset_manifest["figure_data"]["fig6_corrected_map"][
            "mAP"
        ][model]
        require(
            abs(map_value - float(row["mAP"])) <= 1e-12,
            f"Figure 6 source value differs from table for {model}.",
        )

    for name in REQUIRED_ASSETS:
        path = asset_dir / name
        require(path.is_file() and path.stat().st_size > 0,
                f"Required figure is missing or empty: {name}")
        expected = asset_manifest.get("output_files_sha256", {}).get(name)
        require(expected is not None, f"Asset hash not recorded for {name}.")
        if path.is_file() and expected:
            require(sha256(path) == expected, f"Figure hash mismatch: {name}.")
        if path.suffix == ".png" and path.is_file():
            try:
                with Image.open(path) as image:
                    image.verify()
            except OSError as error:
                errors.append(f"Invalid PNG {name}: {error}")
    fig1_svg = asset_dir / "fig1_corrected_pipeline.svg"
    if fig1_svg.is_file():
        diagram_source = fig1_svg.read_text(encoding="utf-8")
        require(
            "0.4845" not in diagram_source and "12.9%" not in diagram_source,
            "Corrected pipeline diagram contains an obsolete historical result claim.",
        )

    report = {
        "status": "passed" if not errors else "failed",
        "run_id": result.get("run_id"),
        "models_checked": list(MODELS),
        "table_metrics_checked": len(MODELS) * len(METRIC_FIELDS),
        "figure_2_and_6_values_match_table": not errors,
        "figure_assets_checked": len(REQUIRED_ASSETS),
        "errors": errors,
    }
    if errors:
        raise AssertionError(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--asset-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.run_dir, args.asset_dir), indent=2))


if __name__ == "__main__":
    main()
