"""Generate paper tables and figures from one independently validated run."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


MODEL_ORDER = ("M1", "M2", "M3", "M4", "M5", "M6")
MODEL_COLORS = {
    "M1": "#4A7BB0",
    "M2": "#5CA4A9",
    "M3": "#E69F53",
    "M4": "#D35D55",
    "M5": "#1E7B34",
    "M6": "#546E7A",
}
METRICS = (
    ("mean_precision_at_10", "Precision@10"),
    ("mean_recall_at_10", "Recall@10"),
    ("mean_f1_at_10", "F1@10"),
    ("mean_average_precision", "mAP"),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def save_figure(fig, output_dir, stem):
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def generate_pipeline_figure(output_dir):
    fig, axis = plt.subplots(figsize=(8.2, 10.5))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 9.4)
    axis.axis("off")
    nodes = [
        (
            8.15,
            "Corel-1K source scan and near-duplicate audit\n"
            "1,000 images; one visually confirmed near-duplicate pair",
        ),
        (
            7.0,
            "Corrected partitions\n"
            "899 gallery | 100 held-out queries | 1 gallery duplicate excluded\n"
            "Gallery-only MLP split: 700 training | 199 validation",
        ),
        (
            5.85,
            "Handcrafted feature extraction\n"
            "HSV colour moments: 9 dimensions | GLCM texture: 4 dimensions\n"
            "Concatenated image descriptor: 13 dimensions",
        ),
        (
            4.7,
            "Gallery-fitted normalization and checkpoint selection\n"
            "Min-Max bounds fitted on corrected gallery only\n"
            "MLP: 13-16-8-2 triplet-margin training; validation-only checkpoint selection",
        ),
        (
            3.55,
            "M1-M4 baseline retrieval\n"
            "M1 colour-only | M2 texture-only\n"
            "M3 equal and M4 fixed 0.7/0.3 feature-level fusion",
        ),
        (
            2.4,
            "M5 adaptive and M6 fixed-mean control retrieval\n"
            "M5 predicts query-specific Softmax weights\n"
            "M6 applies mean M5 weights using the same distance formula",
        ),
        (
            1.25,
            "Ranking and evaluation\n"
            "Rank against the same corrected gallery for all models\n"
            "Precision@10 | Recall@10 | F1@10 | full-gallery AP and mAP",
        ),
    ]
    for index, (y, label) in enumerate(nodes):
        axis.add_patch(plt.Rectangle(
            (0.08, y - 0.39),
            0.84,
            0.78,
            facecolor="#F4F7FB",
            edgecolor="#42658A",
            linewidth=1.4,
            joinstyle="round",
        ))
        axis.text(
            0.5,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8.0,
            linespacing=1.35,
            color="#172B42",
        )
        if index < len(nodes) - 1:
            next_y = nodes[index + 1][0]
            axis.annotate(
                "",
                xy=(0.5, next_y + 0.42),
                xytext=(0.5, y - 0.42),
                arrowprops={"arrowstyle": "-|>", "color": "#34495E", "lw": 1.2},
            )
    axis.text(
        0.5,
        0.3,
        "No held-out query or its identified near-duplicate is used for model selection.",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#334155",
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "fig1_corrected_pipeline")


def generate(run_dir, output_dir, dataset_root):
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    dataset_root = Path(dataset_root).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    validation = json.loads((run_dir / "independent_validation.json").read_text())
    results = json.loads((run_dir / "results.json").read_text())
    run_manifest = json.loads(
        (run_dir / "reproducibility_manifest.json").read_text()
    )
    if validation.get("status") != "passed":
        raise ValueError("Paper assets require a passed independent validation.")
    if results.get("run_id") != run_manifest.get("run_id"):
        raise ValueError("Results and manifest refer to different runs.")
    if set(results["aggregate_metrics"]) != set(MODEL_ORDER):
        raise ValueError("Corrected results must contain exactly M1-M6.")
    if results["counts"] != {
        "dataset_images": 1000,
        "gallery": 899,
        "queries": 100,
        "excluded_duplicate_gallery": 1,
        "mlp_train": 700,
        "mlp_validation": 199,
    }:
        raise ValueError("Corrected reference split counts do not match expected protocol.")
    if validation.get("maximum_metric_absolute_error") != 0.0:
        raise ValueError("Saved-ranking metric verification was not exact.")

    metrics = results["aggregate_metrics"]
    table_rows = []
    for model in MODEL_ORDER:
        row = {"model": model}
        for key, label in METRICS:
            row[label] = metrics[model][key]
        row["mean_ranking_sort_ms_per_query"] = (
            metrics[model]["mean_ranking_sort_seconds"] * 1000
        )
        table_rows.append(row)
    with (output_dir / "table_III_corrected_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.DictWriter(target, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    weights = read_csv(run_dir / "per_query_weights.csv")
    category_rows = read_csv(run_dir / "per_category_metrics.csv")
    categories = sorted({row["category"] for row in weights})
    category_weight_rows = []
    for category in categories:
        selected = [row for row in weights if row["category"] == category]
        category_weight_rows.append({
            "category": category,
            "query_count": len(selected),
            "mean_color_weight": float(np.mean(
                [float(row["m5_color_weight"]) for row in selected]
            )),
            "mean_texture_weight": float(np.mean(
                [float(row["m5_texture_weight"]) for row in selected]
            )),
        })
    with (output_dir / "category_mean_weights.csv").open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.DictWriter(target, fieldnames=list(category_weight_rows[0]))
        writer.writeheader()
        writer.writerows(category_weight_rows)

    cat_metric_index = {
        (row["category"], row["model"]): row for row in category_rows
    }
    category_difference_rows = []
    for category in categories:
        m5 = cat_metric_index[(category, "M5")]
        m6 = cat_metric_index[(category, "M6")]
        category_difference_rows.append({
            "category": category,
            "M5_mAP": float(m5["mean_average_precision"]),
            "M6_mAP": float(m6["mean_average_precision"]),
            "M5_minus_M6_mAP": (
                float(m5["mean_average_precision"])
                - float(m6["mean_average_precision"])
            ),
        })
    with (output_dir / "category_m5_m6_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.DictWriter(
            target, fieldnames=list(category_difference_rows[0])
        )
        writer.writeheader()
        writer.writerows(category_difference_rows)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    generate_pipeline_figure(output_dir)

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(METRICS))
    width = 0.12
    for index, model in enumerate(MODEL_ORDER):
        values = [
            metrics[model][key] for key, _ in METRICS
        ]
        offset = (index - (len(MODEL_ORDER) - 1) / 2) * width
        bars = axis.bar(
            x + offset,
            values,
            width,
            label=model,
            color=MODEL_COLORS[model],
            edgecolor="#263238" if model in ("M5", "M6") else "none",
            linewidth=0.7 if model in ("M5", "M6") else 0,
            hatch="///" if model == "M5" else ("\\\\\\" if model == "M6" else None),
        )
        if model in ("M5", "M6"):
            for bar, value in zip(bars, values):
                axis.annotate(
                    f"{value:.3f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    fontsize=5.8,
                    rotation=90,
                )
    axis.set_ylabel("Score")
    axis.set_xticks(x, [label for _, label in METRICS])
    axis.set_ylim(0, max(0.85, max(
        metrics[model][key] for model in MODEL_ORDER for key, _ in METRICS
    ) * 1.15))
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    axis.set_axisbelow(True)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18),
                frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig2_corrected_grouped_metrics")

    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    bars = axis.bar(
        MODEL_ORDER,
        [metrics[model]["mean_average_precision"] for model in MODEL_ORDER],
        color=[MODEL_COLORS[model] for model in MODEL_ORDER],
    )
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_ylabel("Mean Average Precision (mAP)")
    axis.set_xlabel("CBIR model")
    axis.set_ylim(0, 0.55)
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    axis.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig6_corrected_map")

    color_weights = np.asarray(
        [float(row["m5_color_weight"]) for row in weights]
    )
    texture_weights = np.asarray(
        [float(row["m5_texture_weight"]) for row in weights]
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    axes[0].hist(color_weights, bins=15, density=True, color="#4A7BB0",
                 edgecolor="white")
    axes[0].set_title("Colour weights")
    axes[0].set_xlabel("Predicted colour weight")
    axes[0].set_ylabel("Density")
    axes[1].hist(texture_weights, bins=15, density=True, color="#5CA4A9",
                 edgecolor="white")
    axes[1].set_title("Texture weights")
    axes[1].set_xlabel("Predicted texture weight")
    for axis in axes:
        axis.grid(axis="y", linestyle=":", alpha=0.5)
        axis.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig3_corrected_weight_distributions")

    category_display = {
        "mountains_and_snow": "mountains and snow",
        "peolpe_and_villages_in_Africa": "people and villages in Africa",
    }
    fig, axis = plt.subplots(figsize=(10.0, 5.2))
    positions = np.arange(len(categories))
    means_color = [row["mean_color_weight"] for row in category_weight_rows]
    means_texture = [row["mean_texture_weight"] for row in category_weight_rows]
    axis.bar(positions, means_color, label="Mean colour weight", color="#4A7BB0")
    axis.bar(positions, means_texture, bottom=means_color,
             label="Mean texture weight", color="#5CA4A9")
    axis.set_ylabel("Mean predicted weight (sum = 1)")
    axis.set_xlabel("Corel-1K category")
    axis.set_xticks(
        positions,
        [category_display.get(category, category) for category in categories],
        rotation=25,
        ha="right",
    )
    axis.set_ylim(0, 1.05)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=2,
    )
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    axis.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig4_corrected_category_weights")

    fig, axis = plt.subplots(figsize=(9.0, 4.2))
    deltas = [row["M5_minus_M6_mAP"] for row in category_difference_rows]
    colors = ["#1E7B34" if value >= 0 else "#C62828" for value in deltas]
    axis.barh(categories, deltas, color=colors)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_xlabel("Category mean AP difference (M5 - M6)")
    axis.set_ylabel("Corel-1K category")
    axis.grid(axis="x", linestyle=":", alpha=0.5)
    axis.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig7_corrected_category_ap_difference")

    ranking_file = np.load(run_dir / "rankings.npz", allow_pickle=False)
    query_paths = ranking_file["query_paths"].tolist()
    gallery_paths = ranking_file["gallery_paths"].tolist()
    query_index = next(
        index for index, path in enumerate(query_paths)
        if Path(path).parent.name == "beaches"
    )
    query_path = dataset_root / query_paths[query_index]
    model_rankings = {
        model: ranking_file[model][query_index] for model in ("M3", "M5")
    }
    fig, axes = plt.subplots(2, 11, figsize=(18, 4.7))
    query_image = Image.open(query_path).convert("RGB")
    for row_index, model in enumerate(("M3", "M5")):
        axes[row_index, 0].imshow(query_image)
        axes[row_index, 0].set_ylabel(
            "M3 equal fusion" if model == "M3" else "M5 adaptive fusion"
        )
        axes[row_index, 0].set_title("Held-out query" if row_index == 0 else "")
        axes[row_index, 0].set_xticks([])
        axes[row_index, 0].set_yticks([])
        for rank, gallery_index in enumerate(model_rankings[model][:10], start=1):
            image_path = dataset_root / gallery_paths[int(gallery_index)]
            image = Image.open(image_path).convert("RGB")
            axis = axes[row_index, rank]
            axis.imshow(image)
            relevant = Path(gallery_paths[int(gallery_index)]).parent.name == "beaches"
            color = "#2E7D32" if relevant else "#C62828"
            axis.set_title(f"Rank {rank}")
            axis.set_xlabel(
                Path(gallery_paths[int(gallery_index)]).parent.name,
                fontsize=6,
                color=color,
            )
            for spine in axis.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        "Corrected-run qualitative example; green = same category, red = different",
        fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "fig5_corrected_qualitative_example")
    ranking_file.close()

    manifest = {
        "source_run_id": results["run_id"],
        "source_run_directory": str(run_dir),
        "source_revision": run_manifest.get("source_revision"),
        "source_results_sha256": sha256(run_dir / "results.json"),
        "source_validation_sha256": sha256(run_dir / "independent_validation.json"),
        "source_checkpoint_sha256": sha256(run_dir / "m5_checkpoint.pt"),
        "source_query_metrics_sha256": sha256(run_dir / "per_query_metrics.csv"),
        "source_query_weights_sha256": sha256(run_dir / "per_query_weights.csv"),
        "source_rankings_sha256": sha256(run_dir / "rankings.npz"),
        "gallery_query_training_validation_counts": results["counts"],
        "output_files_sha256": {},
        "figure_data": {
            "fig2_corrected_grouped_metrics": {
                "models": list(MODEL_ORDER),
                "metrics": [label for _, label in METRICS],
                "values": {
                    model: {
                        label: metrics[model][key] for key, label in METRICS
                    } for model in MODEL_ORDER
                },
                "caption": "Retrieval metrics for M1-M6 on the corrected 899-image Corel gallery and 100 held-out queries.",
            },
            "fig6_corrected_map": {
                "mAP": {
                    model: metrics[model]["mean_average_precision"]
                    for model in MODEL_ORDER
                }
            },
            "fig5_query": query_paths[query_index],
            "fig5_rankings": {
                model: [
                    gallery_paths[int(index)]
                    for index in model_rankings[model][:10]
                ] for model in ("M3", "M5")
            },
        },
    }
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "asset_manifest.json":
            manifest["output_files_sha256"][path.name] = sha256(path)
    (output_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()
    manifest = generate(args.run_dir, args.output_dir, args.dataset_root)
    print(json.dumps({
        "status": "generated",
        "output_dir": str(args.output_dir.resolve()),
        "output_files": sorted(manifest["output_files_sha256"]),
    }, indent=2))


if __name__ == "__main__":
    main()
