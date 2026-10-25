"""Generate publication charts from a verified, current M1-M6 experiment run."""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config
from data.dataset_loader import load_dataset


def create_ieee_chart():
    with open(config.EXPERIMENT_RESULTS_PATH, 'rb') as result_file:
        results = pickle.load(result_file)
    experiment_config = results.get('experiment_config', {})
    metrics_by_model = results.get('metrics', {})
    model_names = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    if (
        experiment_config.get('status') != 'freshly_computed'
        or not all(name in metrics_by_model for name in model_names)
    ):
        raise ValueError(
            "A complete fresh M1-M6 run is required before generating the grouped chart."
        )

    gallery_paths, _, query_paths, _ = load_dataset()
    from main import _feature_fingerprint

    current_fingerprint = _feature_fingerprint(gallery_paths, query_paths)
    if current_fingerprint != experiment_config.get('feature_fingerprint'):
        raise ValueError(
            "Experiment results do not match the current dataset; rerun the pipeline."
        )

    metric_specs = [
        ('mean_precision', 'Precision@10'),
        ('mean_recall', 'Recall@10'),
        ('mean_f1', 'F1@10'),
        ('mAP', 'mAP'),
    ]
    values = np.asarray([
        [metrics_by_model[name][key] for key, _ in metric_specs]
        for name in model_names
    ])
    colors = ['#4A7BB0', '#5CA4A9', '#E69F53', '#D35D55', '#1E7B34', '#546E7A']
    hatches = ['', '', '', '', '///', '\\\\\\']
    sns.set_theme(style='white')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.labelsize': 8.5,
        'axes.titlesize': 9,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 6.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(metric_specs))
    width = 0.12
    offsets = (np.arange(len(model_names)) - (len(model_names) - 1) / 2) * width
    for index, model_name in enumerate(model_names):
        bars = ax.bar(
            x + offsets[index],
            values[index],
            width=width,
            label=model_name,
            color=colors[index],
            edgecolor='#263238' if model_name in ('M5', 'M6') else 'none',
            linewidth=0.7 if model_name in ('M5', 'M6') else 0,
            hatch=hatches[index],
        )
        if model_name in ('M5', 'M6'):
            for bar, value in zip(bars, values[index]):
                ax.annotate(
                    f'{value:.3f}',
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2),
                    textcoords='offset points',
                    ha='center',
                    fontsize=5.8,
                    rotation=90,
                )

    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_specs])
    ax.set_ylim(0, max(0.85, float(values.max()) * 1.15))
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.18), frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()

    for output_dir in (config.FIGURES_DIR, config.PLOTS_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'cbir_model_comparison_grouped_bar.png', dpi=300,
                    bbox_inches='tight')
        fig.savefig(output_dir / 'cbir_model_comparison_annotated.png', dpi=300,
                    bbox_inches='tight')
        for filename in (
            'cbir_model_comparison_grouped_bar.svg',
            'cbir_model_comparison_annotated.svg',
        ):
            svg_path = output_dir / filename
            fig.savefig(svg_path, bbox_inches='tight')
            svg_lines = svg_path.read_text(encoding='utf-8').splitlines()
            svg_path.write_text(
                '\n'.join(line.rstrip() for line in svg_lines) + '\n',
                encoding='utf-8',
            )
    plt.close(fig)
    print("[SUCCESS] Fresh M1-M6 grouped charts generated.")


if __name__ == '__main__':
    create_ieee_chart()
