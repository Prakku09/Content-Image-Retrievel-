"""
Refined IEEE Publication-Quality Grouped Bar Chart Generator
CBIR Model Performance Comparison (M1 to M5 across 4 metrics)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def create_ieee_chart():
    # Set Matplotlib parameters for IEEE Transactions publication standard
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
        'mathtext.fontset': 'stix',
        'font.size': 8.0,
        'axes.labelsize': 8.5,
        'axes.titlesize': 8.5,
        'xtick.labelsize': 7.8,
        'ytick.labelsize': 7.8,
        'legend.fontsize': 6.8,
        'figure.titlesize': 9.0,
        'axes.linewidth': 0.75,
        'grid.linewidth': 0.5,
        'lines.linewidth': 0.8,
        'patch.linewidth': 0.6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    # Metrics and models data
    metrics = ['Precision@10', 'Recall@10', 'F1-score@10', 'mAP']
    
    # Models configuration with publication-grade color scheme
    # M5 (AI Adaptive) is highlighted with high-contrast emerald green, diagonal hatching, and a bold outline
    models = [
        {
            'key': 'M1',
            'name': 'M1: Colour Only',
            'scores': [0.6200, 0.0689, 0.1240, 0.4048],
            'color': '#4A7BB0',       # Steel Blue
            'edgecolor': '#284E75',
            'hatch': '',
            'alpha': 0.95,
            'linewidth': 0.6
        },
        {
            'key': 'M2',
            'name': 'M2: Texture Only',
            'scores': [0.4680, 0.0520, 0.0936, 0.3508],
            'color': '#5CA4A9',       # Slate Teal
            'edgecolor': '#2C5A5D',
            'hatch': '',
            'alpha': 0.95,
            'linewidth': 0.6
        },
        {
            'key': 'M3',
            'name': 'M3: Equal Fusion',
            'scores': [0.6990, 0.0777, 0.1398, 0.4652],
            'color': '#E69F53',       # Warm Amber
            'edgecolor': '#9E5B10',
            'hatch': '',
            'alpha': 0.95,
            'linewidth': 0.6
        },
        {
            'key': 'M4',
            'name': 'M4: Fixed Fusion',
            'scores': [0.6730, 0.0748, 0.1346, 0.4340],
            'color': '#D35D55',       # Rust Coral
            'edgecolor': '#852922',
            'hatch': '',
            'alpha': 0.95,
            'linewidth': 0.6
        },
        {
            'key': 'M5',
            'name': 'M5: AI Adaptive (Proposed)',
            'scores': [0.7200, 0.0800, 0.1440, 0.4845],
            'color': '#1E7B34',       # Vibrant Emerald Green (Standout)
            'edgecolor': '#0C4018',
            'hatch': '///',           # Diagonal hatching for instant distinction
            'alpha': 0.98,
            'linewidth': 1.0
        }
    ]

    num_metrics = len(metrics)
    num_models = len(models)
    
    # -------------------------------------------------------------
    # VARIANT 1: Primary IEEE Publication Figure (Legend Above Chart)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 2.65))

    x = np.arange(num_metrics)  # 0, 1, 2, 3
    bar_width = 0.145
    group_span = bar_width * num_models
    offsets = np.linspace(-group_span / 2 + bar_width / 2, 
                          group_span / 2 - bar_width / 2, 
                          num_models)

    bars_dict = {}
    for i, m in enumerate(models):
        pos = x + offsets[i]
        bars = ax.bar(
            pos,
            m['scores'],
            width=bar_width,
            label=m['name'],
            color=m['color'],
            edgecolor=m['edgecolor'],
            hatch=m['hatch'],
            alpha=m['alpha'],
            linewidth=m['linewidth'],
            zorder=3
        )
        bars_dict[m['key']] = bars

    # Format Axes
    ax.set_ylabel('Score', fontsize=8.5, fontweight='bold', color='#1A1A1A', labelpad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=7.8, fontweight='bold', color='#1A1A1A')
    
    ax.set_ylim(0, 0.80)
    ax.set_yticks(np.arange(0, 0.81, 0.1))
    
    # Subtle Y-gridlines
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.45, color='#B0BEC5', zorder=0)
    ax.set_axisbelow(True)

    # Spine formatting
    sns.despine(ax=ax, top=True, right=True)
    ax.spines['left'].set_color('#37474F')
    ax.spines['bottom'].set_color('#37474F')
    ax.tick_params(axis='both', which='both', colors='#1A1A1A', length=3.0, width=0.75)

    # Legend placed neatly above the plot area across 2 rows (3 and 2 items)
    legend = ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        facecolor='#FFFFFF',
        edgecolor='#CFD8DC',
        framealpha=0.95,
        columnspacing=0.8,
        handletextpad=0.3,
        handlelength=1.2,
        handleheight=0.8,
        borderpad=0.3,
        fontsize=6.5
    )
    legend.get_frame().set_linewidth(0.6)

    # Save Variant 1
    png_path = Path("cbir_model_comparison_grouped_bar.png")
    svg_path = Path("cbir_model_comparison_grouped_bar.svg")
    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)

    # -------------------------------------------------------------
    # VARIANT 2: Compact Value-Annotated Version
    # -------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(3.5, 2.75))
    for i, m in enumerate(models):
        pos = x + offsets[i]
        b = ax2.bar(
            pos,
            m['scores'],
            width=bar_width,
            label=m['name'],
            color=m['color'],
            edgecolor=m['edgecolor'],
            hatch=m['hatch'],
            alpha=m['alpha'],
            linewidth=m['linewidth'],
            zorder=3
        )
        # Annotate M5 values
        if m['key'] == 'M5':
            for bar, val in zip(b, m['scores']):
                h = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.012,
                    f'{val:.3f}' if val < 0.1 else f'{val:.2f}',
                    ha='center', va='bottom',
                    fontsize=5.8,
                    fontweight='bold',
                    color='#0C4018'
                )

    ax2.set_ylabel('Score', fontsize=8.5, fontweight='bold', color='#1A1A1A', labelpad=4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=7.8, fontweight='bold', color='#1A1A1A')
    ax2.set_ylim(0, 0.82)
    ax2.set_yticks(np.arange(0, 0.81, 0.1))
    ax2.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.45, color='#B0BEC5', zorder=0)
    ax2.set_axisbelow(True)
    sns.despine(ax=ax2, top=True, right=True)
    ax2.spines['left'].set_color('#37474F')
    ax2.spines['bottom'].set_color('#37474F')
    ax2.tick_params(axis='both', which='both', colors='#1A1A1A', length=3.0, width=0.75)

    legend2 = ax2.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        facecolor='#FFFFFF',
        edgecolor='#CFD8DC',
        framealpha=0.95,
        columnspacing=0.8,
        handletextpad=0.3,
        handlelength=1.2,
        handleheight=0.8,
        borderpad=0.3,
        fontsize=6.5
    )
    legend2.get_frame().set_linewidth(0.6)

    annotated_png = Path("cbir_model_comparison_annotated.png")
    annotated_svg = Path("cbir_model_comparison_annotated.svg")
    fig2.savefig(annotated_png, dpi=300, bbox_inches='tight', pad_inches=0.03)
    fig2.savefig(annotated_svg, format='svg', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig2)

    # Copy to output/part2/plots as well
    out_plots_dir = Path("output/part2/plots")
    if out_plots_dir.exists():
        import shutil
        shutil.copy(png_path, out_plots_dir / "fig2_model_performance_comparison.png")
        shutil.copy(svg_path, out_plots_dir / "fig2_model_performance_comparison.svg")

    print("[SUCCESS] Refined IEEE charts generated successfully.")

if __name__ == '__main__':
    create_ieee_chart()
