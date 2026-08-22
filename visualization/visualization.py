import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from pathlib import Path
import config

# Set seaborn style for premium aesthetics
sns.set_theme(style="white")

def plot_metric_comparison(results, metric_name, ylabel, title, save_path):
    """
    Generates a premium bar plot comparing M1-M4 for a single metric.
    
    Args:
        results (dict): Dict mapping model name ('M1', 'M2', 'M3', 'M4') to metric value.
        metric_name (str): Name of the metric.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        save_path (Path or str): File path to save the generated plot.
    """
    plt.figure(figsize=(7, 4.5))
    models = list(results.keys())
    values = list(results.values())
    
    # Visual-tailored colors
    colors = ['#1565C0', '#00897B', '#EF6C00', '#C62828'] # deep blue, green, orange, red
    
    bars = plt.bar(models, values, color=colors, edgecolor='none', width=0.55)
    
    # Annotate values on top of bars
    max_val = max(values) if values else 1.0
    for bar in bars:
        height = bar.get_height()
        if 'time' in metric_name.lower():
            label = f"{height*1000:.2f} ms"
        else:
            label = f"{height:.4f}"
        plt.text(
            bar.get_x() + bar.get_width()/2.0, 
            height + (max_val * 0.015), 
            label, 
            ha='center', 
            va='bottom', 
            fontsize=9.5, 
            fontweight='semibold',
            color='#37474F'
        )
        
    plt.title(title, fontsize=12, fontweight='bold', pad=15, color='#263238')
    plt.ylabel(ylabel, fontsize=10, fontweight='bold', color='#37474F')
    plt.xlabel('Baseline Models', fontsize=10, fontweight='bold', color='#37474F')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.gca().set_axisbelow(True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_query_comparison(query_path, query_label, model_results, save_path):
    """
    Generates a stacked grid visualization showing the query image and the top-10
    retrieved results across all 4 baseline models. Relevant retrievals are bordered
    in green, non-relevant in red.
    
    Args:
        query_path (Path or str): Path to the query image.
        query_label (str): True category label of the query image.
        model_results (dict): Dict mapping model name ('M1', 'M2', 'M3', 'M4') to
                             tuple of (retrieved_paths, retrieved_labels) for top-10.
        save_path (Path or str): Path to save the plot.
    """
    # 4 rows (M1, M2, M3, M4), 11 columns (Query, Rank 1-10)
    fig, axes = plt.subplots(4, 11, figsize=(18, 8.5))
    
    for row_idx, model_name in enumerate(['M1', 'M2', 'M3', 'M4']):
        retrieved_paths, retrieved_labels = model_results[model_name]
        
        # Load and plot query image in column 0
        img_q = Image.open(query_path)
        axes[row_idx, 0].imshow(img_q)
        if row_idx == 0:
            axes[row_idx, 0].set_title("Query Image\n" + query_label, fontsize=11, fontweight='bold', color='#1A237E')
        axes[row_idx, 0].set_ylabel(model_name, fontsize=12, fontweight='bold', labelpad=15, color='#37474F')
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        
        # Plot the top-10 retrieved images
        for i, (path, label) in enumerate(zip(retrieved_paths, retrieved_labels)):
            img_r = Image.open(path)
            axes[row_idx, i+1].imshow(img_r)
            
            is_relevant = (label == query_label)
            border_color = '#2E7D32' if is_relevant else '#C62828' # green for relevant, red for non-relevant
            
            if row_idx == 0:
                axes[row_idx, i+1].set_title(f"Rank {i+1}", fontsize=10, fontweight='semibold')
                
            # Apply colored border representing relevance
            for spine in axes[row_idx, i+1].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3.0)
                spine.set_visible(True)
                
            axes[row_idx, i+1].set_xlabel(label, fontsize=8.5, color=border_color, fontweight='semibold')
            axes[row_idx, i+1].set_xticks([])
            axes[row_idx, i+1].set_yticks([])
            
    plt.suptitle(f"CBIR Top-10 Retrieval Results Comparison for Query: {query_label}", fontsize=14, fontweight='bold', y=0.98, color='#263238')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def generate_comparison_table(eval_results):
    """
    Formulates the evaluation metrics into an ASCII/Markdown table.
    
    Args:
        eval_results (dict): Dict mapping model names to their evaluated metrics.
        
    Returns:
        str: Markdown table string.
    """
    table = []
    table.append("| Model | Mean Precision@10 | Mean Recall@10 | Mean F1-score@10 | mAP (Full Gallery) | Avg Retrieval Time |")
    table.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    models = [m for m in ['M1', 'M2', 'M3', 'M4', 'M5'] if m in eval_results]
    for model in models:
        m = eval_results[model]
        table.append(f"| **{model}** | {m['mean_precision']:.4f} | {m['mean_recall']:.4f} | {m['mean_f1']:.4f} | {m['mAP']:.4f} | {m['mean_time']*1000:.2f} ms |")
    return "\n".join(table)
