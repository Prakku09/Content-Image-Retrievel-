import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from pathlib import Path
import config


# Set premium styling
sns.set_theme(style="white")

def plot_m1_m5_metric_comparison(results, metric_name, ylabel, title, save_path):
    """
    Plots a bar chart comparing metrics for M1, M2, M3, M4, and M5.
    """
    plt.figure(figsize=(7.5, 5))
    models = ['M1', 'M2', 'M3', 'M4', 'M5']
    values = [results[m] for m in models]
    
    # Harmony color palette
    colors = ['#1565C0', '#00897B', '#EF6C00', '#C62828', '#6A1B9A'] # deep blue, green, orange, red, purple
    
    bars = plt.bar(models, values, color=colors, edgecolor='none', width=0.55)
    
    # Annotate bar values
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
    plt.xlabel('CBIR Models', fontsize=10, fontweight='bold', color='#37474F')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.gca().set_axisbelow(True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_weight_distributions(color_weights, texture_weights, save_dir):
    """
    Plots distribution histograms with kernel density estimates (KDE) for the weights.
    """
    plt.figure(figsize=(10, 4.5))
    
    # Color Weights Distribution
    plt.subplot(1, 2, 1)
    sns.histplot(color_weights, kde=True, color='#1565C0', bins=15, edgecolor='white')
    plt.title('Color Weights ($w_{colour}$) Distribution', fontsize=11, fontweight='bold')
    plt.xlabel('Weight Value')
    plt.ylabel('Density / Frequency')
    plt.grid(axis='both', linestyle=':', alpha=0.5)
    
    # Texture Weights Distribution
    plt.subplot(1, 2, 2)
    sns.histplot(texture_weights, kde=True, color='#00897B', bins=15, edgecolor='white')
    plt.title('Texture Weights ($w_{texture}$) Distribution', fontsize=11, fontweight='bold')
    plt.xlabel('Weight Value')
    plt.ylabel('Density / Frequency')
    plt.grid(axis='both', linestyle=':', alpha=0.5)
    
    plt.suptitle('MLP-Predicted Adaptive Weight Distributions (100 Queries)', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_dir / 'weight_distributions.png', dpi=300)
    plt.close()

def plot_category_wise_weights(categories, avg_color_w, avg_texture_w, save_path):
    """
    Plots a stacked bar chart showing average colour and texture weights per category.
    """
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(categories))
    width = 0.55
    
    plt.bar(x, avg_color_w, width, label='Color Weight ($w_{colour}$)', color='#1565C0', alpha=0.9)
    plt.bar(x, avg_texture_w, width, bottom=avg_color_w, label='Texture Weight ($w_{texture}$)', color='#00897B', alpha=0.9)
    
    plt.title('Category-Wise Average Learned Weights (M5 Adaptive Model)', fontsize=12, fontweight='bold', pad=15)
    plt.ylabel('Stacked Weight Proportion (Sum = 1.0)', fontsize=10, fontweight='bold')
    plt.xlabel('Corel-1K Categories', fontsize=10, fontweight='bold')
    plt.xticks(x, categories, rotation=35, ha='right', fontsize=9)
    plt.legend(loc='lower left', frameon=True)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_m5_query_grid(query_path, query_label, retrieved_paths, retrieved_labels, w_c, w_t, save_path):
    """
    Plots a single row visual grid showing the query, predicted weights, and top-10 M5 results.
    """
    fig, axes = plt.subplots(1, 11, figsize=(18, 2.5))
    
    # Anchor Query Image
    img_q = Image.open(query_path)
    axes[0].imshow(img_q)
    axes[0].set_title(f"Query: {query_label}\n$w_c$: {w_c:.2f}, $w_t$: {w_t:.2f}", fontsize=9, fontweight='bold', color='#1A237E')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    for i, (path, label) in enumerate(zip(retrieved_paths, retrieved_labels)):
        img_r = Image.open(path)
        axes[i+1].imshow(img_r)
        
        is_relevant = (label == query_label)
        color = '#2E7D32' if is_relevant else '#C62828'
        
        axes[i+1].set_title(f"Rank {i+1}", fontsize=9, fontweight='semibold')
        for spine in axes[i+1].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
            spine.set_visible(True)
            
        axes[i+1].set_xlabel(label, fontsize=8.5, color=color, fontweight='semibold')
        axes[i+1].set_xticks([])
        axes[i+1].set_yticks([])
        
    plt.suptitle(f"M5 Adaptive Retrieval Example for {query_label}", fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_m3_vs_m5_comparison_grid(query_path, query_label, m3_retrieval, m5_retrieval, w_c, w_t, save_path):
    """
    Generates a stacked grid comparing M3 (Equal weights) vs M5 (Adaptive weights) for a single query.
    """
    fig, axes = plt.subplots(2, 11, figsize=(18, 5))
    
    # M3 Row (Row 0)
    m3_paths, m3_labels = m3_retrieval
    img_q = Image.open(query_path)
    axes[0, 0].imshow(img_q)
    axes[0, 0].set_title(f"Query: {query_label}", fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel("M3 Equal Fusion\n(0.5, 0.5)", fontsize=11, fontweight='bold', labelpad=15)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    
    for i, (path, label) in enumerate(zip(m3_paths, m3_labels)):
        img_r = Image.open(path)
        axes[0, i+1].imshow(img_r)
        is_rel = (label == query_label)
        color = '#2E7D32' if is_rel else '#C62828'
        
        axes[0, i+1].set_title(f"Rank {i+1}", fontsize=9)
        for spine in axes[0, i+1].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
            spine.set_visible(True)
        axes[0, i+1].set_xlabel(label, fontsize=8.5, color=color)
        axes[0, i+1].set_xticks([])
        axes[0, i+1].set_yticks([])
        
    # M5 Row (Row 1)
    m5_paths, m5_labels = m5_retrieval
    axes[1, 0].imshow(img_q)
    axes[1, 0].set_ylabel(f"M5 Adaptive\n($w_c$={w_c:.2f}, $w_t$={w_t:.2f})", fontsize=11, fontweight='bold', labelpad=15)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    
    for i, (path, label) in enumerate(zip(m5_paths, m5_labels)):
        img_r = Image.open(path)
        axes[1, i+1].imshow(img_r)
        is_rel = (label == query_label)
        color = '#2E7D32' if is_rel else '#C62828'
        
        for spine in axes[1, i+1].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
            spine.set_visible(True)
        axes[1, i+1].set_xlabel(label, fontsize=8.5, color=color)
        axes[1, i+1].set_xticks([])
        axes[1, i+1].set_yticks([])
        
    plt.suptitle(f"CBIR Search Comparison for {query_label}: M3 Baseline vs M5 Adaptive Model", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
