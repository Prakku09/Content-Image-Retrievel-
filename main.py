"""
Unified Main Entry Point for Corel-1K CBIR Research Pipeline
End-to-End Execution of Baseline (M1-M4) and Proposed AI Adaptive Weighting (M5).
"""

import sys
import argparse
import time
import pickle
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Import pipeline packages
import config
from data.dataset_loader import load_dataset
from features.feature_extractor import extract_all_features
from features.normalization import MinMaxNormalizer
from model.mlp_model import AdaptiveWeightMLP
from model.mlp_trainer import train_mlp
from retrieval.retrieval import get_representation, retrieve_images
from retrieval.evaluation import evaluate_all
from visualization.visualization import plot_metric_comparison, plot_query_comparison, generate_comparison_table
from visualization.visualize_part2 import (
    plot_m1_m5_metric_comparison,
    plot_weight_distributions,
    plot_category_wise_weights,
    plot_m5_query_grid,
    plot_m3_vs_m5_comparison_grid
)
from visualization.generate_grouped_bar_chart import create_ieee_chart

def run_part1(force_reextract=False):
    """
    Executes Part 1: Dataset loading, feature extraction (HSV + GLCM), Min-Max normalization,
    and baseline retrieval evaluation (M1, M2, M3, M4).
    """
    print("\n" + "="*65)
    print("  STAGE 1: FEATURE EXTRACTION & BASELINE EVALUATION (M1-M4)")
    print("="*65)
    
    pkl_path = config.FEATURES_PKL_PATH
    
    if pkl_path.exists() and not force_reextract:
        print(f"[CACHE] Found pre-extracted features at: {pkl_path}")
        print("        Loading features from disk (use --force-extract to recompute)...")
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
    else:
        print("\n1. Loading dataset and generating deterministic splits...")
        gallery_paths, gallery_labels, query_paths, query_labels = load_dataset()
        print(f"   Gallery images: {len(gallery_paths)} (90/category)")
        print(f"   Query images  : {len(query_paths)} (10/category)")
        
        print("\n2. Extracting HSV color moments (9-dim) and GLCM texture (4-dim)...")
        gal_c, gal_t = [], []
        for p in tqdm(gallery_paths, desc="Gallery Features"):
            c, t = extract_all_features(p)
            gal_c.append(c)
            gal_t.append(t)
            
        q_c, q_t = [], []
        for p in tqdm(query_paths, desc="Query Features"):
            c, t = extract_all_features(p)
            q_c.append(c)
            q_t.append(t)
            
        gal_c = np.array(gal_c, dtype=np.float32)
        gal_t = np.array(gal_t, dtype=np.float32)
        q_c = np.array(q_c, dtype=np.float32)
        q_t = np.array(q_t, dtype=np.float32)
        
        print("\n3. Fitting Min-Max Normalizer on gallery distribution...")
        c_norm = MinMaxNormalizer()
        t_norm = MinMaxNormalizer()
        norm_gal_c = c_norm.fit_transform(gal_c)
        norm_q_c = c_norm.transform(q_c)
        norm_gal_t = t_norm.fit_transform(gal_t)
        norm_q_t = t_norm.transform(q_t)
        
        metadata = {
            'dataset_path': str(config.DATASET_PATH),
            'categories': config.EXPECTED_CATEGORIES,
            'random_seed': config.RANDOM_SEED,
            'glcm_params': {'distance': config.GLCM_DISTANCE, 'angles_deg': config.GLCM_ANGLES_DEG},
            'normalization_parameters': {
                'color_min': c_norm.min_val.tolist(), 'color_max': c_norm.max_val.tolist(),
                'texture_min': t_norm.min_val.tolist(), 'texture_max': t_norm.max_val.tolist()
            },
            'm3_weights': [config.M3_COLOR_WEIGHT, config.M3_TEXTURE_WEIGHT],
            'm4_weights': [config.M4_COLOR_WEIGHT, config.M4_TEXTURE_WEIGHT],
            'k_value': config.EVAL_K,
            'query_image_paths': [str(p.relative_to(config.DATASET_PATH)) for p in query_paths],
            'gallery_image_paths': [str(p.relative_to(config.DATASET_PATH)) for p in gallery_paths],
            'gallery_color_features': gal_c,
            'gallery_texture_features': gal_t,
            'query_color_features': q_c,
            'query_texture_features': q_t,
            'normalized_gallery_color': norm_gal_c,
            'normalized_gallery_texture': norm_gal_t,
            'normalized_query_color': norm_q_c,
            'normalized_query_texture': norm_q_t
        }
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"[SUCCESS] Features and metadata saved to: {pkl_path}")

    # Evaluate Baselines M1-M4
    print("\n4. Evaluating baseline CBIR models (M1-M4)...")
    norm_gal_c = metadata['normalized_gallery_color']
    norm_gal_t = metadata['normalized_gallery_texture']
    norm_q_c = metadata['normalized_query_color']
    norm_q_t = metadata['normalized_query_texture']
    
    gallery_paths = [config.DATASET_PATH / p for p in metadata['gallery_image_paths']]
    query_paths = [config.DATASET_PATH / p for p in metadata['query_image_paths']]
    gallery_labels = [Path(p).parent.name for p in gallery_paths]
    query_labels = [Path(p).parent.name for p in query_paths]
    
    baseline_results = {}
    for model_name in ['M1', 'M2', 'M3', 'M4']:
        gal_rep = get_representation(norm_gal_c, norm_gal_t, model_name)
        q_rep = get_representation(norm_q_c, norm_q_t, model_name)
        
        all_ranked_indices = []
        retrieval_times = []
        for q_idx in range(len(query_paths)):
            t0 = time.perf_counter()
            ranked_idx, dists = retrieve_images(q_rep[q_idx], gal_rep)
            elapsed = time.perf_counter() - t0
            all_ranked_indices.append(ranked_idx)
            retrieval_times.append(elapsed)
            
        metrics = evaluate_all(all_ranked_indices, gallery_labels, query_labels, retrieval_times, k=config.EVAL_K)
        baseline_results[model_name] = metrics
        print(f"   [{model_name}] Precision@10: {metrics['mean_precision']:.4f} | Recall@10: {metrics['mean_recall']:.4f} | F1: {metrics['mean_f1']:.4f} | mAP: {metrics['mAP']:.4f}")
        
    return baseline_results, metadata

def run_part2(baseline_results=None, metadata=None):
    """
    Executes Part 2: MLP metric learning training, adaptive weight inference for M5,
    ablation analysis, and comparative visual figure generation.
    """
    print("\n" + "="*65)
    print("  STAGE 2: MLP ADAPTIVE WEIGHTING (M5) TRAINING & EVALUATION")
    print("="*65)
    
    if metadata is None:
        with open(config.FEATURES_PKL_PATH, 'rb') as f:
            metadata = pickle.load(f)
            
    # 1. Train MLP on gallery features
    print("\n1. Training PyTorch MLP on gallery using Triplet Margin Loss...")
    model, split_info, history = train_mlp(config.FEATURES_PKL_PATH, seed=config.PART2_SEED)
    
    # 2. Predict adaptive weights for the 100 queries
    print("\n2. Generating adaptive feature weights for 100 query images...")
    model.eval()
    norm_gal_c = metadata['normalized_gallery_color']
    norm_gal_t = metadata['normalized_gallery_texture']
    norm_q_c = metadata['normalized_query_color']
    norm_q_t = metadata['normalized_query_texture']
    
    gallery_paths = [config.DATASET_PATH / p for p in metadata['gallery_image_paths']]
    query_paths = [config.DATASET_PATH / p for p in metadata['query_image_paths']]
    gallery_labels = [Path(p).parent.name for p in gallery_paths]
    query_labels = [Path(p).parent.name for p in query_paths]
    
    combined_query_feats = np.hstack([norm_q_c, norm_q_t])
    with torch.no_grad():
        predicted_weights = model(torch.tensor(combined_query_feats, dtype=torch.float32)).numpy()
        
    q_color_weights = predicted_weights[:, 0]
    q_texture_weights = predicted_weights[:, 1]
    
    # 3. Perform Adaptive Distance Retrieval (M5)
    print("\n3. Evaluating Model M5 with adaptive distance metric...")
    all_ranked_indices = []
    retrieval_times = []
    m5_retrievals = []
    
    for q_idx in range(len(query_paths)):
        q_c = norm_q_c[q_idx]
        q_t = norm_q_t[q_idx]
        w_c = q_color_weights[q_idx]
        w_t = q_texture_weights[q_idx]
        
        t0 = time.perf_counter()
        dc_sq = np.sum((norm_gal_c - q_c)**2, axis=1)
        dt_sq = np.sum((norm_gal_t - q_t)**2, axis=1)
        dists = np.sqrt(w_c * dc_sq + w_t * dt_sq)
        ranked_idx = np.argsort(dists)
        elapsed = time.perf_counter() - t0
        
        all_ranked_indices.append(ranked_idx)
        retrieval_times.append(elapsed)
        
        top10_paths = [gallery_paths[idx] for idx in ranked_idx[:config.EVAL_K]]
        top10_labels = [gallery_labels[idx] for idx in ranked_idx[:config.EVAL_K]]
        m5_retrievals.append((top10_paths, top10_labels))
        
    m5_metrics = evaluate_all(all_ranked_indices, gallery_labels, query_labels, retrieval_times, k=config.EVAL_K)
    print(f"   [M5] Precision@10: {m5_metrics['mean_precision']:.4f} | Recall@10: {m5_metrics['mean_recall']:.4f} | F1: {m5_metrics['mean_f1']:.4f} | mAP: {m5_metrics['mAP']:.4f}")
    
    # 4. Weight Statistics & Ablation
    mean_w_c, std_w_c = float(np.mean(q_color_weights)), float(np.std(q_color_weights))
    mean_w_t, std_w_t = float(np.mean(q_texture_weights)), float(np.std(q_texture_weights))
    min_w_c, max_w_c = float(np.min(q_color_weights)), float(np.max(q_color_weights))
    min_w_t, max_w_t = float(np.min(q_texture_weights)), float(np.max(q_texture_weights))
    
    categories = sorted(config.EXPECTED_CATEGORIES)
    cat_avg_color_w, cat_avg_texture_w = [], []
    cat_weights_summary = {}
    for cat in categories:
        cat_idx = [i for i, lbl in enumerate(query_labels) if lbl == cat]
        avg_c = float(np.mean(q_color_weights[cat_idx]))
        avg_t = float(np.mean(q_texture_weights[cat_idx]))
        cat_avg_color_w.append(avg_c)
        cat_avg_texture_w.append(avg_t)
        cat_weights_summary[cat] = {'avg_color': avg_c, 'avg_texture': avg_t}
        
    # 5. Full M1-M5 Comparison Table
    if baseline_results is None:
        baseline_results = {
            'M1': {'mean_precision': 0.6200, 'mean_recall': 0.0689, 'mean_f1': 0.1240, 'mAP': 0.4048, 'mean_time': 0.000134},
            'M2': {'mean_precision': 0.4680, 'mean_recall': 0.0520, 'mean_f1': 0.0936, 'mAP': 0.3508, 'mean_time': 0.000087},
            'M3': {'mean_precision': 0.6990, 'mean_recall': 0.0777, 'mean_f1': 0.1398, 'mAP': 0.4652, 'mean_time': 0.000103},
            'M4': {'mean_precision': 0.6730, 'mean_recall': 0.0748, 'mean_f1': 0.1346, 'mAP': 0.4340, 'mean_time': 0.000109}
        }
        
    all_results = {**baseline_results, 'M5': m5_metrics}
    table_str = generate_comparison_table(all_results)
    print("\n" + table_str + "\n")
    
    # 6. Generate Figures
    print("4. Exporting plots and publication figures...")
    plot_m1_m5_metric_comparison(
        {m: r['mAP'] for m, r in all_results.items()},
        'mAP', 'Mean Average Precision (mAP)', 'mAP Comparison (M1-M5)',
        config.FIGURES_DIR / 'fig2_model_performance_comparison.png'
    )
    plot_weight_distributions(q_color_weights, q_texture_weights, config.FIGURES_DIR)
    # Rename to standard naming
    if (config.FIGURES_DIR / "weight_distributions.png").exists():
        (config.FIGURES_DIR / "weight_distributions.png").replace(config.FIGURES_DIR / "fig3_weight_distributions.png")
        
    plot_category_wise_weights(categories, cat_avg_color_w, cat_avg_texture_w, config.FIGURES_DIR / 'fig4_category_weights.png')
    
    # Search grid for Beaches
    b_idx = query_labels.index("beaches")
    b_path = query_paths[b_idx]
    b_lbl = query_labels[b_idx]
    
    d_c_sq = np.sum((norm_gal_c - norm_q_c[b_idx])**2, axis=1)
    d_t_sq = np.sum((norm_gal_t - norm_q_t[b_idx])**2, axis=1)
    m3_dists = np.sqrt(0.5 * d_c_sq + 0.5 * d_t_sq)
    m3_ranked = np.argsort(m3_dists)
    m3_paths = [gallery_paths[idx] for idx in m3_ranked[:config.EVAL_K]]
    m3_lbls = [gallery_labels[idx] for idx in m3_ranked[:config.EVAL_K]]
    
    plot_m3_vs_m5_comparison_grid(
        b_path, b_lbl, (m3_paths, m3_lbls), m5_retrievals[b_idx],
        q_color_weights[b_idx], q_texture_weights[b_idx],
        config.FIGURES_DIR / 'fig5_query_m3_vs_m5_beaches.png'
    )
    
    # Also generate the IEEE Grouped Bar Chart
    create_ieee_chart()
    
    # 7. Save Part 2 Results Cache
    part2_metadata = {
        'split_info': split_info,
        'training_history': history,
        'query_color_weights': q_color_weights.tolist(),
        'query_texture_weights': q_texture_weights.tolist(),
        'm5_metrics': m5_metrics,
        'ablation_statistics': {
            'mean_color_w': mean_w_c, 'mean_texture_w': mean_w_t,
            'std_color_w': std_w_c, 'std_texture_w': std_w_t,
            'min_color_w': min_w_c, 'max_color_w': max_w_c,
            'min_texture_w': min_w_t, 'max_texture_w': max_w_t,
            'category_wise_averages': cat_weights_summary
        }
    }
    with open(config.PART2_RESULTS_PATH, 'wb') as f:
        pickle.dump(part2_metadata, f)
        
    print(f"[SUCCESS] Part 2 completed. Figures exported to: {config.FIGURES_DIR}")
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Corel-1K CBIR Pipeline Runner")
    parser.add_argument(
        "--stage", 
        choices=["all", "part1", "part2", "figures", "sanity"], 
        default="all", 
        help="Pipeline stage to execute (default: all)"
    )
    parser.add_argument(
        "--force-extract", 
        action="store_true", 
        help="Force re-extraction of image features"
    )
    args = parser.parse_args()
    
    if args.stage == "sanity":
        from sanity_check import main as run_sanity
        run_sanity()
    elif args.stage == "figures":
        create_ieee_chart()
        print(f"[SUCCESS] Figures regenerated in: {config.FIGURES_DIR}")
    elif args.stage == "part1":
        run_part1(force_reextract=args.force_extract)
    elif args.stage == "part2":
        run_part2()
    else:  # "all"
        base_res, meta = run_part1(force_reextract=args.force_extract)
        run_part2(baseline_results=base_res, metadata=meta)

if __name__ == "__main__":
    main()
