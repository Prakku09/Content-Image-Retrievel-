import numpy as np
from pathlib import Path
import time
import pickle
from tqdm import tqdm
import config
from dataset_loader import load_dataset
from feature_extractor import extract_all_features
from normalization import MinMaxNormalizer
from retrieval import get_representation, retrieve_images
from evaluation import evaluate_all
from visualization import plot_metric_comparison, plot_query_comparison, generate_comparison_table

def main():
    print("=== STARTING COREL-1K FEATURE EXTRACTION AND BASELINE CBIR EXPERIMENT (PART 1) ===")
    
    # 1. Load Dataset & Split
    print("\nLoading dataset and generating deterministic splits...")
    gallery_paths, gallery_labels, query_paths, query_labels = load_dataset()
    print(f"Dataset Split Loaded:")
    print(f"  - Gallery set: {len(gallery_paths)} images (90 per category)")
    print(f"  - Query set  : {len(query_paths)} images (10 per category)")
    
    # 2. Feature Extraction
    print("\nExtracting HSV color (9-dim) and GLCM texture (4-dim) features...")
    
    gallery_color_feats = []
    gallery_texture_feats = []
    print("Extracting gallery features:")
    for path in tqdm(gallery_paths, desc="Gallery Features"):
        color, texture = extract_all_features(path)
        gallery_color_feats.append(color)
        gallery_texture_feats.append(texture)
        
    query_color_feats = []
    query_texture_feats = []
    print("Extracting query features:")
    for path in tqdm(query_paths, desc="Query Features"):
        color, texture = extract_all_features(path)
        query_color_feats.append(color)
        query_texture_feats.append(texture)
        
    gallery_color_feats = np.array(gallery_color_feats, dtype=np.float32)
    gallery_texture_feats = np.array(gallery_texture_feats, dtype=np.float32)
    query_color_feats = np.array(query_color_feats, dtype=np.float32)
    query_texture_feats = np.array(query_texture_feats, dtype=np.float32)
    
    # 3. Normalization (Fit on Gallery Only)
    print("\nNormalizing features using Min-Max scaling fitted on gallery features...")
    color_norm = MinMaxNormalizer()
    texture_norm = MinMaxNormalizer()
    
    norm_gallery_color = color_norm.fit_transform(gallery_color_feats)
    norm_query_color = color_norm.transform(query_color_feats)
    
    norm_gallery_texture = texture_norm.fit_transform(gallery_texture_feats)
    norm_query_texture = texture_norm.transform(query_texture_feats)
    
    # 4. Save Structured Features and Metadata
    print("\nSaving extracted features and metadata for Part 2...")
    metadata = {
        'dataset_path': str(config.DATASET_PATH),
        'categories': config.EXPECTED_CATEGORIES,
        'random_seed': config.RANDOM_SEED,
        'glcm_params': {
            'distance': config.GLCM_DISTANCE,
            'angles_deg': config.GLCM_ANGLES_DEG
        },
        'normalization_parameters': {
            'color_min': color_norm.min_val.tolist(),
            'color_max': color_norm.max_val.tolist(),
            'texture_min': texture_norm.min_val.tolist(),
            'texture_max': texture_norm.max_val.tolist()
        },
        'm3_weights': [config.M3_COLOR_WEIGHT, config.M3_TEXTURE_WEIGHT],
        'm4_weights': [config.M4_COLOR_WEIGHT, config.M4_TEXTURE_WEIGHT],
        'k_value': config.EVAL_K,
        'query_image_paths': [str(p.relative_to(config.DATASET_PATH)) for p in query_paths],
        'gallery_image_paths': [str(p.relative_to(config.DATASET_PATH)) for p in gallery_paths],
        'gallery_color_features': gallery_color_feats,
        'gallery_texture_features': gallery_texture_feats,
        'query_color_features': query_color_feats,
        'query_texture_features': query_texture_feats,
        'normalized_gallery_color': norm_gallery_color,
        'normalized_gallery_texture': norm_gallery_texture,
        'normalized_query_color': norm_query_color,
        'normalized_query_texture': norm_query_texture
    }
    
    pkl_path = config.OUTPUT_DIR / "corel1k_features.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Features and metadata saved permanently to: {pkl_path}")
    
    # 5. Evaluate Baseline Models
    print("\nEvaluating baseline CBIR models M1, M2, M3, M4...")
    eval_results = {}
    visual_retrieval_data = {q_idx: {} for q_idx in range(len(query_paths))}
    
    for model_name in ['M1', 'M2', 'M3', 'M4']:
        print(f"  Running model {model_name}...")
        
        # Build representations
        gal_rep = get_representation(norm_gallery_color, norm_gallery_texture, model_name)
        q_rep = get_representation(norm_query_color, norm_query_texture, model_name)
        
        # Retrieve
        all_ranked_indices = []
        retrieval_times = []
        
        for q_idx in range(len(query_paths)):
            start_time = time.perf_counter()
            ranked_idx, dists = retrieve_images(q_rep[q_idx], gal_rep)
            elapsed = time.perf_counter() - start_time
            
            all_ranked_indices.append(ranked_idx)
            retrieval_times.append(elapsed)
            
            # Store top-10 retrieval paths and labels for visualization mapping
            top10_paths = [gallery_paths[idx] for idx in ranked_idx[:config.EVAL_K]]
            top10_labels = [gallery_labels[idx] for idx in ranked_idx[:config.EVAL_K]]
            visual_retrieval_data[q_idx][model_name] = (top10_paths, top10_labels)
            
        # Compute mean evaluation metrics
        metrics = evaluate_all(
            all_ranked_indices, 
            gallery_labels, 
            query_labels, 
            retrieval_times, 
            k=config.EVAL_K
        )
        eval_results[model_name] = metrics
        print(f"    Precision@10: {metrics['mean_precision']:.4f} | Recall@10: {metrics['mean_recall']:.4f} | F1: {metrics['mean_f1']:.4f} | mAP: {metrics['mAP']:.4f} | Time: {metrics['mean_time']*1000:.3f} ms")
        
    # 6. Generate Comparison Table & Plots
    print("\nGenerating evaluation visualizations and plots...")
    
    # Text Table
    comparison_table = generate_comparison_table(eval_results)
    print("\n" + comparison_table + "\n")
    
    # Save individual plots
    plot_metric_comparison(
        {m: r['mean_precision'] for m, r in eval_results.items()},
        'Precision', 'Mean Precision@10', 'Precision@10 Comparison for M1-M4',
        config.PLOTS_DIR / 'precision_comparison.png'
    )
    
    plot_metric_comparison(
        {m: r['mean_recall'] for m, r in eval_results.items()},
        'Recall', 'Mean Recall@10', 'Recall@10 Comparison for M1-M4',
        config.PLOTS_DIR / 'recall_comparison.png'
    )
    
    plot_metric_comparison(
        {m: r['mean_f1'] for m, r in eval_results.items()},
        'F1-score', 'Mean F1-score@10', 'F1-score@10 Comparison for M1-M4',
        config.PLOTS_DIR / 'f1_comparison.png'
    )
    
    plot_metric_comparison(
        {m: r['mAP'] for m, r in eval_results.items()},
        'mAP', 'Mean Average Precision (mAP)', 'mAP (Full Gallery) Comparison for M1-M4',
        config.PLOTS_DIR / 'map_comparison.png'
    )
    
    plot_metric_comparison(
        {m: r['mean_time'] for m, r in eval_results.items()},
        'Retrieval Time', 'Average Retrieval Time (seconds)', 'Average Retrieval Time Comparison for M1-M4',
        config.PLOTS_DIR / 'time_comparison.png'
    )
    
    # 7. Generate Example Queries Comparisons
    print("\nGenerating example query retrieval grids...")
    visual_categories = ["beaches", "dinosaurs", "flowers"]
    for cat in visual_categories:
        # Find the first query index for this category
        q_idx = query_labels.index(cat)
        q_path = query_paths[q_idx]
        q_label = query_labels[q_idx]
        
        plot_query_comparison(
            q_path,
            q_label,
            visual_retrieval_data[q_idx],
            config.PLOTS_DIR / f'query_{cat}.png'
        )
        print(f"  Saved visual grid for '{cat}' query to: output/plots/query_{cat}.png")
        
    # 8. Write Experiment Report
    print("\nCompiling experiment report...")
    report_path = config.OUTPUT_DIR / "part1_experiment_report.md"
    
    report_content = f"""# Corel-1K CBIR Pipeline Experiment Report (Part 1)

This report details the execution and baseline evaluation of the Content-Based Image Retrieval (CBIR) pipeline on the Corel-1K dataset.

## Experimental Setup

- **Dataset Path**: `{config.DATASET_PATH}`
- **Random Seed**: `{config.RANDOM_SEED}`
- **Query/Gallery Split**: 10 query images per category (100 total) and 90 gallery images per category (900 total)
- **Features Extracted**:
  - **HSV Color (9-dim)**: Channel mean, standard deviation, and skewness for H, S, V.
  - **GLCM Texture (4-dim)**: Contrast, correlation, energy, and homogeneity averaged over 4 angles (0, 45, 90, 135 degrees) with distance = 1.
- **Normalization**: Min-Max normalization fitted strictly on the 900 gallery features.
- **Fusion Weighting**:
  - **M3 (Equal Fusion)**: Colour Weight = {config.M3_COLOR_WEIGHT}, Texture Weight = {config.M3_TEXTURE_WEIGHT}
  - **M4 (Fixed Fusion)**: Colour Weight = {config.M4_COLOR_WEIGHT}, Texture Weight = {config.M4_TEXTURE_WEIGHT} (Fixed baseline, no claims of optimality)
- **Evaluation**: Evaluated at K = 10 for Precision, Recall, and F1. Mean Average Precision (mAP) computed over the complete 900-image gallery.

## Evaluation Results

{comparison_table}

## Analysis of Baseline Performance

- **M1 (Colour Only)**: Performs feature retrieval focusing purely on color distributions in HSV space.
- **M2 (Texture Only)**: Performs feature retrieval focusing on spatial gray-level variations.
- **M3 (Equal Fusion)**: Fuses both color and texture with equal weights, balancing both descriptors.
- **M4 (Fixed Weight Fusion)**: Combines representations using a fixed weighting scheme of {config.M4_COLOR_WEIGHT} colour and {config.M4_TEXTURE_WEIGHT} texture.

## Visual Artifacts

The following comparison plots have been generated and saved to the `output/plots/` directory:
- [Precision comparison plot](file:///{config.PLOTS_DIR.as_posix()}/precision_comparison.png)
- [Recall comparison plot](file:///{config.PLOTS_DIR.as_posix()}/recall_comparison.png)
- [F1-score comparison plot](file:///{config.PLOTS_DIR.as_posix()}/f1_comparison.png)
- [mAP comparison plot](file:///{config.PLOTS_DIR.as_posix()}/map_comparison.png)
- [Retrieval-time comparison plot](file:///{config.PLOTS_DIR.as_posix()}/time_comparison.png)

Additionally, visual retrieval examples showing M1-M4 side-by-side rankings for queries in different categories have been saved:
- [Beaches Query Comparison](file:///{config.PLOTS_DIR.as_posix()}/query_beaches.png)
- [Dinosaurs Query Comparison](file:///{config.PLOTS_DIR.as_posix()}/query_dinosaurs.png)
- [Flowers Query Comparison](file:///{config.PLOTS_DIR.as_posix()}/query_flowers.png)
"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Experiment report compiled and saved to: {report_path}")
    print("\n=== EXPERIMENT (PART 1) COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
