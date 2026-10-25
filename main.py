"""
Unified Main Entry Point for Corel-1K CBIR Research Pipeline
End-to-End Execution of Baseline (M1-M4) and Proposed AI Adaptive Weighting (M5).
"""

import argparse
import time
import pickle
import hashlib
import json
import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Import pipeline packages
import config
from data.dataset_loader import load_dataset
from features.feature_extractor import extract_all_features
from features.normalization import MinMaxNormalizer
from model.mlp_trainer import train_mlp
from retrieval.retrieval import (
    get_representation,
    retrieve_images,
    validate_fusion_weights,
    weighted_euclidean_distances,
)
from retrieval.evaluation import evaluate_all
from visualization.visualization import generate_comparison_table
from visualization.visualize_part2 import (
    plot_m1_m5_metric_comparison,
    plot_weight_distributions,
    plot_category_wise_weights,
    plot_m3_vs_m5_comparison_grid
)


def _feature_fingerprint(gallery_paths, query_paths):
    """Hash source image contents and feature/split parameters for cache validation."""
    feature_config = {
        'schema_version': 2,
        'categories': config.EXPECTED_CATEGORIES,
        'random_seed': config.RANDOM_SEED,
        'gallery_per_category': config.GALLERY_IMAGES_PER_CAT,
        'query_per_category': config.QUERY_IMAGES_PER_CAT,
        'color_feature_dim': config.COLOR_FEATURE_DIM,
        'texture_feature_dim': config.TEXTURE_FEATURE_DIM,
        'glcm_distance': config.GLCM_DISTANCE,
        'glcm_angles_deg': config.GLCM_ANGLES_DEG,
        'image_extensions': sorted(config.IMAGE_EXTENSIONS),
        'feature_extractor': 'hsv-moments-glcm-v1',
        'feature_sources_sha256': _feature_source_fingerprints(),
    }
    digest = hashlib.sha256(json.dumps(feature_config, sort_keys=True).encode('utf-8'))
    all_paths = sorted(gallery_paths + query_paths)
    for path in all_paths:
        relative_path = path.relative_to(config.DATASET_PATH).as_posix()
        digest.update(relative_path.encode('utf-8'))
        file_digest = hashlib.sha256()
        with open(path, 'rb') as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b''):
                file_digest.update(chunk)
        digest.update(file_digest.digest())
    return digest.hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_source_fingerprints():
    source_root = Path(__file__).resolve().parent
    return {
        'feature_extractor.py': _sha256_file(
            source_root / 'features' / 'feature_extractor.py'
        ),
        'normalization.py': _sha256_file(
            source_root / 'features' / 'normalization.py'
        ),
    }


def _relative_paths(paths):
    return [path.relative_to(config.DATASET_PATH).as_posix() for path in paths]


def _feature_cache_matches(metadata, fingerprint, gallery_paths, query_paths):
    if not isinstance(metadata, dict):
        return False
    if metadata.get('feature_cache_schema_version') != 2:
        return False
    if metadata.get('feature_fingerprint') != fingerprint:
        return False
    if metadata.get('gallery_image_paths') != _relative_paths(gallery_paths):
        return False
    if metadata.get('query_image_paths') != _relative_paths(query_paths):
        return False

    expected_shapes = {
        'gallery_color_features': (len(gallery_paths), config.COLOR_FEATURE_DIM),
        'gallery_texture_features': (len(gallery_paths), config.TEXTURE_FEATURE_DIM),
        'query_color_features': (len(query_paths), config.COLOR_FEATURE_DIM),
        'query_texture_features': (len(query_paths), config.TEXTURE_FEATURE_DIM),
        'normalized_gallery_color': (len(gallery_paths), config.COLOR_FEATURE_DIM),
        'normalized_gallery_texture': (len(gallery_paths), config.TEXTURE_FEATURE_DIM),
        'normalized_query_color': (len(query_paths), config.COLOR_FEATURE_DIM),
        'normalized_query_texture': (len(query_paths), config.TEXTURE_FEATURE_DIM),
    }
    for key, shape in expected_shapes.items():
        values = metadata.get(key)
        if not isinstance(values, np.ndarray) or values.shape != shape:
            return False
        if not np.all(np.isfinite(values)):
            return False

    normalization = metadata.get('normalization_parameters', {})
    try:
        color_normalizer = MinMaxNormalizer(
            normalization['color_min'], normalization['color_max']
        )
        texture_normalizer = MinMaxNormalizer(
            normalization['texture_min'], normalization['texture_max']
        )
        expected_normalized = (
            color_normalizer.transform(metadata['gallery_color_features']),
            color_normalizer.transform(metadata['query_color_features']),
            texture_normalizer.transform(metadata['gallery_texture_features']),
            texture_normalizer.transform(metadata['query_texture_features']),
        )
    except (KeyError, TypeError, ValueError):
        return False

    actual_normalized = (
        metadata['normalized_gallery_color'],
        metadata['normalized_query_color'],
        metadata['normalized_gallery_texture'],
        metadata['normalized_query_texture'],
    )
    return all(
        np.allclose(expected, actual, rtol=1e-6, atol=1e-7)
        for expected, actual in zip(expected_normalized, actual_normalized)
    )

def run_part1(force_reextract=False):
    """
    Executes Part 1: Dataset loading, feature extraction (HSV + GLCM), Min-Max normalization,
    and baseline retrieval evaluation (M1, M2, M3, M4).
    """
    print("\n" + "="*65)
    print("  STAGE 1: FEATURE EXTRACTION & BASELINE EVALUATION (M1-M4)")
    print("="*65)
    
    pkl_path = config.FEATURES_PKL_PATH
    print("\n1. Loading dataset and generating deterministic splits...")
    gallery_paths, gallery_labels, query_paths, query_labels = load_dataset()
    print(f"   Gallery images: {len(gallery_paths)}")
    print(f"   Query images  : {len(query_paths)}")
    fingerprint = _feature_fingerprint(gallery_paths, query_paths)

    metadata = None
    if pkl_path.exists() and not force_reextract:
        with open(pkl_path, 'rb') as f:
            cached_metadata = pickle.load(f)
        if _feature_cache_matches(
            cached_metadata, fingerprint, gallery_paths, query_paths
        ):
            metadata = cached_metadata
            metadata['feature_cache_status'] = 'verified_cache_reused'
            print(f"[CACHE] Verified feature cache: {pkl_path}")
        else:
            print("[CACHE] Feature cache provenance or contents do not match; re-extracting.")

    if metadata is None:
        print("\n2. Extracting HSV color moments (9-dim) and GLCM texture (4-dim)...")
        gal_c, gal_t = [], []
        for path in tqdm(gallery_paths, desc="Gallery Features"):
            color, texture = extract_all_features(path)
            gal_c.append(color)
            gal_t.append(texture)

        q_c, q_t = [], []
        for path in tqdm(query_paths, desc="Query Features"):
            color, texture = extract_all_features(path)
            q_c.append(color)
            q_t.append(texture)

        gal_c = np.asarray(gal_c, dtype=np.float32)
        gal_t = np.asarray(gal_t, dtype=np.float32)
        q_c = np.asarray(q_c, dtype=np.float32)
        q_t = np.asarray(q_t, dtype=np.float32)
        if not all(np.all(np.isfinite(values)) for values in (gal_c, gal_t, q_c, q_t)):
            raise ValueError("Feature extraction produced NaN or infinite values.")

        print("\n3. Fitting Min-Max Normalizer on gallery distribution...")
        c_norm = MinMaxNormalizer()
        t_norm = MinMaxNormalizer()
        norm_gal_c = c_norm.fit_transform(gal_c)
        norm_q_c = c_norm.transform(q_c)
        norm_gal_t = t_norm.fit_transform(gal_t)
        norm_q_t = t_norm.transform(q_t)

        metadata = {
            'dataset_path': str(config.DATASET_PATH),
            'categories': list(config.EXPECTED_CATEGORIES),
            'random_seed': config.RANDOM_SEED,
            'feature_fingerprint': fingerprint,
            'feature_cache_schema_version': 2,
            'feature_cache_status': 'recomputed',
            'glcm_params': {
                'distance': config.GLCM_DISTANCE,
                'angles_deg': list(config.GLCM_ANGLES_DEG),
            },
            'normalization_parameters': {
                'color_min': c_norm.min_val.tolist(),
                'color_max': c_norm.max_val.tolist(),
                'texture_min': t_norm.min_val.tolist(),
                'texture_max': t_norm.max_val.tolist(),
            },
            'm3_weights': [config.M3_COLOR_WEIGHT, config.M3_TEXTURE_WEIGHT],
            'm4_weights': [config.M4_COLOR_WEIGHT, config.M4_TEXTURE_WEIGHT],
            'k_value': config.EVAL_K,
            'query_image_paths': _relative_paths(query_paths),
            'gallery_image_paths': _relative_paths(gallery_paths),
            'gallery_color_features': gal_c,
            'gallery_texture_features': gal_t,
            'query_color_features': q_c,
            'query_texture_features': q_t,
            'normalized_gallery_color': norm_gal_c,
            'normalized_gallery_texture': norm_gal_t,
            'normalized_query_color': norm_q_c,
            'normalized_query_texture': norm_q_t,
        }
        with open(pkl_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"[SUCCESS] Fresh features and metadata saved to: {pkl_path}")

    # Evaluate Baselines M1-M4
    print("\n4. Evaluating baseline CBIR models (M1-M4)...")
    norm_gal_c = metadata['normalized_gallery_color']
    norm_gal_t = metadata['normalized_gallery_texture']
    norm_q_c = metadata['normalized_query_color']
    norm_q_t = metadata['normalized_query_texture']
    
    gallery_labels = list(gallery_labels)
    query_labels = list(query_labels)
    
    baseline_results = {}
    baseline_rankings = {}
    baseline_times = {}
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
        baseline_rankings[model_name] = np.asarray(all_ranked_indices, dtype=np.int32)
        baseline_times[model_name] = retrieval_times
        print(f"   [{model_name}] Precision@10: {metrics['mean_precision']:.4f} | Recall@10: {metrics['mean_recall']:.4f} | F1: {metrics['mean_f1']:.4f} | mAP: {metrics['mAP']:.4f}")

    metadata['gallery_labels'] = gallery_labels
    metadata['query_labels'] = query_labels
    metadata['baseline_rankings'] = baseline_rankings
    metadata['baseline_retrieval_times'] = baseline_times
    return baseline_results, metadata

def _evaluate_weighted_model(weights, norm_gal_c, norm_gal_t, norm_q_c, norm_q_t,
                             gallery_paths, gallery_labels, query_labels):
    rankings = []
    retrieval_times = []
    top_results = []
    for query_index, weight_pair in enumerate(weights):
        start = time.perf_counter()
        distances = weighted_euclidean_distances(
            norm_q_c[query_index],
            norm_gal_c,
            norm_q_t[query_index],
            norm_gal_t,
            weight_pair,
        )
        ranked_indices = np.argsort(distances)
        retrieval_times.append(time.perf_counter() - start)
        rankings.append(ranked_indices)
        top_indices = ranked_indices[:config.EVAL_K]
        top_results.append((
            [gallery_paths[index] for index in top_indices],
            [gallery_labels[index] for index in top_indices],
        ))
    metrics = evaluate_all(
        rankings, gallery_labels, query_labels, retrieval_times, k=config.EVAL_K
    )
    return metrics, np.asarray(rankings, dtype=np.int32), retrieval_times, top_results


def _write_experiment_report(results, weight_statistics, category_weights, mean_weights,
                             experiment_config):
    metric_rows = []
    for model_name in ('M1', 'M2', 'M3', 'M4', 'M5', 'M6'):
        metrics = results[model_name]
        metric_rows.append(
            f"| {model_name} | {metrics['mean_precision']:.4f} | "
            f"{metrics['mean_recall']:.4f} | {metrics['mean_f1']:.4f} | "
            f"{metrics['mAP']:.4f} | {metrics['mean_time'] * 1000:.3f} ms |"
        )
    category_rows = [
        f"| {category} | {weights['avg_color']:.4f} | {weights['avg_texture']:.4f} |"
        for category, weights in category_weights.items()
    ]
    metric_deltas = {
        'Precision@10': results['M5']['mean_precision'] - results['M6']['mean_precision'],
        'Recall@10': results['M5']['mean_recall'] - results['M6']['mean_recall'],
        'F1@10': results['M5']['mean_f1'] - results['M6']['mean_f1'],
        'mAP': results['M5']['mAP'] - results['M6']['mAP'],
    }
    comparison = ', '.join(
        f"{name} {'higher' if difference > 0 else 'lower' if difference < 0 else 'equal'} "
        f"by {abs(difference):.4f}"
        for name, difference in metric_deltas.items()
    )
    report = f"""# Corel-1K CBIR Experiment Report (M1-M6)

## Run provenance and protocol

- Run status: freshly evaluated; feature cache status: `{experiment_config['feature_cache_status']}`.
- Dataset fingerprint: `{experiment_config['feature_fingerprint']}`.
- Split: {experiment_config['gallery_count']} gallery and {experiment_config['query_count']} held-out queries; seed {experiment_config['random_seed']}.
- MLP training/validation split: 700/200 images drawn only from the gallery; held-out queries were used only for final evaluation and descriptive weight summaries.
- Normalization: min-max bounds fitted on gallery features and applied to gallery/query features.
- Metrics: Precision@10, Recall@10, F1@10, full-gallery mAP and mean distance-plus-sort time. Retrieval timing excludes feature extraction, MLP inference, image I/O and plotting; model inference is done outside the timed retrieval loop.

## Distance definitions

- M1/M2: Euclidean distance on colour-only / texture-only normalized features.
- M3/M4: weighted feature concatenation, so squared Euclidean distance is `wc^2 * dc^2 + wt^2 * dt^2`. M3 uses ({config.M3_COLOR_WEIGHT}, {config.M3_TEXTURE_WEIGHT}); M4 uses ({config.M4_COLOR_WEIGHT}, {config.M4_TEXTURE_WEIGHT}).
- M5/M6: direct distance-level weighting, `sqrt(wc * dc^2 + wt * dt^2)`. M5 uses a query-specific Softmax pair; M6 applies the M5 mean pair ({mean_weights[0]:.6f}, {mean_weights[1]:.6f}) to every query. This keeps the original M3/M4 definitions intact and makes M6 a direct-formula control for M5.

M3/M4 coefficients therefore do not have the same interpretation as the M5/M6 coefficients: they are squared in the induced distance. The runs intentionally preserve that methodological difference rather than silently harmonizing historical baselines.
For rankings, M3's squared coefficients differ from direct 0.5/0.5 weighting only by a common scale. M4's squared coefficients have normalized relative weights of about 0.845/0.155, rather than 0.7/0.3. M6 uses the direct M5 formula to isolate the effect of per-query weight variation.

## Metrics

| Model | Precision@10 | Recall@10 | F1@10 | mAP | Mean retrieval time |
|---|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

M5 compared with M6: {comparison}. These are descriptive metric differences; no statistical significance is claimed.

## M5 query-weight distribution

| Weight | Mean | Population std. dev. | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Colour | {weight_statistics['mean_color_w']:.6f} | {weight_statistics['std_color_w']:.6f} | {weight_statistics['min_color_w']:.6f} | {weight_statistics['max_color_w']:.6f} |
| Texture | {weight_statistics['mean_texture_w']:.6f} | {weight_statistics['std_texture_w']:.6f} | {weight_statistics['min_texture_w']:.6f} | {weight_statistics['max_texture_w']:.6f} |

The M5 weights vary across queries: {weight_statistics['weights_vary_across_queries']}. The variation is descriptive and is not, by itself, evidence of improved retrieval.

### Category mean weights

| Category | Mean colour | Mean texture |
|---|---:|---:|
{chr(10).join(category_rows)}

## Reproducibility artifacts

The run saves the complete rankings, per-query weights, experiment configuration, MLP checkpoint, training history, and metric summaries next to this report in `output/part2/`.
"""
    report_path = config.PART2_OUTPUT_DIR / 'part2_experiment_report.md'
    report_path.write_text(report, encoding='utf-8')
    print(f"Experiment report saved to: {report_path}")


def run_part2(baseline_results=None, metadata=None):
    """Train/infer M5, evaluate the M6 mean-weight control, and persist a full run."""
    print("\n" + "=" * 65)
    print("  STAGE 2: MLP ADAPTIVE WEIGHTING (M5) AND MEAN-WEIGHT CONTROL (M6)")
    print("=" * 65)

    if baseline_results is None or metadata is None:
        baseline_results, metadata = run_part1()

    print("\n1. Training or loading a provenance-verified MLP checkpoint...")
    model, split_info, history = train_mlp(
        config.FEATURES_PKL_PATH, seed=config.PART2_SEED
    )

    print("\n2. Predicting weights for the held-out queries...")
    model.eval()
    norm_gal_c = metadata['normalized_gallery_color']
    norm_gal_t = metadata['normalized_gallery_texture']
    norm_q_c = metadata['normalized_query_color']
    norm_q_t = metadata['normalized_query_texture']
    gallery_paths = [config.DATASET_PATH / path for path in metadata['gallery_image_paths']]
    query_paths = [config.DATASET_PATH / path for path in metadata['query_image_paths']]
    gallery_labels = metadata['gallery_labels']
    query_labels = metadata['query_labels']
    combined_query_feats = np.hstack([norm_q_c, norm_q_t])
    with torch.no_grad():
        predicted_weights = model(
            torch.tensor(combined_query_feats, dtype=torch.float32)
        ).cpu().numpy()
    predicted_weights = validate_fusion_weights(predicted_weights)
    if predicted_weights.shape != (len(query_paths), 2):
        raise ValueError("MLP prediction count does not match the held-out query count.")

    q_color_weights = predicted_weights[:, 0]
    q_texture_weights = predicted_weights[:, 1]

    print("\n3. Evaluating M5 and fixed mean-weight M6 with identical queries/gallery...")
    m5_metrics, m5_rankings, m5_times, m5_retrievals = _evaluate_weighted_model(
        predicted_weights, norm_gal_c, norm_gal_t, norm_q_c, norm_q_t,
        gallery_paths, gallery_labels, query_labels
    )
    mean_weights = validate_fusion_weights(np.mean(predicted_weights, axis=0))
    m6_weights = np.tile(mean_weights, (len(query_paths), 1))
    m6_metrics, m6_rankings, m6_times, _ = _evaluate_weighted_model(
        m6_weights, norm_gal_c, norm_gal_t, norm_q_c, norm_q_t,
        gallery_paths, gallery_labels, query_labels
    )

    all_results = {**baseline_results, 'M5': m5_metrics, 'M6': m6_metrics}
    print(f"   M5 mean weights: colour={mean_weights[0]:.6f}, texture={mean_weights[1]:.6f}")
    print(f"   [M5] Precision@10={m5_metrics['mean_precision']:.4f} | "
          f"Recall@10={m5_metrics['mean_recall']:.4f} | F1@10={m5_metrics['mean_f1']:.4f} | "
          f"mAP={m5_metrics['mAP']:.4f}")
    print(f"   [M6] Precision@10={m6_metrics['mean_precision']:.4f} | "
          f"Recall@10={m6_metrics['mean_recall']:.4f} | F1@10={m6_metrics['mean_f1']:.4f} | "
          f"mAP={m6_metrics['mAP']:.4f}")

    weight_statistics = {
        'mean_color_w': float(np.mean(q_color_weights)),
        'mean_texture_w': float(np.mean(q_texture_weights)),
        'std_color_w': float(np.std(q_color_weights, ddof=0)),
        'std_texture_w': float(np.std(q_texture_weights, ddof=0)),
        'min_color_w': float(np.min(q_color_weights)),
        'max_color_w': float(np.max(q_color_weights)),
        'min_texture_w': float(np.min(q_texture_weights)),
        'max_texture_w': float(np.max(q_texture_weights)),
        'weights_vary_across_queries': bool(
            np.std(q_color_weights, ddof=0) > 1e-6
            and np.std(q_texture_weights, ddof=0) > 1e-6
        ),
    }
    categories = sorted(config.EXPECTED_CATEGORIES)
    cat_avg_color_w, cat_avg_texture_w = [], []
    category_weights = {}
    for category in categories:
        category_indices = [
            index for index, label in enumerate(query_labels) if label == category
        ]
        if not category_indices:
            raise ValueError(f"No held-out query images found for category '{category}'.")
        avg_color = float(np.mean(q_color_weights[category_indices]))
        avg_texture = float(np.mean(q_texture_weights[category_indices]))
        cat_avg_color_w.append(avg_color)
        cat_avg_texture_w.append(avg_texture)
        category_weights[category] = {
            'avg_color': avg_color,
            'avg_texture': avg_texture,
        }
    weight_statistics['category_wise_averages'] = category_weights

    table_str = generate_comparison_table(all_results)
    print("\n" + table_str + "\n")

    print("4. Generating M1-M6 comparison and M5 weight plots...")
    plot_metrics = [
        ('mean_precision', 'Precision@10', 'Mean Precision@10', 'precision_comparison.png'),
        ('mean_recall', 'Recall@10', 'Mean Recall@10', 'recall_comparison.png'),
        ('mean_f1', 'F1@10', 'Mean F1-score@10', 'f1_comparison.png'),
        ('mAP', 'mAP', 'Mean Average Precision (mAP)', 'map_comparison.png'),
        ('mean_time', 'time', 'Mean retrieval time (ms)', 'time_comparison.png'),
    ]
    for key, metric_name, ylabel, filename in plot_metrics:
        plot_m1_m5_metric_comparison(
            {name: metrics[key] for name, metrics in all_results.items()},
            metric_name,
            ylabel,
            f"{ylabel} Comparison (M1-M6)",
            config.PLOTS_DIR / filename,
        )
    plot_m1_m5_metric_comparison(
        {name: metrics['mAP'] for name, metrics in all_results.items()},
        'mAP', 'Mean Average Precision (mAP)', 'Model Performance Comparison (M1-M6)',
        config.FIGURES_DIR / 'fig2_model_performance_comparison.png',
    )
    plot_weight_distributions(q_color_weights, q_texture_weights, config.PLOTS_DIR)
    shutil.copyfile(
        config.PLOTS_DIR / 'weight_distributions.png',
        config.FIGURES_DIR / 'fig3_weight_distributions.png',
    )
    plot_category_wise_weights(
        categories, cat_avg_color_w, cat_avg_texture_w,
        config.PLOTS_DIR / 'category_weights.png',
    )
    shutil.copyfile(
        config.PLOTS_DIR / 'category_weights.png',
        config.FIGURES_DIR / 'fig4_category_weights.png',
    )

    beaches_index = query_labels.index('beaches')
    m3_ranked = metadata['baseline_rankings']['M3'][beaches_index]
    m3_paths = [gallery_paths[index] for index in m3_ranked[:config.EVAL_K]]
    m3_labels = [gallery_labels[index] for index in m3_ranked[:config.EVAL_K]]
    plot_m3_vs_m5_comparison_grid(
        query_paths[beaches_index], query_labels[beaches_index],
        (m3_paths, m3_labels), m5_retrievals[beaches_index],
        q_color_weights[beaches_index], q_texture_weights[beaches_index],
        config.FIGURES_DIR / 'fig5_query_m3_vs_m5_beaches.png',
    )

    experiment_config = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'freshly_computed',
        'feature_cache_status': metadata['feature_cache_status'],
        'feature_fingerprint': metadata['feature_fingerprint'],
        'dataset_path': str(config.DATASET_PATH),
        'categories': list(config.EXPECTED_CATEGORIES),
        'random_seed': config.RANDOM_SEED,
        'mlp_seed': config.PART2_SEED,
        'gallery_count': len(gallery_paths),
        'query_count': len(query_paths),
        'gallery_paths': metadata['gallery_image_paths'],
        'query_paths': metadata['query_image_paths'],
        'gallery_labels': gallery_labels,
        'query_labels': query_labels,
        'evaluation_k': config.EVAL_K,
        'feature_extraction': {
            'color_feature_dim': config.COLOR_FEATURE_DIM,
            'texture_feature_dim': config.TEXTURE_FEATURE_DIM,
            'glcm_distance': config.GLCM_DISTANCE,
            'glcm_angles_deg': list(config.GLCM_ANGLES_DEG),
        },
        'normalization': 'Min-Max fitted on gallery features only',
        'training': {
            'architecture': [config.TOTAL_FEATURE_DIM, 16, 8, 2],
            'optimizer': 'Adam',
            'learning_rate': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE,
            'max_epochs': config.EPOCHS,
            'triplet_margin': config.MARGIN,
            'early_stopping_patience': 10,
            'train_images_per_category': config.INTERNAL_TRAIN_PER_CAT,
            'validation_images_per_category': config.INTERNAL_VAL_PER_CAT,
            'training_positive_samples_per_query': 5,
            'training_negative_samples_per_query': 10,
            'validation_positive_samples_per_query': 2,
            'validation_negative_samples_per_query': 5,
        },
        'm3_m4_fusion_weights': {
            'M3': [config.M3_COLOR_WEIGHT, config.M3_TEXTURE_WEIGHT],
            'M4': [config.M4_COLOR_WEIGHT, config.M4_TEXTURE_WEIGHT],
        },
        'm3': {'color_weight': config.M3_COLOR_WEIGHT, 'texture_weight': config.M3_TEXTURE_WEIGHT,
               'distance': 'sqrt(wc^2 * dc^2 + wt^2 * dt^2)'},
        'm4': {'color_weight': config.M4_COLOR_WEIGHT, 'texture_weight': config.M4_TEXTURE_WEIGHT,
               'distance': 'sqrt(wc^2 * dc^2 + wt^2 * dt^2)'},
        'm5': {'distance': 'sqrt(wc * dc^2 + wt * dt^2)', 'weights': 'query-specific MLP Softmax'},
        'm6': {'distance': 'sqrt(wc * dc^2 + wt * dt^2)',
               'weights': 'mean M5 query weights',
               'mean_color_weight': float(mean_weights[0]),
               'mean_texture_weight': float(mean_weights[1])},
        'timing_protocol': (
            'Per-query distance calculation plus ranking only; excludes feature extraction, '
            'MLP inference, image I/O, and plotting. Query weights and baseline representations '
            'are prepared outside the timed loop.'
        ),
        'training_validation_split': {
            'train_count': len(split_info['train_indices']),
            'validation_count': len(split_info['val_indices']),
            'best_epoch': split_info['best_epoch'],
            'best_validation_loss': split_info['best_val_loss'],
        },
    }
    all_rankings = {
        **metadata['baseline_rankings'],
        'M5': m5_rankings,
        'M6': m6_rankings,
    }
    all_times = {
        **metadata['baseline_retrieval_times'],
        'M5': m5_times,
        'M6': m6_times,
    }
    result_payload = {
        'experiment_config': experiment_config,
        'metrics': all_results,
        'rankings': all_rankings,
        'retrieval_times_seconds': all_times,
        'split_info': split_info,
        'training_history': history,
        'query_color_weights': q_color_weights.tolist(),
        'query_texture_weights': q_texture_weights.tolist(),
        'm5_metrics': m5_metrics,
        'm6_metrics': m6_metrics,
        'm6_mean_weights': mean_weights.tolist(),
        'ablation_statistics': weight_statistics,
    }
    for result_path in (config.PART2_RESULTS_PATH, config.EXPERIMENT_RESULTS_PATH):
        with open(result_path, 'wb') as f:
            pickle.dump(result_payload, f)
    with open(config.EXPERIMENT_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(experiment_config, f, indent=2)
    with open(config.TRAINING_HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'split_info': split_info,
            'training_history': history,
        }, f, indent=2)
    with open(config.QUERY_WEIGHTS_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['query_path', 'category', 'color_weight', 'texture_weight'])
        for index, query_path in enumerate(metadata['query_image_paths']):
            writer.writerow([
                query_path,
                query_labels[index],
                float(q_color_weights[index]),
                float(q_texture_weights[index]),
            ])

    _write_experiment_report(
        all_results, weight_statistics, category_weights, mean_weights, experiment_config
    )
    print(f"Saved complete rankings and metrics: {config.EXPERIMENT_RESULTS_PATH}")
    print(f"Saved experiment config: {config.EXPERIMENT_CONFIG_PATH}")
    print(f"Saved per-query M5 weights: {config.QUERY_WEIGHTS_CSV_PATH}")
    print(f"Saved training history: {config.TRAINING_HISTORY_PATH}")
    return all_results


def _legacy_output_paths(stage):
    """Return fixed-path artifacts that a legacy stage may overwrite."""
    paths = [config.FEATURES_PKL_PATH]
    if stage in {"all", "part2", "figures"}:
        paths.extend([
            config.MODEL_CHECKPOINT_PATH,
            config.PART2_RESULTS_PATH,
            config.EXPERIMENT_RESULTS_PATH,
            config.EXPERIMENT_CONFIG_PATH,
            config.QUERY_WEIGHTS_CSV_PATH,
            config.TRAINING_HISTORY_PATH,
            config.PART2_OUTPUT_DIR / "part2_experiment_report.md",
            *(config.PLOTS_DIR / name for name in (
                "precision_comparison.png",
                "recall_comparison.png",
                "f1_comparison.png",
                "map_comparison.png",
                "time_comparison.png",
                "weight_distributions.png",
                "category_weights.png",
            )),
            *(config.FIGURES_DIR / name for name in (
                "fig2_model_performance_comparison.png",
                "fig3_weight_distributions.png",
                "fig4_category_weights.png",
                "fig5_query_m3_vs_m5_beaches.png",
            )),
        ])
    return paths


def _guard_legacy_output_overwrites(stage, allow_overwrite=False):
    existing = [path for path in _legacy_output_paths(stage) if path.exists()]
    if existing and not allow_overwrite:
        paths = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Legacy pipeline outputs already exist and are protected from "
            "accidental overwrite. Use --allow-overwrite-results only if "
            "replacing these artifacts is intentional:\n" + paths
        )


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
    parser.add_argument(
        "--allow-overwrite-results",
        action="store_true",
        help="Allow legacy fixed-path outputs and figures to be overwritten.",
    )
    args = parser.parse_args()
    
    if args.stage == "sanity":
        from sanity_check import main as run_sanity
        run_sanity()
        return
    else:
        try:
            _guard_legacy_output_overwrites(
                args.stage, allow_overwrite=args.allow_overwrite_results
            )
        except FileExistsError as error:
            parser.error(str(error))

    if args.stage == "figures":
        baseline_results, metadata = run_part1(
            force_reextract=args.force_extract
        )
        run_part2(baseline_results=baseline_results, metadata=metadata)
    elif args.stage == "part1":
        run_part1(force_reextract=args.force_extract)
    elif args.stage == "part2":
        baseline_results, metadata = run_part1(
            force_reextract=args.force_extract
        )
        run_part2(baseline_results=baseline_results, metadata=metadata)
    else:  # "all"
        baseline_results, metadata = run_part1(
            force_reextract=args.force_extract
        )
        run_part2(baseline_results=baseline_results, metadata=metadata)

if __name__ == "__main__":
    main()
