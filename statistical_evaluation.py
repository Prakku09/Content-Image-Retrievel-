"""Prespecified paired statistical evaluation for M5, M6 and M7."""

import csv
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wilcoxon

import config
from data.dataset_loader import load_dataset
from features.normalization import MinMaxNormalizer
from main import _feature_cache_matches, _feature_fingerprint, _relative_paths
from model.mlp_trainer import train_mlp
from retrieval.evaluation import evaluate_retrieval
from retrieval.retrieval import validate_fusion_weights, weighted_euclidean_distances


METRICS = ('average_precision', 'precision_at_10', 'recall_at_10', 'f1_at_10')
PRIMARY_COMPARISON = ('M5', 'M6', 'average_precision')
SECONDARY_COMPARISONS = (
    ('M5', 'M7', 'average_precision'),
    ('M5', 'M6', 'precision_at_10'),
    ('M5', 'M6', 'recall_at_10'),
    ('M5', 'M6', 'f1_at_10'),
    ('M5', 'M7', 'precision_at_10'),
    ('M5', 'M7', 'recall_at_10'),
    ('M5', 'M7', 'f1_at_10'),
)


def _paired_bootstrap_intervals(differences, categories, replicates, seed):
    """Percentile CIs from a category-stratified paired-query bootstrap."""
    differences = np.asarray(differences, dtype=float)
    categories = np.asarray(categories)
    unique_categories = np.unique(categories)
    grouped_indices = [
        np.flatnonzero(categories == category) for category in unique_categories
    ]
    if any(len(indices) == 0 for indices in grouped_indices):
        raise ValueError("Each bootstrap category stratum must contain queries.")

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(replicates, dtype=float)
    bootstrap_dz = np.empty(replicates, dtype=float)
    samples_per_category = len(grouped_indices[0])
    if any(len(indices) != samples_per_category for indices in grouped_indices):
        raise ValueError("The prespecified stratified bootstrap requires balanced categories.")
    for replicate in range(replicates):
        sampled = np.concatenate([
            rng.choice(indices, size=samples_per_category, replace=True)
            for indices in grouped_indices
        ])
        sample = differences[sampled]
        bootstrap_means[replicate] = np.mean(sample)
        standard_deviation = float(np.std(sample, ddof=1))
        bootstrap_dz[replicate] = (
            np.mean(sample) / standard_deviation
            if standard_deviation > 0 else np.nan
        )
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    finite_dz = bootstrap_dz[np.isfinite(bootstrap_dz)]
    dz_interval = (
        [float(value) for value in np.quantile(finite_dz, [0.025, 0.975])]
        if finite_dz.size else None
    )
    return {
        'mean_difference_ci': [float(lower), float(upper)],
        'paired_cohen_dz_ci': dz_interval,
    }


def paired_bootstrap_ci(differences, categories, replicates, seed):
    """Percentile CI for the mean from a category-stratified paired bootstrap."""
    return _paired_bootstrap_intervals(
        differences, categories, replicates, seed
    )['mean_difference_ci']


def paired_permutation_test(differences, replicates, seed):
    """Two-sided Monte Carlo paired sign-flip permutation test for a mean."""
    differences = np.asarray(differences, dtype=float)
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    chunk_size = min(5000, replicates)
    while completed < replicates:
        batch_size = min(chunk_size, replicates - completed)
        signs = rng.choice((-1.0, 1.0), size=(batch_size, len(differences)))
        permuted_means = np.mean(signs * differences, axis=1)
        extreme += int(np.count_nonzero(
            np.abs(permuted_means) >= abs(observed) - 1e-15
        ))
        completed += batch_size
    return (extreme + 1) / (replicates + 1)


def holm_adjust(p_values):
    """Return Holm step-down adjusted p-values in original order."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or not np.all(np.isfinite(p_values)):
        raise ValueError("Holm correction requires a finite one-dimensional p-value list.")
    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p-values must be in [0, 1].")
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted.tolist()


def _metric_rows(rankings, gallery_labels, query_labels, query_paths, seed):
    rows = []
    for model_name, model_rankings in rankings.items():
        for query_index, ranking in enumerate(model_rankings):
            metrics = evaluate_retrieval(
                ranking, gallery_labels, query_labels[query_index], k=config.EVAL_K
            )
            rows.append({
                'seed': int(seed),
                'query_path': query_paths[query_index],
                'category': query_labels[query_index],
                'model': model_name,
                'average_precision': float(metrics['ap']),
                'precision_at_10': float(metrics['precision_k']),
                'recall_at_10': float(metrics['recall_k']),
                'f1_at_10': float(metrics['f1_k']),
            })
    return rows


def _summarize(rows, seed):
    return {
        model: {
            metric: float(np.mean([
                row[metric] for row in rows
                if row['model'] == model and row['seed'] == seed
            ]))
            for metric in METRICS
        }
        for model in ('M5', 'M6', 'M7')
    }


def _test_comparison(rows, category_labels, left, right, metric, seed,
                     bootstrap_seed, permutation_seed):
    left_values = np.asarray([
        row[metric] for row in rows
        if row['seed'] == seed and row['model'] == left
    ])
    right_values = np.asarray([
        row[metric] for row in rows
        if row['seed'] == seed and row['model'] == right
    ])
    if left_values.shape != right_values.shape or len(left_values) != len(category_labels):
        raise ValueError(f"Paired metric rows are incomplete for {left} versus {right}.")
    differences = left_values - right_values
    bootstrap_intervals = _paired_bootstrap_intervals(
        differences,
        category_labels,
        config.STATISTICAL_BOOTSTRAP_REPLICATES,
        bootstrap_seed,
    )
    permutation_p = paired_permutation_test(
        differences,
        config.STATISTICAL_PERMUTATION_REPLICATES,
        permutation_seed,
    )
    standard_deviation = float(np.std(differences, ddof=1))
    cohen_dz = (
        float(np.mean(differences) / standard_deviation)
        if standard_deviation > 0
        else None
    )
    return {
        'left_model': left,
        'right_model': right,
        'metric': metric,
        'n_paired_queries': len(differences),
        'mean_paired_improvement': float(np.mean(differences)),
        'paired_cohen_dz': cohen_dz,
        'paired_bootstrap_95_ci': bootstrap_intervals['mean_difference_ci'],
        'paired_cohen_dz_bootstrap_95_ci': bootstrap_intervals['paired_cohen_dz_ci'],
        'paired_permutation_p_two_sided': float(permutation_p),
        'holm_adjusted_p': None,
    }, differences


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_features(feature_lookup, dataset_fingerprint, seed):
    gallery_paths, gallery_labels, query_paths, query_labels = load_dataset(seed=seed)
    gallery_rel = _relative_paths(gallery_paths)
    query_rel = _relative_paths(query_paths)
    gallery_color_raw = np.asarray([feature_lookup[path][0] for path in gallery_rel],
                                   dtype=np.float32)
    gallery_texture_raw = np.asarray([feature_lookup[path][1] for path in gallery_rel],
                                     dtype=np.float32)
    query_color_raw = np.asarray([feature_lookup[path][0] for path in query_rel],
                                 dtype=np.float32)
    query_texture_raw = np.asarray([feature_lookup[path][1] for path in query_rel],
                                   dtype=np.float32)

    color_normalizer = MinMaxNormalizer()
    texture_normalizer = MinMaxNormalizer()
    gallery_color = color_normalizer.fit_transform(gallery_color_raw)
    query_color = color_normalizer.transform(query_color_raw)
    gallery_texture = texture_normalizer.fit_transform(gallery_texture_raw)
    query_texture = texture_normalizer.transform(query_texture_raw)
    seed_digest = hashlib.sha256()
    seed_digest.update(dataset_fingerprint.encode('ascii'))
    seed_digest.update(json.dumps({
        'seed': int(seed),
        'gallery_paths': gallery_rel,
        'query_paths': query_rel,
        'training_protocol_version': 3,
        'normalization': 'gallery-fitted MinMaxNormalizer',
    }, sort_keys=True).encode('utf-8'))
    for array in (
        gallery_color, gallery_texture, query_color, query_texture,
        color_normalizer.min_val, color_normalizer.max_val,
        texture_normalizer.min_val, texture_normalizer.max_val,
    ):
        seed_digest.update(np.asarray(array, dtype=np.float32).tobytes())

    seed_data = {
        'dataset_path': str(config.DATASET_PATH),
        'feature_fingerprint': seed_digest.hexdigest(),
        'dataset_fingerprint': dataset_fingerprint,
        'random_seed': int(seed),
        'categories': list(config.EXPECTED_CATEGORIES),
        'gallery_image_paths': gallery_rel,
        'query_image_paths': query_rel,
        'normalized_gallery_color': gallery_color,
        'normalized_gallery_texture': gallery_texture,
        'normalized_query_color': query_color,
        'normalized_query_texture': query_texture,
        'normalization_parameters': {
            'color_min': color_normalizer.min_val.tolist(),
            'color_max': color_normalizer.max_val.tolist(),
            'texture_min': texture_normalizer.min_val.tolist(),
            'texture_max': texture_normalizer.max_val.tolist(),
        },
    }
    if len(set(gallery_rel) & set(query_rel)):
        raise ValueError(f"Seed {seed} has overlapping gallery and query images.")
    return seed_data, gallery_labels, query_labels


def _evaluate_seed(seed, seed_data, gallery_labels, query_labels, run_dir):
    seed_dir = run_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    feature_path = seed_dir / 'seed_features.pkl'
    checkpoint_path = seed_dir / 'm5_checkpoint.pt'
    history_path = seed_dir / 'training_history.json'
    with open(feature_path, 'wb') as feature_file:
        pickle.dump(seed_data, feature_file)

    model, split_info, history = train_mlp(
        feature_path, seed=seed, checkpoint_path=checkpoint_path
    )
    train_indices = set(split_info['train_indices'])
    validation_indices = set(split_info['val_indices'])
    if (
        len(train_indices) != 700 or len(validation_indices) != 200
        or train_indices & validation_indices
        or train_indices | validation_indices != set(range(900))
    ):
        raise ValueError(f"Seed {seed} has an invalid train/validation partition.")
    gallery_categories = np.asarray([
        Path(path).parent.name for path in seed_data['gallery_image_paths']
    ])
    train_counts = {
        category: int(np.count_nonzero(
            gallery_categories[list(train_indices)] == category
        ))
        for category in config.EXPECTED_CATEGORIES
    }
    validation_counts = {
        category: int(np.count_nonzero(
            gallery_categories[list(validation_indices)] == category
        ))
        for category in config.EXPECTED_CATEGORIES
    }
    if set(train_counts.values()) != {config.INTERNAL_TRAIN_PER_CAT} or set(
        validation_counts.values()
    ) != {config.INTERNAL_VAL_PER_CAT}:
        raise ValueError(f"Seed {seed} train/validation category counts are invalid.")

    model.eval()
    norm_gallery_color = seed_data['normalized_gallery_color']
    norm_gallery_texture = seed_data['normalized_gallery_texture']
    norm_query_color = seed_data['normalized_query_color']
    norm_query_texture = seed_data['normalized_query_texture']
    combined_queries = np.hstack([norm_query_color, norm_query_texture])
    with torch.no_grad():
        predicted = model(torch.tensor(combined_queries, dtype=torch.float32)).cpu().numpy()
    predicted = validate_fusion_weights(predicted)
    if predicted.shape != (len(query_labels), 2):
        raise ValueError(f"Seed {seed} prediction count does not match its query set.")

    mean_weights = validate_fusion_weights(np.mean(predicted, axis=0))
    m7_weights = np.array([0.5, 0.5], dtype=predicted.dtype)
    weights_by_model = {
        'M5': predicted,
        'M6': np.tile(mean_weights, (len(query_labels), 1)),
        'M7': np.tile(m7_weights, (len(query_labels), 1)),
    }
    rankings = {}
    for model_name, model_weights in weights_by_model.items():
        model_rankings = []
        for index, weights in enumerate(model_weights):
            distances = weighted_euclidean_distances(
                norm_query_color[index], norm_gallery_color,
                norm_query_texture[index], norm_gallery_texture, weights,
            )
            model_rankings.append(np.argsort(distances))
        rankings[model_name] = model_rankings

    if len(set(query_labels)) != len(config.EXPECTED_CATEGORIES):
        raise ValueError(f"Seed {seed} query split does not cover every category.")
    rows = _metric_rows(
        rankings, gallery_labels, query_labels,
        seed_data['query_image_paths'], seed,
    )
    with open(history_path, 'w', encoding='utf-8') as history_file:
        json.dump({
            'split_info': split_info,
            'training_history': history,
            'checkpoint_sha256': _file_sha256(checkpoint_path),
            'feature_fingerprint': seed_data['feature_fingerprint'],
            'dataset_fingerprint': seed_data['dataset_fingerprint'],
        }, history_file, indent=2)
    return rows, predicted, mean_weights, split_info, history, rankings


def _category_summary(rows, seed):
    categories = sorted(config.EXPECTED_CATEGORIES)
    comparisons = (('M5', 'M6'), ('M5', 'M7'))
    result = []
    for category in categories:
        cat_rows = [row for row in rows if row['seed'] == seed and row['category'] == category]
        for left, right in comparisons:
            for metric in METRICS:
                left_values = [row[metric] for row in cat_rows if row['model'] == left]
                right_values = [row[metric] for row in cat_rows if row['model'] == right]
                differences = np.asarray(left_values) - np.asarray(right_values)
                result.append({
                    'seed': int(seed),
                    'category': category,
                    'left_model': left,
                    'right_model': right,
                    'metric': metric,
                    'n_queries': len(differences),
                    'mean_paired_improvement': float(np.mean(differences)),
                    'queries_improved': int(np.count_nonzero(differences > 0)),
                    'queries_tied': int(np.count_nonzero(differences == 0)),
                    'queries_worse': int(np.count_nonzero(differences < 0)),
                })
    return result


def _format_ci(interval):
    if interval is None:
        return "NA"
    return f"[{interval[0]:.6f}, {interval[1]:.6f}]"


def _write_report(run_dir, run_config, summaries, primary, secondary,
                  wilcoxon_result, category_rows):
    primary_supported = (
        primary['paired_permutation_p_two_sided'] < 0.05
        and primary['paired_bootstrap_95_ci'][0] > 0
    )
    conclusion = (
        "The prespecified primary analysis supports a positive M5 versus M6 mAP "
        "difference for the seed-42 query population."
        if primary_supported else
        "The prespecified primary analysis is inconclusive for a positive M5 versus "
        "M6 mAP difference; statistical superiority is not established."
    )
    secondary_lines = []
    for comparison in secondary:
        secondary_lines.append(
            f"| {comparison['left_model']}−{comparison['right_model']} "
            f"{comparison['metric']} | {comparison['mean_paired_improvement']:.6f} | "
            f"{_format_ci(comparison['paired_bootstrap_95_ci'])} | "
            f"{comparison['paired_cohen_dz'] if comparison['paired_cohen_dz'] is not None else 'NA'} "
            f"{_format_ci(comparison['paired_cohen_dz_bootstrap_95_ci'])} | "
            f"{comparison['paired_permutation_p_two_sided']:.6g} | "
            f"{comparison['holm_adjusted_p']:.6g} |"
        )
    summary_lines = []
    for seed in run_config['seeds']:
        seed_summary = summaries[str(seed)]
        for model in ('M5', 'M6', 'M7'):
            metrics = seed_summary[model]
            summary_lines.append(
                f"| {seed} | {model} | {metrics['average_precision']:.4f} | "
                f"{metrics['precision_at_10']:.4f} | {metrics['recall_at_10']:.4f} | "
                f"{metrics['f1_at_10']:.4f} |"
            )
    category_m5_m6 = [
        row for row in category_rows
        if row['seed'] == 42 and row['left_model'] == 'M5'
        and row['right_model'] == 'M6' and row['metric'] == 'average_precision'
    ]
    category_m5_m7 = [
        row for row in category_rows
        if row['seed'] == 42 and row['left_model'] == 'M5'
        and row['right_model'] == 'M7' and row['metric'] == 'average_precision'
    ]
    category_table = [
        "| Category | M5−M6 mAP | Queries M5 better/tied/worse | "
        "M5−M7 mAP | Queries M5 better/tied/worse |",
        "|---|---:|---:|---:|---:|",
    ]
    for category in sorted(config.EXPECTED_CATEGORIES):
        a = next(row for row in category_m5_m6 if row['category'] == category)
        b = next(row for row in category_m5_m7 if row['category'] == category)
        category_table.append(
            f"| {category} | {a['mean_paired_improvement']:.5f} | "
            f"{a['queries_improved']}/{a['queries_tied']}/{a['queries_worse']} | "
            f"{b['mean_paired_improvement']:.5f} | "
            f"{b['queries_improved']}/{b['queries_tied']}/{b['queries_worse']} |"
        )
    positive_m6 = sum(row['mean_paired_improvement'] > 0 for row in category_m5_m6)
    positive_m7 = sum(row['mean_paired_improvement'] > 0 for row in category_m5_m7)
    seed42_weights = run_config['seed_artifacts']['42']['m5_weight_summary']

    report = f"""# Prespecified Statistical Evaluation: M5 versus M6 and M7

## Methods and protocol

- **Primary comparison/outcome:** M5 minus M6 per-query average precision; two-sided paired sign-flip permutation test at alpha=0.05 with a category-stratified paired bootstrap 95% percentile CI.
- **Secondary family (Holm corrected):** M5−M7 on average precision, and M5−M6 / M5−M7 on Precision@10, Recall@10, and F1@10. All secondary raw p-values are paired two-sided permutation tests; Holm correction is across these seven prespecified tests.
- **Secondary robustness test:** two-sided Wilcoxon signed-rank test for seed-42 paired M5−M6 mAP differences.
- **Effect size:** mean paired difference in metric units and paired Cohen's dz (mean difference divided by sample SD of paired differences). Bootstrap CI is for the mean paired difference.
- **Bootstrap:** {run_config['bootstrap_replicates']:,} resamples, resampling query images with replacement within each of the ten categories, preserving category balance and each query's paired model outcomes.
- **Permutation:** {run_config['permutation_replicates']:,} Monte Carlo paired sign flips, two-sided, plus-one correction.
- **Primary sample:** seed 42 only, {primary['n_paired_queries']} distinct paired query images. Seeds 42, 123, 2025, 31415, and 271828 are prespecified sensitivity runs. Their metrics are reported separately; overlapping images across seeds are not pooled as independent observations and are not added to the inferential sample.
- **Protocol:** same gallery, query set, ground truth and metric definitions across M5/M6/M7 within each seed. Min-max normalization is fitted separately on that seed's 900-image gallery. The MLP is trained/validated only on the gallery (70/20 per category); held-out queries do not enter training or checkpoint selection.
- **Training configuration:** MLP 13→16→8→2, Adam (learning rate {config.LEARNING_RATE}), batch size {config.BATCH_SIZE}, maximum {config.EPOCHS} epochs, triplet margin {config.MARGIN}, patience 10; 5 positives × 10 negatives per training query and 2 × 5 validation triplets per validation query.
- **Methods:** M5 predicts query-specific Softmax weights; M6 fixes the mean M5 weights for that seed; M7 uses fixed distance-level weights (0.5, 0.5). Each uses `sqrt(wc * dc^2 + wt * dt^2)`.
- **Data/cache integrity:** source image bytes, split paths, normalized features and training configuration are fingerprinted. Each seed has its own checkpoint. Metrics are recomputed; no historical metric cache is used. Checkpoint hashes and split/fingerprint metadata are saved per seed.

## Prespecified primary result (seed 42)

| Contrast | Mean paired mAP improvement | Paired bootstrap 95% CI | Cohen's dz (bootstrap 95% CI) | Paired permutation p (two-sided) |
|---|---:|---:|---:|---:|
| M5−M6 | {primary['mean_paired_improvement']:.6f} | {_format_ci(primary['paired_bootstrap_95_ci'])} | {primary['paired_cohen_dz'] if primary['paired_cohen_dz'] is not None else 'NA'} ({_format_ci(primary['paired_cohen_dz_bootstrap_95_ci'])}) | {primary['paired_permutation_p_two_sided']:.6g} |

Wilcoxon signed-rank (M5−M6 mAP): W={wilcoxon_result['statistic']:.4f}, raw two-sided p={wilcoxon_result['p_value']:.6g}, non-zero pairs={wilcoxon_result['n_nonzero_pairs']}.

The Wilcoxon result is not significant at 0.05, unlike the prespecified mean-based permutation test. This disagreement makes the result sensitive to the chosen estimand/test; the primary conclusion is limited to the prespecified mean mAP comparison and does not imply a median/location shift.

**Conclusion:** {conclusion}

## Prespecified secondary comparisons

| Contrast / outcome | Mean paired improvement | Mean-difference 95% CI | Cohen's dz (bootstrap 95% CI) | Raw permutation p | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
{chr(10).join(secondary_lines)}

## Per-seed metric summaries (sensitivity analysis; not pooled)

| Seed | Model | mAP | Precision@10 | Recall@10 | F1@10 |
|---:|---|---:|---:|---:|---:|
{chr(10).join(summary_lines)}

## Seed-42 M5 predicted weights

| Weight | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Colour | {seed42_weights['color_mean']:.6f} | {seed42_weights['color_std']:.6f} | {seed42_weights['color_min']:.6f} | {seed42_weights['color_max']:.6f} |
| Texture | {seed42_weights['texture_mean']:.6f} | {seed42_weights['texture_std']:.6f} | {seed42_weights['texture_min']:.6f} | {seed42_weights['texture_max']:.6f} |

## Category-level seed-42 mAP comparisons

M5 has positive mean mAP differences in {positive_m6}/10 categories versus M6 and {positive_m7}/10 versus M7. These category summaries are descriptive; ten queries per category provide limited precision and no category-wise significance claims are made.

{chr(10).join(category_table)}

## Assumptions and limitations

- Paired methods use identical query images and gallery rankings within each seed. The sign-flip permutation test assumes exchangeability of paired differences under the null; query-level dependence induced by a shared gallery and semantic category structure can limit its calibration. The stratified bootstrap conditions on the ten benchmark categories and resamples queries within each category; it does not capture uncertainty from sampling a different gallery or dataset.
- Wilcoxon signed-rank does not test the mean difference. Its usual location interpretation assumes a roughly symmetric paired-difference distribution; zeros and ties reduce information. It is secondary and not substituted for the prespecified mean-based test.
- The seed-42 inferential sample is one query split on one fixed Corel-1K dataset. Additional random seeds change both query/gallery membership and model fitting; repeated image appearances overlap across seeds. Therefore seed results are sensitivity analyses only, not independent replications or a basis for inflating sample size.
- M5, M6 and M7 share learned features/model fitting where applicable; model comparisons are paired, but this is not evidence of population-level generalization beyond the tested dataset and protocol. No hyperparameters or checkpoints were selected using held-out queries.
- The paired bootstrap interval and permutation p-value are conditional on this benchmark protocol. No claim of universal or category-general superiority follows from weight variability alone.

## Reproducibility

- Dataset fingerprint: `{run_config['dataset_fingerprint']}`
- Seed-specific split/training/checkpoint metadata and history: `seed_<seed>/training_history.json`
- Seed-specific verified M5 checkpoint: `seed_<seed>/m5_checkpoint.pt`
- Per-query metrics: `per_query_metrics.csv`
- Per-query predicted/control weights: `per_query_weights.csv`
- Category comparison table: `category_comparisons.csv`
- Configuration and statistical tests: `statistical_config.json` and `statistical_results.json`
- Charts: `plots/`
- Seeds: `{run_config['seeds']}`
- Bootstrap/permutation RNG seeds: {run_config['bootstrap_rng_seed']} / {run_config['permutation_rng_seed']}
"""
    (run_dir / 'statistical_evaluation_report.md').write_text(report, encoding='utf-8')


def run_statistical_evaluation():
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = config.STATISTICAL_OUTPUT_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    seed42_gallery, _, seed42_queries, _ = load_dataset(seed=42)
    dataset_fingerprint = _feature_fingerprint(seed42_gallery, seed42_queries)
    with open(config.FEATURES_PKL_PATH, 'rb') as cache_file:
        cache = pickle.load(cache_file)
    if not _feature_cache_matches(
        cache, dataset_fingerprint, seed42_gallery, seed42_queries
    ):
        raise ValueError(
            "The base feature cache is stale or unverified. Run the feature pipeline first."
        )

    feature_lookup = {}
    for path_key, color, texture in (
        (path, color, texture)
        for path, color, texture in zip(
            cache['gallery_image_paths'],
            cache['gallery_color_features'],
            cache['gallery_texture_features'],
        )
    ):
        if path_key in feature_lookup:
            raise ValueError(f"Duplicate source feature path in cache: {path_key}")
        feature_lookup[path_key] = (color, texture)
    for path_key, color, texture in zip(
        cache['query_image_paths'],
        cache['query_color_features'],
        cache['query_texture_features'],
    ):
        if path_key in feature_lookup:
            raise ValueError(f"Duplicate source feature path in cache: {path_key}")
        feature_lookup[path_key] = (color, texture)
    if len(feature_lookup) != 1000:
        raise ValueError(f"Expected cached features for 1000 unique images, got {len(feature_lookup)}.")

    run_config = {
        'analysis_version': 1,
        'status': 'in_progress',
        'dataset_fingerprint': dataset_fingerprint,
        'seeds': list(config.STATISTICAL_SEEDS),
        'primary_comparison': list(PRIMARY_COMPARISON),
        'secondary_comparisons': [list(item) for item in SECONDARY_COMPARISONS],
        'bootstrap_replicates': config.STATISTICAL_BOOTSTRAP_REPLICATES,
        'permutation_replicates': config.STATISTICAL_PERMUTATION_REPLICATES,
        'bootstrap_rng_seed': 60042,
        'permutation_rng_seed': 80042,
        'confidence_level': 0.95,
        'alpha': 0.05,
        'model_formula': 'sqrt(wc * dc^2 + wt * dt^2)',
        'm7_weights': [0.5, 0.5],
        'metrics': list(METRICS),
        'query_gallery_protocol': '90 gallery and 10 query per category',
        'train_validation_protocol': '70 train and 20 validation per category from gallery only',
        'training_configuration': {
            'architecture': [config.TOTAL_FEATURE_DIM, 16, 8, 2],
            'optimizer': 'Adam',
            'learning_rate': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE,
            'maximum_epochs': config.EPOCHS,
            'triplet_margin': config.MARGIN,
            'early_stopping_patience': 10,
            'train_positive_samples_per_query': 5,
            'train_negative_samples_per_query': 10,
            'validation_positive_samples_per_query': 2,
            'validation_negative_samples_per_query': 5,
        },
    }
    with open(run_dir / 'statistical_config.json', 'w', encoding='utf-8') as config_file:
        json.dump(run_config, config_file, indent=2)

    all_rows = []
    seed_summaries = {}
    category_rows = []
    seed_artifacts = {}
    seed_predictions = {}
    seed42_categories = None
    seed42_m5_m6_differences = None
    seed42_primary = None
    seed42_secondary = []
    seed42_secondary_differences = []

    for seed in run_config['seeds']:
        print(f"\nStatistical evaluation seed {seed}...")
        seed_data, gallery_labels, query_labels = _seed_features(
            feature_lookup, dataset_fingerprint, seed
        )
        rows, predictions, mean_weights, split_info, history, rankings = _evaluate_seed(
            seed, seed_data, gallery_labels, query_labels, run_dir
        )
        all_rows.extend(rows)
        summary = _summarize(rows, seed)
        seed_summaries[str(seed)] = summary
        category_rows.extend(_category_summary(rows, seed))
        seed_predictions[str(seed)] = predictions
        checkpoint_path = run_dir / f"seed_{seed}" / 'm5_checkpoint.pt'
        seed_artifacts[str(seed)] = {
            'feature_fingerprint': seed_data['feature_fingerprint'],
            'checkpoint_sha256': _file_sha256(checkpoint_path),
            'gallery_paths': seed_data['gallery_image_paths'],
            'query_paths': seed_data['query_image_paths'],
            'gallery_labels': gallery_labels,
            'query_labels': query_labels,
            'split_info': split_info,
            'best_epoch': split_info['best_epoch'],
            'best_validation_loss': split_info['best_val_loss'],
            'mean_m5_weights': [float(value) for value in mean_weights],
            'm5_weight_summary': {
                'color_mean': float(np.mean(predictions[:, 0])),
                'color_std': float(np.std(predictions[:, 0])),
                'color_min': float(np.min(predictions[:, 0])),
                'color_max': float(np.max(predictions[:, 0])),
                'texture_mean': float(np.mean(predictions[:, 1])),
                'texture_std': float(np.std(predictions[:, 1])),
                'texture_min': float(np.min(predictions[:, 1])),
                'texture_max': float(np.max(predictions[:, 1])),
            },
        }
        if seed == 42:
            seed42_categories = query_labels

    primary, seed42_m5_m6_differences = _test_comparison(
        all_rows, seed42_categories, *PRIMARY_COMPARISON, seed=42,
        bootstrap_seed=run_config['bootstrap_rng_seed'],
        permutation_seed=run_config['permutation_rng_seed'],
    )
    secondary_p_values = []
    for index, comparison in enumerate(SECONDARY_COMPARISONS):
        result, differences = _test_comparison(
            all_rows, seed42_categories, *comparison, seed=42,
            bootstrap_seed=run_config['bootstrap_rng_seed'] + index + 1,
            permutation_seed=run_config['permutation_rng_seed'] + index + 1,
        )
        seed42_secondary.append(result)
        seed42_secondary_differences.append(differences)
        secondary_p_values.append(result['paired_permutation_p_two_sided'])
    adjusted = holm_adjust(secondary_p_values)
    for result, adjusted_p in zip(seed42_secondary, adjusted):
        result['holm_adjusted_p'] = adjusted_p

    wilcoxon_values = wilcoxon(
        seed42_m5_m6_differences,
        alternative='two-sided',
        zero_method='wilcox',
        method='auto',
    )
    wilcoxon_result = {
        'statistic': float(wilcoxon_values.statistic),
        'p_value': float(wilcoxon_values.pvalue),
        'n_nonzero_pairs': int(np.count_nonzero(seed42_m5_m6_differences)),
        'zero_method': 'wilcox',
        'alternative': 'two-sided',
        'method': 'auto',
    }

    per_query_path = run_dir / 'per_query_metrics.csv'
    with open(per_query_path, 'w', newline='', encoding='utf-8') as metrics_file:
        fieldnames = [
            'seed', 'query_path', 'category', 'model',
            'average_precision', 'precision_at_10', 'recall_at_10', 'f1_at_10',
        ]
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    with open(run_dir / 'per_query_weights.csv', 'w', newline='', encoding='utf-8') as weights_file:
        fieldnames = ['seed', 'query_path', 'category', 'm5_color_weight',
                      'm5_texture_weight', 'm6_color_weight', 'm6_texture_weight',
                      'm7_color_weight', 'm7_texture_weight']
        writer = csv.DictWriter(weights_file, fieldnames=fieldnames)
        writer.writeheader()
        for seed in run_config['seeds']:
            weights = seed_predictions[str(seed)]
            mean = np.mean(weights, axis=0)
            query_labels = seed_artifacts[str(seed)]['query_labels']
            query_paths = seed_artifacts[str(seed)]['query_paths']
            for index, path in enumerate(query_paths):
                writer.writerow({
                    'seed': seed,
                    'query_path': path,
                    'category': query_labels[index],
                    'm5_color_weight': float(weights[index, 0]),
                    'm5_texture_weight': float(weights[index, 1]),
                    'm6_color_weight': float(mean[0]),
                    'm6_texture_weight': float(mean[1]),
                    'm7_color_weight': 0.5,
                    'm7_texture_weight': 0.5,
                })

    with open(run_dir / 'category_comparisons.csv', 'w', newline='', encoding='utf-8') as category_file:
        fieldnames = list(category_rows[0])
        writer = csv.DictWriter(category_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(category_rows)

    statistical_results = {
        'primary_comparison': primary,
        'secondary_comparisons': seed42_secondary,
        'wilcoxon_secondary_analysis': wilcoxon_result,
        'per_seed_metrics': seed_summaries,
        'per_seed_artifacts': seed_artifacts,
        'category_comparisons': category_rows,
        'note': 'Seed 42 is the sole inferential sample; other seeds are sensitivity analyses and are not pooled.',
    }
    run_config['seed_artifacts'] = seed_artifacts
    statistical_results['run_config'] = run_config
    with open(run_dir / 'statistical_config.json', 'w', encoding='utf-8') as config_file:
        json.dump(run_config, config_file, indent=2)
    with open(run_dir / 'statistical_results.json', 'w', encoding='utf-8') as results_file:
        json.dump(statistical_results, results_file, indent=2)

    plot_dir = run_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    m6_category = [
        row for row in category_rows
        if row['seed'] == 42 and row['left_model'] == 'M5'
        and row['right_model'] == 'M6' and row['metric'] == 'average_precision'
    ]
    m7_category = [
        row for row in category_rows
        if row['seed'] == 42 and row['left_model'] == 'M5'
        and row['right_model'] == 'M7' and row['metric'] == 'average_precision'
    ]
    x = np.arange(len(config.EXPECTED_CATEGORIES))
    width = 0.38
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar(
        x - width / 2,
        [next(row['mean_paired_improvement'] for row in m6_category
              if row['category'] == category) for category in config.EXPECTED_CATEGORIES],
        width, label='M5 − M6',
    )
    axis.bar(
        x + width / 2,
        [next(row['mean_paired_improvement'] for row in m7_category
              if row['category'] == category) for category in config.EXPECTED_CATEGORIES],
        width, label='M5 − M7',
    )
    axis.axhline(0, color='black', linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(config.EXPECTED_CATEGORIES, rotation=35, ha='right')
    axis.set_ylabel('Mean paired per-query AP difference')
    axis.set_title('Seed 42: category-level mAP improvement (descriptive)')
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / 'category_map_improvements.png', dpi=300)
    plt.close(fig)

    m6_differences = seed42_m5_m6_differences
    m7_differences = np.asarray([
        row['average_precision'] for row in all_rows
        if row['seed'] == 42 and row['model'] == 'M5'
    ]) - np.asarray([
        row['average_precision'] for row in all_rows
        if row['seed'] == 42 and row['model'] == 'M7'
    ])
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.boxplot([m6_differences, m7_differences])
    axis.set_xticks([1, 2])
    axis.set_xticklabels(['M5 − M6', 'M5 − M7'])
    axis.axhline(0, color='black', linewidth=0.8)
    axis.set_ylabel('Per-query average precision difference')
    axis.set_title('Seed 42: paired per-query AP differences')
    fig.tight_layout()
    fig.savefig(plot_dir / 'paired_map_differences.png', dpi=300)
    plt.close(fig)

    _write_report(
        run_dir, run_config, seed_summaries, primary, seed42_secondary,
        wilcoxon_result, category_rows,
    )
    run_config['status'] = 'freshly_computed'
    with open(run_dir / 'statistical_config.json', 'w', encoding='utf-8') as config_file:
        json.dump(run_config, config_file, indent=2)
    statistical_results['run_config'] = run_config
    with open(run_dir / 'statistical_results.json', 'w', encoding='utf-8') as results_file:
        json.dump(statistical_results, results_file, indent=2)
    print(f"\nStatistical report: {run_dir / 'statistical_evaluation_report.md'}")
    print(f"Primary M5-M6 mAP mean difference: {primary['mean_paired_improvement']:.6f}")
    print(f"95% CI: {_format_ci(primary['paired_bootstrap_95_ci'])}")
    print(f"Permutation p-value: {primary['paired_permutation_p_two_sided']:.6g}")
    return statistical_results


if __name__ == '__main__':
    run_statistical_evaluation()
