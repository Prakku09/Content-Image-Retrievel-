import numpy as np

def evaluate_retrieval(ranked_indices, gallery_labels, query_label, k=10):
    """
    Computes Precision@K, Recall@K, F1@K, and AP (Average Precision) for a single query.
    
    Args:
        ranked_indices (np.ndarray): Shape (G,) ranked gallery indices for this query.
        gallery_labels (list of str): List of category labels for the gallery images.
        query_label (str): The category label of the query image.
        k (int): Top-K value for Precision/Recall/F1.
        
    Returns:
        dict: Metrics for this query.
    """
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k must be a positive integer.")
    ranked_indices = np.asarray(ranked_indices)
    gallery_labels_arr = np.asarray(gallery_labels)
    if ranked_indices.ndim != 1:
        raise ValueError("Ranked indices must be one-dimensional.")
    if not np.issubdtype(ranked_indices.dtype, np.integer):
        raise ValueError("Ranked indices must contain integers.")
    if gallery_labels_arr.ndim != 1:
        raise ValueError("Gallery labels must be one-dimensional.")
    if len(ranked_indices) != len(gallery_labels_arr):
        raise ValueError("The ranking length must match the gallery label count.")
    if k > len(ranked_indices):
        raise ValueError("k cannot exceed the gallery size.")
    if not np.array_equal(
        np.sort(ranked_indices), np.arange(len(gallery_labels_arr))
    ):
        raise ValueError("Ranked indices must be a permutation of the gallery.")
    ranked_labels = gallery_labels_arr[ranked_indices]
    
    # Relevance mask (1 for relevant, 0 for non-relevant)
    relevance = (ranked_labels == query_label).astype(int)
    
    # Total number of relevant images in the gallery
    total_relevant = np.sum(relevance)
    
    # Precision@K
    tp_k = np.sum(relevance[:k])
    precision_k = tp_k / k
    
    # Recall@K
    recall_k = tp_k / total_relevant if total_relevant > 0 else 0.0
    
    # F1-score@K
    if precision_k + recall_k > 0:
        f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
    else:
        f1_k = 0.0
        
    # Average Precision (AP) over the complete gallery (G elements)
    # AP = sum(P(i) * rel(i)) / total_relevant
    ranks = np.arange(1, len(relevance) + 1)
    cum_relevant = np.cumsum(relevance)
    precisions_at_ranks = cum_relevant / ranks
    
    if total_relevant > 0:
        ap = np.sum(precisions_at_ranks * relevance) / total_relevant
    else:
        ap = 0.0
        
    return {
        'precision_k': precision_k,
        'recall_k': recall_k,
        'f1_k': f1_k,
        'ap': ap
    }

def evaluate_all(all_ranked_indices, gallery_labels, query_labels, retrieval_times, k=10):
    """
    Computes mean metrics over all queries.
    
    Args:
        all_ranked_indices (list of np.ndarray): List of ranked indices for each query.
        gallery_labels (list of str): Labels of the gallery.
        query_labels (list of str): Labels of the queries.
        retrieval_times (list of float): Time taken for each query retrieval.
        k (int): Top-K value.
        
    Returns:
        dict: Mean evaluation metrics.
    """
    if len(all_ranked_indices) != len(query_labels):
        raise ValueError("The number of rankings must match the number of query labels.")
    if len(retrieval_times) != len(query_labels):
        raise ValueError("The number of retrieval times must match the number of queries.")
    if not query_labels:
        raise ValueError("At least one query is required for evaluation.")
    if any(
        not np.all(np.isfinite(np.asarray(ranking)))
        or not np.array_equal(
            np.sort(np.asarray(ranking)), np.arange(len(gallery_labels))
        )
        for ranking in all_ranked_indices
    ):
        raise ValueError("Each ranking must be a permutation of the gallery indices.")
    if not np.all(np.isfinite(np.asarray(retrieval_times, dtype=float))):
        raise ValueError("Retrieval times must be finite.")
    if np.any(np.asarray(retrieval_times, dtype=float) < 0):
        raise ValueError("Retrieval times cannot be negative.")

    precisions = []
    recalls = []
    f1s = []
    aps = []
    
    for indices, q_label in zip(all_ranked_indices, query_labels):
        metrics = evaluate_retrieval(indices, gallery_labels, q_label, k=k)
        precisions.append(metrics['precision_k'])
        recalls.append(metrics['recall_k'])
        f1s.append(metrics['f1_k'])
        aps.append(metrics['ap'])
        
    return {
        'mean_precision': float(np.mean(precisions)),
        'mean_recall': float(np.mean(recalls)),
        'mean_f1': float(np.mean(f1s)),
        'mAP': float(np.mean(aps)),
        'mean_time': float(np.mean(retrieval_times))
    }
