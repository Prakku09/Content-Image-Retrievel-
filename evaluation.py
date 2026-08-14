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
    gallery_labels_arr = np.array(gallery_labels)
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
