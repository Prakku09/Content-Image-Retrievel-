import numpy as np
import config

def get_representation(color_feats, texture_feats, model_name):
    """
    Constructs the feature representation for a given model (M1, M2, M3, M4).
    
    Args:
        color_feats (np.ndarray): Shape (N, 9) normalized color features.
        texture_feats (np.ndarray): Shape (N, 4) normalized texture features.
        model_name (str): 'M1', 'M2', 'M3', or 'M4'.
        
    Returns:
        np.ndarray: Constructed feature matrix.
    """
    if model_name == 'M1':
        return color_feats
    elif model_name == 'M2':
        return texture_feats
    elif model_name == 'M3':
        wc = config.M3_COLOR_WEIGHT
        wt = config.M3_TEXTURE_WEIGHT
        return np.hstack([wc * color_feats, wt * texture_feats])
    elif model_name == 'M4':
        wc = config.M4_COLOR_WEIGHT
        wt = config.M4_TEXTURE_WEIGHT
        return np.hstack([wc * color_feats, wt * texture_feats])
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def retrieve_images(query_feat, gallery_feats):
    """
    Computes Euclidean distances between a single query representation and all gallery representations,
    then ranks the gallery images.
    
    Args:
        query_feat (np.ndarray): 1D array of shape (D,) query features.
        gallery_feats (np.ndarray): 2D array of shape (N, D) gallery features.
        
    Returns:
        tuple: (ranked_indices, distances)
               ranked_indices (np.ndarray): Gallery indices sorted by distance (ascending).
               distances (np.ndarray): Corresponding sorted Euclidean distances.
    """
    # Vectorized Euclidean distance calculation: sqrt(sum((q - g)^2))
    diff = gallery_feats - query_feat
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    
    # Sort from smallest to largest distance
    ranked_indices = np.argsort(dists)
    
    return ranked_indices, dists[ranked_indices]

def retrieve_all(query_feats, gallery_feats):
    """
    Performs retrieval for all queries against the gallery.
    
    Args:
        query_feats (np.ndarray): Shape (Q, D) query features.
        gallery_feats (np.ndarray): Shape (G, D) gallery features.
        
    Returns:
        tuple: (all_ranked_indices, all_distances)
               all_ranked_indices (list of np.ndarray): Ranked gallery indices for each query.
               all_distances (list of np.ndarray): Corresponding sorted distances for each query.
    """
    all_ranked_indices = []
    all_distances = []
    for q in query_feats:
        indices, dists = retrieve_images(q, gallery_feats)
        all_ranked_indices.append(indices)
        all_distances.append(dists)
    return all_ranked_indices, all_distances
