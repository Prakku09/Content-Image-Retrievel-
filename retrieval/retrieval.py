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


def validate_fusion_weights(weights, atol=1e-6):
    """Validate one or more colour/texture weight pairs without changing them."""
    weights = np.asarray(weights)
    if not np.issubdtype(weights.dtype, np.number) or np.issubdtype(
        weights.dtype, np.complexfloating
    ):
        raise ValueError("Fusion weights must be real numeric values.")
    if weights.ndim not in (1, 2) or weights.shape[-1] != 2:
        raise ValueError("Fusion weights must have shape (2,) or (N, 2).")
    weights_float = weights.astype(np.float64, copy=False)
    if not np.all(np.isfinite(weights_float)):
        raise ValueError("Fusion weights must be finite.")
    if np.any(weights_float < 0):
        raise ValueError("Fusion weights must be non-negative.")
    if not np.allclose(
        np.sum(weights_float, axis=-1), 1.0, atol=atol, rtol=0.0
    ):
        raise ValueError(f"Fusion weights must sum to one within {atol}.")
    return weights


def weighted_euclidean_distances(
    query_color, gallery_color, query_texture, gallery_texture, weights
):
    """
    Compute the M5/M6 direct-weight distance:
    sqrt(w_colour * d_colour**2 + w_texture * d_texture**2).
    """
    weights = validate_fusion_weights(weights)
    if weights.ndim != 1:
        raise ValueError("A single query requires exactly one fusion weight pair.")

    query_color = np.asarray(query_color)
    gallery_color = np.asarray(gallery_color)
    query_texture = np.asarray(query_texture)
    gallery_texture = np.asarray(gallery_texture)
    if query_color.ndim != 1 or query_texture.ndim != 1:
        raise ValueError("Query colour and texture features must be one-dimensional.")
    if gallery_color.ndim != 2 or gallery_texture.ndim != 2:
        raise ValueError("Gallery colour and texture features must be two-dimensional.")
    if gallery_color.shape[0] != gallery_texture.shape[0]:
        raise ValueError("Colour and texture galleries must have the same size.")
    if gallery_color.shape[1] != query_color.shape[0]:
        raise ValueError("Query and gallery colour feature dimensions must match.")
    if gallery_texture.shape[1] != query_texture.shape[0]:
        raise ValueError("Query and gallery texture feature dimensions must match.")

    color_dist_sq = np.sum((gallery_color - query_color) ** 2, axis=1)
    texture_dist_sq = np.sum((gallery_texture - query_texture) ** 2, axis=1)
    squared_distances = weights[0] * color_dist_sq + weights[1] * texture_dist_sq
    return np.sqrt(np.maximum(squared_distances, 0.0))


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
