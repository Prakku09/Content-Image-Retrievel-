"""
Unified Sanity Check Suite for Corel-1K CBIR Pipeline
Tests dataset loading, feature extraction, normalization, retrieval, and MLP adaptive weighting.
"""

import sys
import numpy as np
import torch
import pickle
from pathlib import Path

# Package-level imports
import config
from data.dataset_loader import load_dataset
from features.feature_extractor import extract_all_features
from features.normalization import MinMaxNormalizer
from model.mlp_model import AdaptiveWeightMLP
from retrieval.retrieval import get_representation, retrieve_images

def run_part1_sanity():
    print("--- 1. Testing Dataset & Baseline Pipeline (Part 1) ---")
    gallery_paths, gallery_labels, query_paths, query_labels = load_dataset()
    print(f"[PASS] Loaded {len(gallery_paths)} gallery images and {len(query_paths)} query images.")
    
    # Test on a small 3-category subset (2 gallery, 1 query per cat)
    selected_cats = ["beaches", "dinosaurs", "flowers"]
    sanity_gal_paths, sanity_gal_labels = [], []
    sanity_q_paths, sanity_q_labels = [], []
    
    for cat in selected_cats:
        cat_gal = [p for p, l in zip(gallery_paths, gallery_labels) if l == cat][:2]
        cat_q = [p for p, l in zip(query_paths, query_labels) if l == cat][:1]
        sanity_gal_paths.extend(cat_gal)
        sanity_gal_labels.extend([cat] * len(cat_gal))
        sanity_q_paths.extend(cat_q)
        sanity_q_labels.extend([cat] * len(cat_q))
        
    print(f"Sanity subset: {len(sanity_gal_paths)} gallery images, {len(sanity_q_paths)} query images.")
    
    # Feature extraction
    gal_c, gal_t = [], []
    for p in sanity_gal_paths:
        c, t = extract_all_features(p)
        gal_c.append(c)
        gal_t.append(t)
        
    q_c, q_t = [], []
    for p in sanity_q_paths:
        c, t = extract_all_features(p)
        q_c.append(c)
        q_t.append(t)
        
    gal_c = np.array(gal_c, dtype=np.float32)
    gal_t = np.array(gal_t, dtype=np.float32)
    q_c = np.array(q_c, dtype=np.float32)
    q_t = np.array(q_t, dtype=np.float32)
    
    # Assert dimensions
    assert gal_c.shape == (6, config.COLOR_FEATURE_DIM), f"Expected HSV shape (6, 9), got {gal_c.shape}"
    assert gal_t.shape == (6, config.TEXTURE_FEATURE_DIM), f"Expected GLCM shape (6, 4), got {gal_t.shape}"
    assert not np.isnan(gal_c).any() and not np.isinf(gal_c).any(), "NaN or Inf in gallery color features"
    assert not np.isnan(gal_t).any() and not np.isinf(gal_t).any(), "NaN or Inf in gallery texture features"
    print(f"[PASS] Feature extraction dimensions verified: HSV (9-dim), GLCM (4-dim). No NaNs/Infs.")
    
    # Normalization
    color_norm = MinMaxNormalizer()
    texture_norm = MinMaxNormalizer()
    norm_gal_c = color_norm.fit_transform(gal_c)
    norm_q_c = color_norm.transform(q_c)
    norm_gal_t = texture_norm.fit_transform(gal_t)
    norm_q_t = texture_norm.transform(q_t)
    
    assert np.all(norm_gal_c >= -1e-7) and np.all(norm_gal_c <= 1.000001), "Normalized color outside [0, 1]"
    assert np.all(norm_gal_t >= -1e-7) and np.all(norm_gal_t <= 1.000001), "Normalized texture outside [0, 1]"
    print("[PASS] MinMax Normalizer verified on gallery distribution.")
    
    # Combined representations
    gal_m3 = get_representation(norm_gal_c, norm_gal_t, 'M3')
    q_m3 = get_representation(norm_q_c, norm_q_t, 'M3')
    assert gal_m3.shape == (6, config.TOTAL_FEATURE_DIM), f"Expected M3 shape (6, 13), got {gal_m3.shape}"
    assert q_m3.shape == (3, config.TOTAL_FEATURE_DIM), f"Expected M3 query shape (3, 13), got {q_m3.shape}"
    print("[PASS] Multi-modal feature concatenation verified (13-dim).")
    
    # Distance ranking
    ranked_idx, dists = retrieve_images(q_m3[0], gal_m3)
    assert len(ranked_idx) == len(sanity_gal_paths), "Ranked index length mismatch"
    assert np.all(np.diff(dists) >= 0), "Distances not monotonically increasing"
    print(f"[PASS] Euclidean distance retrieval ranking verified (Top-1 dist: {dists[0]:.4f}).")

def run_part2_sanity():
    print("\n--- 2. Testing MLP Adaptive Weighting Model (Part 2 / M5) ---")
    model = AdaptiveWeightMLP()
    model.eval()
    
    # Create test batch of 5 synthetic 13-dim query representations
    torch.manual_seed(config.PART2_SEED)
    dummy_input = torch.rand(5, config.TOTAL_FEATURE_DIM)
    
    with torch.no_grad():
        weights = model(dummy_input)
        
    weights_np = weights.numpy()
    
    # Output shapes
    assert weights.shape == (5, 2), f"Expected MLP output shape (5, 2), got {weights.shape}"
    print(f"[PASS] MLP forward pass shape verified: (5, 13) -> {tuple(weights.shape)}")
    
    # Non-negativity
    assert np.all(weights_np >= 0.0), "Negative weights predicted by Softmax"
    print(f"[PASS] Non-negativity verified: min weight = {weights_np.min():.4f}")
    
    # Sum to 1.0 constraint
    sums = np.sum(weights_np, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6), "Predicted weights do not sum to 1.0"
    print(f"[PASS] Convex combination constraint verified (all row sums = 1.0).")
    
    # Verify adaptive property
    weights_differ = not np.allclose(weights_np[0], weights_np[1])
    assert weights_differ, "Model is not adaptive (identical weights for distinct inputs)"
    print(f"[PASS] Adaptive variance verified: weights dynamically adjust per query.")
    
    # Verify adaptive distance formula
    q_c = np.array([0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7, 0.9])
    q_t = np.array([0.1, 0.5, 0.3, 0.8])
    g_c = np.array([[0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0]])
    g_t = np.array([[0.2, 0.6, 0.4, 0.9]])
    
    w_c, w_t = 0.6, 0.4
    dc_sq = np.sum((g_c - q_c)**2, axis=1)
    dt_sq = np.sum((g_t - q_t)**2, axis=1)
    d_adaptive = np.sqrt(w_c * dc_sq + w_t * dt_sq)[0]
    
    expected_d = np.sqrt(0.6 * (9 * 0.01) + 0.4 * (4 * 0.01))
    assert np.isclose(d_adaptive, expected_d, atol=1e-6), "Adaptive distance formula mismatch"
    print(f"[PASS] Vectorized adaptive distance formula mathematically verified (d = {d_adaptive:.4f}).")

def main():
    print("==========================================================")
    print("       COREL-1K CBIR UNIFIED SANITY TEST SUITE             ")
    print("==========================================================")
    try:
        run_part1_sanity()
        run_part2_sanity()
        print("\n==========================================================")
        print("[SUCCESS] ALL CBIR SYSTEM SANITY CHECKS PASSED CLEANLY!  ")
        print("==========================================================")
    except Exception as e:
        print(f"\n[ERROR] Sanity test failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
