import numpy as np
import torch
import pickle
from pathlib import Path
from mlp_model import AdaptiveWeightMLP

def main():
    print("=== STARTING PART 2 ADAPTIVE WEIGHTING (M5) SANITY CHECK ===")
    
    # 1. Load some features from Part 1 output
    pkl_path = Path("output/corel1k_features.pkl")
    if not pkl_path.exists():
        print(f"Error: {pkl_path} not found. Please run Part 1 first.")
        return
        
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    # Get normalized gallery features
    norm_color = data['normalized_gallery_color']
    norm_texture = data['normalized_gallery_texture']
    gallery_paths = data['gallery_image_paths']
    
    # Construct combined 13-dim features
    combined_feats = np.hstack([norm_color, norm_texture])
    print(f"Loaded normalized features. Combined feature matrix shape: {combined_feats.shape}")
    
    # Instantiate MLP
    model = AdaptiveWeightMLP()
    model.eval()  # set to evaluation mode
    
    # Select a batch of 5 different images from different categories to test
    indices = [0, 100, 200, 300, 400]
    batch_features_np = combined_feats[indices]
    batch_features = torch.tensor(batch_features_np, dtype=torch.float32)
    
    # 2. Check input/output shapes
    print("\n1. Verifying MLP input/output shapes...")
    print(f"MLP input shape: {batch_features.shape} (Expected: (batch_size, 13))")
    assert batch_features.shape == (5, 13), f"Incorrect input shape: {batch_features.shape}"
    
    with torch.no_grad():
        weights = model(batch_features)
    print(f"MLP output shape: {weights.shape} (Expected: (batch_size, 2))")
    assert weights.shape == (5, 2), f"Incorrect output shape: {weights.shape}"
    
    weights_np = weights.numpy()
    
    # 3. Check weights are non-negative
    print("\n2. Verifying weights are non-negative...")
    print(f"Min weight value: {weights_np.min():.6f}")
    assert np.all(weights_np >= 0.0), "Found negative weights!"
    print("Verification passed: All weights are non-negative.")
    
    # 4. Check color weight + texture weight = 1
    print("\n3. Verifying that w_colour + w_texture = 1 for each image...")
    sums = np.sum(weights_np, axis=1)
    print(f"Row weight sums: {sums}")
    assert np.allclose(sums, 1.0, atol=1e-6), "Weight sums do not equal 1.0!"
    print("Verification passed: Weights sum to 1.0.")
    
    # 5. Verify that weights vary for different inputs (adaptive property)
    print("\n4. Verifying that weights vary for different input images...")
    for i, idx in enumerate(indices):
        path = gallery_paths[idx]
        print(f"  Image {idx} ({path}): w_colour = {weights_np[i, 0]:.6f}, w_texture = {weights_np[i, 1]:.6f}")
        
    all_equal = np.allclose(weights_np[0], weights_np[1]) and np.allclose(weights_np[1], weights_np[2])
    print(f"Are all weights identical? {all_equal}")
    assert not all_equal, "Weights are identical for all images. The model is not adaptive."
    print("Verification passed: Weights vary for different inputs.")
    
    # 6. Verify adaptive distance is calculated correctly
    print("\n5. Verifying adaptive distance calculation...")
    q_color = norm_color[indices[0]]
    q_texture = norm_texture[indices[0]]
    x_color = norm_color[indices[1]]
    x_texture = norm_texture[indices[1]]
    
    w_col = weights_np[0, 0]
    w_tex = weights_np[0, 1]
    
    # Calculate separate distances
    D_c_val = np.sqrt(np.sum((q_color - x_color)**2))
    D_t_val = np.sqrt(np.sum((q_texture - x_texture)**2))
    
    # Calculate adaptive distance manually
    d_adaptive_manual = np.sqrt(w_col * D_c_val**2 + w_tex * D_t_val**2)
    print(f"Manual Calculation:")
    print(f"  w_colour = {w_col:.6f}, w_texture = {w_tex:.6f}")
    print(f"  D_colour = {D_c_val:.6f}, D_texture = {D_t_val:.6f}")
    print(f"  Manual D_adaptive = {d_adaptive_manual:.6f}")
    
    # Calculate using a vectorized function
    def compute_adaptive_distance(q_c, q_t, g_c, g_t, w_c, w_t):
        d_c_sq = np.sum((g_c - q_c)**2, axis=1)
        d_t_sq = np.sum((g_t - q_t)**2, axis=1)
        return np.sqrt(w_c * d_c_sq + w_t * d_t_sq)
        
    d_adaptive_func = compute_adaptive_distance(
        q_color, q_texture,
        norm_color[indices[1:2]], norm_texture[indices[1:2]],
        w_col, w_tex
    )[0]
    print(f"Function Calculation: D_adaptive = {d_adaptive_func:.6f}")
    assert np.allclose(d_adaptive_manual, d_adaptive_func, atol=1e-6), "Manual and functional distance calculations do not match!"
    print("Verification passed: Adaptive distance calculation is correct.")
    
    # 7. Check for NaN/Inf values
    print("\n6. Verifying no NaN or Inf values occur in weights or distances...")
    has_nan_weights = np.isnan(weights_np).any()
    has_inf_weights = np.isinf(weights_np).any()
    print(f"Weights contain NaN: {has_nan_weights}, contains Inf: {has_inf_weights}")
    
    # Test distance computation for all 5 queries against all 900 gallery images
    all_dists = []
    for i in range(5):
        w_c = weights_np[i, 0]
        w_t = weights_np[i, 1]
        q_c = norm_color[indices[i]]
        q_t = norm_texture[indices[i]]
        dists = compute_adaptive_distance(q_c, q_t, norm_color, norm_texture, w_c, w_t)
        all_dists.append(dists)
    all_dists = np.array(all_dists)
    
    has_nan_dists = np.isnan(all_dists).any()
    has_inf_dists = np.isinf(all_dists).any()
    print(f"Distances contain NaN: {has_nan_dists}, contains Inf: {has_inf_dists}")
    
    assert not has_nan_weights, "NaN found in weights!"
    assert not has_inf_weights, "Inf found in weights!"
    assert not has_nan_dists, "NaN found in distances!"
    assert not has_inf_dists, "Inf found in distances!"
    
    print("\n=== SANITY CHECK PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    main()
