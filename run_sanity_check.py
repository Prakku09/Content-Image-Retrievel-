import numpy as np
from pathlib import Path
import config
from dataset_loader import load_dataset
from feature_extractor import extract_all_features
from normalization import MinMaxNormalizer
from retrieval import get_representation, retrieve_images

def main():
    print("=== STARTING CBIR PIPELINE SANITY TEST ===")
    
    # 1. Load dataset split
    print("\nLoading dataset and creating query/gallery splits...")
    gallery_paths, gallery_labels, query_paths, query_labels = load_dataset()
    print(f"Loaded {len(gallery_paths)} gallery images and {len(query_paths)} query images.")
    
    # Select 2 gallery images and 1 query image from 3 distinct categories for sanity check
    selected_cats = ["beaches", "dinosaurs", "flowers"]
    
    sanity_gallery_paths = []
    sanity_gallery_labels = []
    sanity_query_paths = []
    sanity_query_labels = []
    
    # Find matching items
    for cat in selected_cats:
        cat_gal = [p for p, l in zip(gallery_paths, gallery_labels) if l == cat][:2]
        cat_q = [p for p, l in zip(query_paths, query_labels) if l == cat][:1]
        
        sanity_gallery_paths.extend(cat_gal)
        sanity_gallery_labels.extend([cat] * len(cat_gal))
        
        sanity_query_paths.extend(cat_q)
        sanity_query_labels.extend([cat] * len(cat_q))
        
    print(f"Sanity Check Subset: {len(sanity_gallery_paths)} gallery images, {len(sanity_query_paths)} query images.")
    
    # 2. Extract features
    print("\nExtracting features for sanity subset...")
    gal_color = []
    gal_texture = []
    for p in sanity_gallery_paths:
        c, t = extract_all_features(p)
        gal_color.append(c)
        gal_texture.append(t)
        
    q_color = []
    q_texture = []
    for p in sanity_query_paths:
        c, t = extract_all_features(p)
        q_color.append(c)
        q_texture.append(t)
        
    gal_color = np.array(gal_color)
    gal_texture = np.array(gal_texture)
    q_color = np.array(q_color)
    q_texture = np.array(q_texture)
    
    # Verify dimensions
    print(f"Number of test images: {len(sanity_gallery_paths) + len(sanity_query_paths)} (Gallery: {len(sanity_gallery_paths)}, Query: {len(sanity_query_paths)})")
    print(f"Gallery Color (HSV) shape: {gal_color.shape} (Expected: (6, 9))")
    print(f"Gallery Texture (GLCM) shape: {gal_texture.shape} (Expected: (6, 4))")
    print(f"Query Color (HSV) shape: {q_color.shape} (Expected: (3, 9))")
    print(f"Query Texture (GLCM) shape: {q_texture.shape} (Expected: (3, 4))")
    
    assert gal_color.shape == (6, 9), "HSV features must have dimension 9."
    assert gal_texture.shape == (6, 4), "GLCM features must have dimension 4."
    
    # Verify no NaN or infinite values
    nan_count = np.isnan(gal_color).sum() + np.isnan(gal_texture).sum() + np.isnan(q_color).sum() + np.isnan(q_texture).sum()
    inf_count = np.isinf(gal_color).sum() + np.isinf(gal_texture).sum() + np.isinf(q_color).sum() + np.isinf(q_texture).sum()
    print(f"NaN count: {nan_count}")
    print(f"Inf count: {inf_count}")
    assert nan_count == 0, "NaN values found in gallery color features."
    assert inf_count == 0, "Infinite values found in gallery color features."
    print("Dimension and NaN/Inf verification passed!")
    
    # 3. Normalization
    print("\nFitting Min-Max Normalizer on gallery features and transforming...")
    color_norm = MinMaxNormalizer()
    texture_norm = MinMaxNormalizer()
    
    norm_gal_color = color_norm.fit_transform(gal_color)
    norm_q_color = color_norm.transform(q_color)
    
    norm_gal_texture = texture_norm.fit_transform(gal_texture)
    norm_q_texture = texture_norm.transform(q_texture)
    
    # Verify normalization bounds
    print(f"Normalized Gallery Color range: [{norm_gal_color.min()}, {norm_gal_color.max()}]")
    print(f"Normalized Gallery Texture range: [{norm_gal_texture.min()}, {norm_gal_texture.max()}]")
    
    # Values should be between 0 and 1 (with float tolerance)
    assert np.all(norm_gal_color >= -1e-7) and np.all(norm_gal_color <= 1.0000001), "Normalized color features outside [0, 1]."
    assert np.all(norm_gal_texture >= -1e-7) and np.all(norm_gal_texture <= 1.0000001), "Normalized texture features outside [0, 1]."
    print("Normalization verification passed!")
    
    # 4. Fusion and Representation
    print("\nBuilding fusion representations...")
    # Test representation building for M3 and M4
    gal_m3 = get_representation(norm_gal_color, norm_gal_texture, 'M3')
    q_m3 = get_representation(norm_q_color, norm_q_texture, 'M3')
    
    gal_m4 = get_representation(norm_gal_color, norm_gal_texture, 'M4')
    q_m4 = get_representation(norm_q_color, norm_q_texture, 'M4')
    
    print(f"Gallery M3 shape: {gal_m3.shape} (Expected: (6, 13))")
    print(f"Query M3 shape: {q_m3.shape} (Expected: (3, 13))")
    assert gal_m3.shape == (6, 13), "Combined M3 representation must have dimension 13."
    assert gal_m4.shape == (6, 13), "Combined M4 representation must have dimension 13."
    print("Combined representation dimension verification passed!")
    
    # 5. Retrieval & Preservation of Labels
    print("\nTesting Euclidean distance retrieval and label preservation...")
    # Use first query as a test case
    test_q_feat = q_m4[0]
    test_q_label = sanity_query_labels[0]
    test_q_path = sanity_query_paths[0]
    
    ranked_idx, dists = retrieve_images(test_q_feat, gal_m4)
    
    print(f"Query image: {test_q_path.name} (Category: {test_q_label})")
    print("Ranked Gallery Retrievals:")
    for rank, (idx, dist) in enumerate(zip(ranked_idx, dists)):
        retrieved_label = sanity_gallery_labels[idx]
        retrieved_path = sanity_gallery_paths[idx]
        is_relevant = (retrieved_label == test_q_label)
        print(f"  Rank {rank+1}: {retrieved_path.name} (Category: {retrieved_label}, Dist: {dist:.4f}, Relevant: {is_relevant})")
        
    # Check if retrieval returns correct types and order
    assert len(ranked_idx) == len(sanity_gallery_paths), "Retrieval did not rank all gallery images."
    assert np.all(np.diff(dists) >= 0), "Euclidean retrieval distances are not sorted in ascending order."
    
    # Check if labels are preserved
    assert [sanity_gallery_labels[idx] for idx in ranked_idx], "Category labels were lost or corrupted during indexing."
    
    print("\n=== SANITY TEST PASSED SUCCESSFULLY! CBIR PIPELINE IS CORRECT AND READY ===")

if __name__ == "__main__":
    main()
