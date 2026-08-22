import numpy as np
from pathlib import Path
import config

def load_dataset(dataset_path=None, seed=None):
    """
    Scans the dataset directory recursively, groups images by their category folder,
    and splits them deterministically into gallery and query sets.
    
    Args:
        dataset_path (Path or str): Path to the dataset directory.
        seed (int): Random seed for reproducibility of the split.
        
    Returns:
        tuple: (gallery_paths, gallery_labels, query_paths, query_labels)
               All paths are returned as absolute Path objects.
    """
    if dataset_path is None:
        dataset_path = config.DATASET_PATH
    if seed is None:
        seed = config.RANDOM_SEED
        
    dataset_path = Path(dataset_path)
    
    # 1. Scan recursively for all supported image file extensions (case-insensitive)
    image_paths = []
    for ext in config.IMAGE_EXTENSIONS:
        image_paths.extend(dataset_path.rglob(f"*{ext}"))
        image_paths.extend(dataset_path.rglob(f"*{ext.upper()}"))
        
    # Deduplicate and resolve absolute paths
    image_paths = sorted(list(set(p.resolve() for p in image_paths)))
    
    # 2. Group by immediate parent folder name (category)
    expected_cats = set(config.EXPECTED_CATEGORIES)
    category_to_paths = {cat: [] for cat in config.EXPECTED_CATEGORIES}
    
    for p in image_paths:
        cat_name = p.parent.name
        if cat_name in expected_cats:
            category_to_paths[cat_name].append(p)
            
    # Validate structure
    for cat in config.EXPECTED_CATEGORIES:
        count = len(category_to_paths[cat])
        if count != 100:
            raise ValueError(f"Category '{cat}' does not have exactly 100 images. Found {count} images.")
            
    # 3. Deterministic split using fixed random seed
    gallery_paths = []
    gallery_labels = []
    query_paths = []
    query_labels = []
    
    rng = np.random.default_rng(seed=seed)
    
    for cat in sorted(config.EXPECTED_CATEGORIES):
        # Sort paths to be 100% deterministic before shuffling
        paths = sorted(category_to_paths[cat])
        indices = np.arange(len(paths))
        rng.shuffle(indices)
        
        # Select first 10 indices as query, remaining 90 as gallery
        query_indices = set(indices[:config.QUERY_IMAGES_PER_CAT])
        
        for idx, path in enumerate(paths):
            if idx in query_indices:
                query_paths.append(path)
                query_labels.append(cat)
            else:
                gallery_paths.append(path)
                gallery_labels.append(cat)
                
    return gallery_paths, gallery_labels, query_paths, query_labels
