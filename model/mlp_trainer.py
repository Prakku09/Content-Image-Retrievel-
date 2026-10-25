import copy
import hashlib
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
import random
import config

try:
    from model.mlp_model import AdaptiveWeightMLP
except ImportError:
    from mlp_model import AdaptiveWeightMLP


class TripletDataset(Dataset):
    """
    Dataset wrapping triplets (q, p, n) and their precalculated color and texture distances.
    """
    def __init__(self, q_feats, d_color_pos, d_texture_pos, d_color_neg, d_texture_neg):
        self.q_feats = torch.tensor(q_feats, dtype=torch.float32)
        self.d_color_pos = torch.tensor(d_color_pos, dtype=torch.float32)
        self.d_texture_pos = torch.tensor(d_texture_pos, dtype=torch.float32)
        self.d_color_neg = torch.tensor(d_color_neg, dtype=torch.float32)
        self.d_texture_neg = torch.tensor(d_texture_neg, dtype=torch.float32)
        
    def __len__(self):
        return len(self.q_feats)
        
    def __getitem__(self, idx):
        return (
            self.q_feats[idx],
            self.d_color_pos[idx],
            self.d_texture_pos[idx],
            self.d_color_neg[idx],
            self.d_texture_neg[idx]
        )

def get_internal_split(gallery_labels, seed=config.PART2_SEED):
    """
    Deterministically splits the 900 gallery images into 700 training (70/cat)
    and 200 validation (20/cat) indices.
    """
    gallery_labels = np.array(gallery_labels)
    train_indices = []
    val_indices = []
    
    rng = np.random.default_rng(seed=seed)
    
    # Each category has 90 images in the gallery
    categories = sorted(list(set(gallery_labels)))
    for c_idx, cat in enumerate(categories):
        # Find indices of this category in the gallery
        cat_indices = np.where(gallery_labels == cat)[0]
        # Sort to ensure absolute determinism before shuffle
        cat_indices = np.sort(cat_indices)
        
        # Shuffle indices
        shuffled = cat_indices.copy()
        rng.shuffle(shuffled)
        
        # 70 for training, 20 for validation
        train_indices.extend(shuffled[:config.INTERNAL_TRAIN_PER_CAT])
        val_indices.extend(shuffled[config.INTERNAL_TRAIN_PER_CAT:])
        
    return sorted(train_indices), sorted(val_indices)

def build_triplets(indices, color_feats, texture_feats, labels, num_p=5, num_n=10, seed=config.PART2_SEED):
    """
    Generates training/validation triplets (q, p, n) and precalculates distances.
    """
    rng = np.random.default_rng(seed=seed)
    labels = np.array(labels)
    
    q_feats_list = []
    d_color_pos_list = []
    d_texture_pos_list = []
    d_color_neg_list = []
    d_texture_neg_list = []
    
    # Store category to index mapping for the allowed subset
    cat_to_indices = {}
    for idx in indices:
        lbl = labels[idx]
        if lbl not in cat_to_indices:
            cat_to_indices[lbl] = []
        cat_to_indices[lbl].append(idx)
        
    for q_idx in indices:
        q_label = labels[q_idx]
        q_c = color_feats[q_idx]
        q_t = texture_feats[q_idx]
        
        # Allowed positives (same category, excluding query itself)
        pos_candidates = [idx for idx in cat_to_indices[q_label] if idx != q_idx]
        # Allowed negatives (different categories)
        neg_candidates = [idx for idx in indices if labels[idx] != q_label]
        
        # Randomly sample positives and negatives
        selected_pos = rng.choice(pos_candidates, size=min(num_p, len(pos_candidates)), replace=False)
        selected_neg = rng.choice(neg_candidates, size=min(num_n, len(neg_candidates)), replace=False)
        
        # Combine query color and texture features
        q_feat = np.concatenate([q_c, q_t])
        
        # Generate triplets
        for p_idx in selected_pos:
            p_c = color_feats[p_idx]
            p_t = texture_feats[p_idx]
            
            d_c_pos_sq = np.sum((q_c - p_c)**2)
            d_t_pos_sq = np.sum((q_t - p_t)**2)
            
            for n_idx in selected_neg:
                n_c = color_feats[n_idx]
                n_t = texture_feats[n_idx]
                
                d_c_neg_sq = np.sum((q_c - n_c)**2)
                d_t_neg_sq = np.sum((q_t - n_t)**2)
                
                # Append to lists
                q_feats_list.append(q_feat)
                d_color_pos_list.append(d_c_pos_sq)
                d_texture_pos_list.append(d_t_pos_sq)
                d_color_neg_list.append(d_c_neg_sq)
                d_texture_neg_list.append(d_t_neg_sq)
                
    return (
        np.array(q_feats_list),
        np.array(d_color_pos_list),
        np.array(d_texture_pos_list),
        np.array(d_color_neg_list),
        np.array(d_texture_neg_list)
    )

def train_mlp(
    data_path=None,
    seed=config.PART2_SEED,
    checkpoint_path=None,
    split_indices=None,
):
    """
    Orchestrates the entire training protocol, splits the gallery, builds triplets,
    trains the PyTorch model with early stopping, and returns training history.
    """
    if data_path is None:
        data_path = config.OUTPUT_DIR / "corel1k_features.pkl"
        
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 1. Load Part 1 Features
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    feature_fingerprint = data.get('feature_fingerprint')
    if not feature_fingerprint:
        raise ValueError(
            "Feature cache has no verified fingerprint; rerun the feature stage before training."
        )

    gallery_color = data['normalized_gallery_color']
    gallery_texture = data['normalized_gallery_texture']
    gallery_labels = [Path(p).parent.name for p in data['gallery_image_paths']]
    if split_indices is None:
        train_idx, val_idx = get_internal_split(gallery_labels, seed=seed)
        split_mode = 'stratified_random'
    else:
        if not isinstance(split_indices, dict):
            raise ValueError("split_indices must contain train_indices and val_indices.")
        try:
            train_idx = [int(index) for index in split_indices['train_indices']]
            val_idx = [int(index) for index in split_indices['val_indices']]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "split_indices must contain valid train_indices and val_indices."
            ) from error
        split_mode = 'explicit_group_disjoint'

    expected_indices = set(range(len(gallery_labels)))
    if (
        not train_idx
        or not val_idx
        or len(set(train_idx)) != len(train_idx)
        or len(set(val_idx)) != len(val_idx)
        or set(train_idx) & set(val_idx)
        or set(train_idx) | set(val_idx) != expected_indices
        or any(
            index < 0 or index >= len(gallery_labels)
            for index in train_idx + val_idx
        )
    ):
        raise ValueError(
            "Internal training/validation indices must form a disjoint gallery partition."
        )

    split_fingerprint = hashlib.sha256(json.dumps({
        'mode': split_mode,
        'train_indices': sorted(train_idx),
        'val_indices': sorted(val_idx),
    }, sort_keys=True).encode('utf-8')).hexdigest()
    model_source_fingerprint = _model_source_fingerprint()
    training_config = {
        'training_protocol_version': 3,
        'seed': int(seed),
        'batch_size': config.BATCH_SIZE,
        'epochs': config.EPOCHS,
        'learning_rate': config.LEARNING_RATE,
        'margin': config.MARGIN,
        'patience': 10,
        'train_per_category': config.INTERNAL_TRAIN_PER_CAT,
        'validation_per_category': config.INTERNAL_VAL_PER_CAT,
        'positive_samples_per_query': 5,
        'negative_samples_per_query': 10,
        'validation_positive_samples_per_query': 2,
        'validation_negative_samples_per_query': 5,
        'model_dimensions': [config.TOTAL_FEATURE_DIM, 16, 8, 2],
        'model_source_fingerprint': model_source_fingerprint,
        'split_mode': split_mode,
        'split_fingerprint': split_fingerprint,
        'train_count': len(train_idx),
        'validation_count': len(val_idx),
    }

    model_save_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else config.MODEL_CHECKPOINT_PATH
    )
    if model_save_path.exists():
        try:
            checkpoint = torch.load(
                model_save_path, map_location='cpu', weights_only=True
            )
        except pickle.UnpicklingError:
            checkpoint = None
        if (
            isinstance(checkpoint, dict)
            and checkpoint.get('checkpoint_format') == 2
            and checkpoint.get('feature_fingerprint') == feature_fingerprint
            and checkpoint.get('training_config') == training_config
        ):
            model = AdaptiveWeightMLP()
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Reusing verified checkpoint: {model_save_path}")
            return model, checkpoint['split_info'], checkpoint['training_history']
        print(
            "Existing checkpoint is unreadable by safe weights-only loading or "
            "lacks matching provenance; training a fresh model."
        )
    
    # 2. Get deterministic internal split
    print("Generating deterministic internal training/validation split...")
    train_idx = [int(index) for index in train_idx]
    val_idx = [int(index) for index in val_idx]
    
    # 3. Build Triplets
    print("Building training and validation triplets...")
    train_triplets = build_triplets(train_idx, gallery_color, gallery_texture, gallery_labels, num_p=5, num_n=10, seed=seed)
    val_triplets = build_triplets(val_idx, gallery_color, gallery_texture, gallery_labels, num_p=2, num_n=5, seed=seed)
    
    train_dataset = TripletDataset(*train_triplets)
    val_dataset = TripletDataset(*val_triplets)
    
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=loader_generator
    )
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Constructed {len(train_dataset)} training triplets and {len(val_dataset)} validation triplets.")
    
    # 4. Instantiate Model, Optimizer, and Loss params
    model = AdaptiveWeightMLP()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    margin = config.MARGIN
    
    # Early stopping trackers
    best_val_loss = float('inf')
    best_epoch = 0
    patience = training_config['patience']
    patience_counter = 0
    best_weights = None
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # 5. Training Loop
    print("\nTraining MLP adaptive weighting model...")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss_accum = 0.0
        
        for q_feats, dc_pos, dt_pos, dc_neg, dt_neg in train_loader:
            optimizer.zero_grad()
            
            # Predict weights: (batch_size, 2)
            weights = model(q_feats)
            w_c = weights[:, 0]
            w_t = weights[:, 1]
            
            # Compute adaptive squared distances
            pos_dist_sq = w_c * dc_pos + w_t * dt_pos
            neg_dist_sq = w_c * dc_neg + w_t * dt_neg
            
            # Triplet margin loss
            losses = torch.clamp(pos_dist_sq - neg_dist_sq + margin, min=0.0)
            loss = losses.mean()
            
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item() * len(q_feats)
            
        train_loss = train_loss_accum / len(train_dataset)
        
        # Validation epoch
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for q_feats, dc_pos, dt_pos, dc_neg, dt_neg in val_loader:
                weights = model(q_feats)
                w_c = weights[:, 0]
                w_t = weights[:, 1]
                
                pos_dist_sq = w_c * dc_pos + w_t * dt_pos
                neg_dist_sq = w_c * dc_neg + w_t * dt_neg
                
                losses = torch.clamp(pos_dist_sq - neg_dist_sq + margin, min=0.0)
                val_loss_accum += losses.mean().item() * len(q_feats)
                
        val_loss = val_loss_accum / len(val_dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch}. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}.")
                break
                
    # Restore best weights
    model.load_state_dict(best_weights)
    
    # Save the trained model
    split_info = {
        'train_indices': train_idx,
        'val_indices': val_idx,
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss)
    }
    checkpoint = {
        'checkpoint_format': 2,
        'feature_fingerprint': feature_fingerprint,
        'training_config': training_config,
        'model_state_dict': model.state_dict(),
        'split_info': split_info,
        'training_history': history,
    }
    torch.save(checkpoint, model_save_path)
    print(f"Best model weights saved to: {model_save_path}")
    
    return model, split_info, history


def _model_source_fingerprint():
    """Fingerprint model and trainer code so incompatible checkpoints are not reused."""
    source_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in (
        source_root / 'mlp_model.py',
        Path(__file__).resolve(),
    ):
        digest.update(path.name.encode('utf-8'))
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()
