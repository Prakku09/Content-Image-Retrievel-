"""
Central Configuration Module for Corel-1K CBIR Pipeline
Single Source of Truth for all dataset, model, training, and evaluation parameters.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==========================================
# 1. Dataset Configuration
# ==========================================
# Configurable path to the external dataset directory
# Can be overridden by setting the environment variable COREL_DATASET_PATH
DEFAULT_DATASET_PATH = r"C:\Users\777Pr\Downloads\archive (1)\dataset"
DATASET_PATH = Path(os.getenv("COREL_DATASET_PATH", DEFAULT_DATASET_PATH)).resolve()

# Categories expected in the Corel-1K dataset (preserving local folder spellings)
EXPECTED_CATEGORIES = [
    "beaches",
    "bus",
    "dinosaurs",
    "elephants",
    "flowers",
    "foods",
    "horses",
    "monuments",
    "mountains_and_snow",
    "peolpe_and_villages_in_Africa"
]

# Supported image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ==========================================
# 2. Experiment & Split Parameters
# ==========================================
RANDOM_SEED = 42
PART2_SEED = 42

# Main benchmark split: 100 queries (10/cat) and 900 gallery images (90/cat)
GALLERY_IMAGES_PER_CAT = 90
QUERY_IMAGES_PER_CAT = 10

# Internal training/validation split on the 900 gallery images
INTERNAL_TRAIN_PER_CAT = 70
INTERNAL_VAL_PER_CAT = 20

# Evaluation Top-K
EVAL_K = 10

# ==========================================
# 3. Feature Extraction Parameters
# ==========================================
# GLCM Texture Parameters (Contrast, Correlation, Energy, Homogeneity)
GLCM_DISTANCE = 1
GLCM_ANGLES_DEG = [0, 45, 90, 135]

# HSV Color Moments (Mean, Std, Skewness for H, S, V) -> 9 dimensions
COLOR_FEATURE_DIM = 9
TEXTURE_FEATURE_DIM = 4
TOTAL_FEATURE_DIM = COLOR_FEATURE_DIM + TEXTURE_FEATURE_DIM  # 13 dimensions

# ==========================================
# 4. Baseline Model Fusion Weights
# ==========================================
# M3: Equal Fusion
M3_COLOR_WEIGHT = 0.5
M3_TEXTURE_WEIGHT = 0.5

# M4: Fixed Weight Fusion
M4_COLOR_WEIGHT = 0.7
M4_TEXTURE_WEIGHT = 0.3

# ==========================================
# 5. MLP Adaptive Model (M5) Parameters
# ==========================================
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
MARGIN = 0.15  # Triplet Margin Loss margin

# ==========================================
# 6. Directories & Output Paths
# ==========================================
OUTPUT_DIR = Path("output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PART2_OUTPUT_DIR = OUTPUT_DIR / "part2"
PART2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = PART2_OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PART2_PLOTS_DIR = PLOTS_DIR

FIGURES_DIR = Path("figures").resolve()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PKL_PATH = OUTPUT_DIR / "corel1k_features.pkl"
MODEL_CHECKPOINT_PATH = PART2_OUTPUT_DIR / "adaptive_weight_mlp.pt"
PART2_RESULTS_PATH = PART2_OUTPUT_DIR / "part2_results.pkl"
EXPERIMENT_RESULTS_PATH = PART2_OUTPUT_DIR / "experiment_results.pkl"
EXPERIMENT_CONFIG_PATH = PART2_OUTPUT_DIR / "experiment_config.json"
QUERY_WEIGHTS_CSV_PATH = PART2_OUTPUT_DIR / "query_weights.csv"
TRAINING_HISTORY_PATH = PART2_OUTPUT_DIR / "training_history.json"

# Separately versioned, prespecified statistical evaluation; does not overwrite
# the original M1-M6 experiment outputs.
STATISTICAL_OUTPUT_DIR = OUTPUT_DIR / "statistical_evaluation_v1"
STATISTICAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATISTICAL_SEEDS = [42, 123, 2025, 31415, 271828]
STATISTICAL_BOOTSTRAP_REPLICATES = 10000
STATISTICAL_PERMUTATION_REPLICATES = 100000
