import config
from pathlib import Path

# Seed for internal split and PyTorch
PART2_SEED = 42

# Training/Validation split on 900 gallery images
# 70 gallery images per category for internal train, 20 for internal validation
INTERNAL_TRAIN_PER_CAT = 70
INTERNAL_VAL_PER_CAT = 20

# MLP Training Parameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
MARGIN = 0.15  # Triplet Margin Loss margin

# Output directories for Part 2
PART2_OUTPUT_DIR = config.OUTPUT_DIR / "part2"
PART2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PART2_PLOTS_DIR = PART2_OUTPUT_DIR / "plots"
PART2_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
