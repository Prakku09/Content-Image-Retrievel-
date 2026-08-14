import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

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

# Experiment Parameters
RANDOM_SEED = 42
GALLERY_IMAGES_PER_CAT = 90
QUERY_IMAGES_PER_CAT = 10
EVAL_K = 10

# GLCM Texture Parameters
GLCM_DISTANCE = 1
GLCM_ANGLES_DEG = [0, 45, 90, 135]

# Baseline Model Fusion Weights
M3_COLOR_WEIGHT = 0.5
M3_TEXTURE_WEIGHT = 0.5

M4_COLOR_WEIGHT = 0.7
M4_TEXTURE_WEIGHT = 0.3

# Output Directory
OUTPUT_DIR = Path("output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
