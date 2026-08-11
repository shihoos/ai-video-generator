from pathlib import Path

# ============================================================
# AI VIDEO PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path("/kaggle/working/ai-video-generator")

# Persistent Kaggle Dataset
DATASET_ROOT = Path(
    "/kaggle/input/datasets/shihoos/ai-video-model"
)

# LTX models
LTX_MODEL = DATASET_ROOT / "ltxv-2b-0.9.8-distilled.safetensors"

LTX_UPSCALER = DATASET_ROOT / "ltxv-spatial-upscaler-0.9.8.safetensors"

# Exact LTX-Video revision
LTX_COMMIT = "bdc8f01"

# LTX repository
LTX_REPO = PROJECT_ROOT / "LTX-Video-0.9.8"

# Official LTX 0.9.8 distilled configuration
LTX_CONFIG = LTX_REPO / "configs" / "ltxv-2b-0.9.8-distilled.yaml"

# Temporary working directories
WORK_DIR = PROJECT_ROOT / "work"
OUTPUT_DIR = WORK_DIR / "output"

# Make sure directories exist

    directory.mkdir(parents=True, exist_ok=True)
