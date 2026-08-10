from pathlib import Path


# ============================================================
# AI VIDEO PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path("/kaggle/working/ai-video-generator")

# Persistent Kaggle Dataset
DATASET_ROOT = Path(
    "/kaggle/input/datasets/shihoos/ai-video-model"
)

# LTX model
LTX_MODEL = DATASET_ROOT / "ltxv-2b-0.9.8-distilled.safetensors"

# LTX spatial upscaler
LTX_UPSCALER = DATASET_ROOT / "ltxv-spatial-upscaler-0.9.8.safetensors"

# LTX repository
LTX_REPO = PROJECT_ROOT / "LTX-Video-0.9.8"

# Temporary working directories
WORK_DIR = PROJECT_ROOT / "work"
CLIPS_DIR = WORK_DIR / "clips"
FRAMES_DIR = WORK_DIR / "frames"
OUTPUT_DIR = WORK_DIR / "output"

# Make sure these directories exist
for directory in [
    WORK_DIR,
    CLIPS_DIR,
    FRAMES_DIR,
    OUTPUT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
