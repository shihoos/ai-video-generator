from pathlib import Path


# ============================================================
# AI VIDEO GENERATOR - CENTRAL TECHNICAL CONFIGURATION
# ============================================================
#
# Keep this file technical only.
# Story, characters, actions, camera decisions and visual
# descriptions come from the story planner and references.
# ============================================================


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path("/kaggle/working/ai-video-generator")
KAGGLE_DIR = PROJECT_ROOT / "kaggle"

GENERATE_SCRIPT = KAGGLE_DIR / "generate.py"


# ============================================================
# WORK DIRECTORIES
# ============================================================

WORK_DIR = PROJECT_ROOT / "work"
CLIPS_DIR = WORK_DIR / "story_clips"
OUTPUT_DIR = WORK_DIR / "output"


# ============================================================
# ASSET DIRECTORIES
# ============================================================

ASSETS_DIR = PROJECT_ROOT / "assets"

CHARACTER_DIR = ASSETS_DIR / "characters"
REFERENCE_DIR = ASSETS_DIR / "references"
AUDIO_DIR = ASSETS_DIR / "audio"


# ============================================================
# VIDEO
# ============================================================

WIDTH = 1280
HEIGHT = 720

# LTX source generation rate for the current T4-safe setup.
SOURCE_FPS = 8

# Final assembled file.
FINAL_FPS = 25


# ============================================================
# STORY SHOTS
# ============================================================

# Qwen uses this as the normal planning duration.
DEFAULT_SHOT_SECONDS = 5.0

# Current T4-safe source generation.
#
# 25 frames / 8 FPS = 3.125 seconds of actual source motion.
#
# FINAL_FPS = 25 changes the output container/frame rate, but
# does not create new motion information.
#
# Keep this at 25 until temporal continuation is implemented.
FRAMES_PER_SHOT = 25

# No fixed shot-count limit.
DEFAULT_SHOT_COUNT = None


# ============================================================
# QWEN STORY PLANNER
# ============================================================

# Hugging Face fallback model.
QWEN_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Kaggle Dataset copy.
QWEN_LOCAL_MODEL = (
    Path("/kaggle/input/datasets/shihoos")
    / "ai-video-model"
    / "qwen3-4b-instruct-2507"
)

# Prefer the persistent Dataset copy.
USE_LOCAL_QWEN = True

# If the Dataset copy is missing, allow Hugging Face fallback.
QWEN_ALLOW_HF_FALLBACK = True

# Planner output budget.
QWEN_MAX_NEW_TOKENS = 2500

# Sampling gives Qwen3-Instruct more natural planning.
# Set False if completely deterministic planning is preferred.
QWEN_DO_SAMPLE = True

QWEN_TEMPERATURE = 0.4
QWEN_TOP_P = 0.8
QWEN_TOP_K = 20


# ============================================================
# LTX
# ============================================================

USE_CPU_OFFLOAD = True

BASE_SEED = 20260810


# ============================================================
# IMAGE REFERENCES
# ============================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

# LTX image-conditioning defaults.
REFERENCE_STRENGTH = 0.85
REFERENCE_START_FRAME = 0


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_CRF = 18
OUTPUT_PRESET = "medium"
PIXEL_FORMAT = "yuv420p"

FINAL_VIDEO_NAME = "story_final.mp4"


# ============================================================
# AUDIO
# ============================================================
#
# Reserved for the future automatic audio engine.
# None means no audio is currently attached.
# ============================================================

DEFAULT_AUDIO = None


# ============================================================
# CLEANUP
# ============================================================

# Remove temporary shot clips before a new story generation.
# Final files in work/output are preserved.
CLEAN_TEMPORARY_CLIPS = True


# ============================================================
# VALIDATION
# ============================================================

def validate_config():
    if WIDTH <= 0 or HEIGHT <= 0:
        raise ValueError("WIDTH and HEIGHT must be positive.")

    if SOURCE_FPS <= 0 or FINAL_FPS <= 0:
        raise ValueError("SOURCE_FPS and FINAL_FPS must be positive.")

    if DEFAULT_SHOT_SECONDS <= 0:
        raise ValueError("DEFAULT_SHOT_SECONDS must be positive.")

    if FRAMES_PER_SHOT <= 0:
        raise ValueError("FRAMES_PER_SHOT must be positive.")

    if OUTPUT_CRF < 0:
        raise ValueError("OUTPUT_CRF cannot be negative.")

    if not QWEN_MODEL_ID:
        raise ValueError("QWEN_MODEL_ID cannot be empty.")

    if not 0.0 <= REFERENCE_STRENGTH <= 1.0:
        raise ValueError(
            "REFERENCE_STRENGTH must be between 0 and 1."
        )


def print_config():
    validate_config()

    print("=" * 70)
    print("AI VIDEO GENERATOR CONFIGURATION")
    print("=" * 70)

    print(f"Project             : {PROJECT_ROOT}")
    print(f"Resolution          : {WIDTH}x{HEIGHT}")
    print(f"Source FPS          : {SOURCE_FPS}")
    print(f"Final FPS           : {FINAL_FPS}")
    print(f"Default shot        : {DEFAULT_SHOT_SECONDS:.2f} sec")
    print(f"Source frames/shot  : {FRAMES_PER_SHOT}")
    print(f"CPU offload         : {USE_CPU_OFFLOAD}")
    print(f"Qwen model          : {QWEN_MODEL_ID}")
    print(f"Qwen local path     : {QWEN_LOCAL_MODEL}")
    print(f"Use local Qwen      : {USE_LOCAL_QWEN}")
    print(f"HF fallback         : {QWEN_ALLOW_HF_FALLBACK}")
    print(f"Reference strength  : {REFERENCE_STRENGTH}")
    print(f"Output CRF          : {OUTPUT_CRF}")
    print(f"Output filename     : {FINAL_VIDEO_NAME}")
    print("=" * 70)


validate_config()
