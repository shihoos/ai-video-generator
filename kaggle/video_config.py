from pathlib import Path


# ============================================================
# AI VIDEO GENERATOR - CENTRAL TECHNICAL CONFIGURATION
# ============================================================
#
# This file contains technical/project settings only.
#
# DO NOT put here:
#   - story
#   - characters
#   - actions
#   - camera decisions
#   - locations
#   - visual descriptions
#
# Those are generated from the user's story by Qwen.
# Optional character/reference images come from assets/.
# ============================================================


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(
    "/kaggle/working/ai-video-generator"
)

KAGGLE_DIR = PROJECT_ROOT / "kaggle"

GENERATE_SCRIPT = (
    KAGGLE_DIR / "generate.py"
)


# ============================================================
# WORK DIRECTORIES
# ============================================================

WORK_DIR = (
    PROJECT_ROOT / "work"
)

CLIPS_DIR = (
    WORK_DIR / "story_clips"
)

OUTPUT_DIR = (
    WORK_DIR / "output"
)


# ============================================================
# ASSET DIRECTORIES
# ============================================================

ASSETS_DIR = (
    PROJECT_ROOT / "assets"
)

CHARACTER_DIR = (
    ASSETS_DIR / "characters"
)

REFERENCE_DIR = (
    ASSETS_DIR / "references"
)

AUDIO_DIR = (
    ASSETS_DIR / "audio"
)


# ============================================================
# VIDEO RESOLUTION
# ============================================================

# Final video resolution.

WIDTH = 1280
HEIGHT = 720


# ============================================================
# FRAME RATE
# ============================================================

# LTX source generation rate.
#
# Current T4-safe setup.

SOURCE_FPS = 8


# Final assembled video FPS.
#
# IMPORTANT:
# This creates a 25-FPS output file.
#
# It does NOT create genuine new motion information from
# an 8-FPS source.
#
# A future temporal interpolation stage can be added later.

FINAL_FPS = 25


# ============================================================
# STORY SHOT PLANNING
# ============================================================

# Qwen's normal planning duration.
#
# This is a STORYBOARD duration hint.
#
# It does not force LTX to generate this many seconds yet.

DEFAULT_SHOT_SECONDS = 5.0


# ============================================================
# LTX SOURCE FRAMES
# ============================================================

# Current T4-safe generation budget.
#
# 25 frames at 8 FPS:
#
#   25 / 8 = 3.125 seconds
#
# Therefore:
#
# DEFAULT_SHOT_SECONDS
#       ≠
# actual source clip duration
#
# at this stage.
#
# The planner can still decide that a shot should be
# approximately 5 seconds, but the current renderer uses
# this safe fixed frame budget.
#
# We will later implement temporal continuation / extension
# so planned duration can control actual generated duration.

FRAMES_PER_SHOT = 25


# ============================================================
# SHOT COUNT
# ============================================================

# There is NO fixed maximum number of shots.
#
# Qwen decides how many shots the story needs.
#
# This setting is only informational/default metadata.

DEFAULT_SHOT_COUNT = None


# ============================================================
# QWEN STORY PLANNER
# ============================================================

# Primary Hugging Face model.

QWEN_MODEL_ID = (
    "Qwen/Qwen3-4B-Instruct-2507"
)


# ============================================================
# LOCAL KAGGLE DATASET MODEL
# ============================================================

# Persistent Kaggle Dataset location.
#
# Expected structure:
#
# /kaggle/input/datasets/shihoos/
# └── ai-video-model/
#     └── qwen3-4b-instruct-2507/
#         ├── config.json
#         ├── tokenizer.json
#         ├── tokenizer_config.json
#         ├── model.safetensors / shards
#         └── ...

QWEN_LOCAL_MODEL = (
    Path("/kaggle/input/datasets/shihoos")
    / "ai-video-model"
    / "qwen3-4b-instruct-2507"
)


# Prefer the Kaggle Dataset copy.

USE_LOCAL_QWEN = True


# If the Dataset copy is unavailable,
# allow downloading from Hugging Face.

QWEN_ALLOW_HF_FALLBACK = True


# ============================================================
# QWEN GENERATION
# ============================================================

# Maximum number of tokens Qwen can generate
# for the storyboard JSON.

QWEN_MAX_NEW_TOKENS = 2500


# Sampling settings.

QWEN_DO_SAMPLE = False

QWEN_TEMPERATURE = 0.4

QWEN_TOP_P = 0.8

QWEN_TOP_K = 20


# ============================================================
# LTX GENERATION
# ============================================================

# Enable CPU offloading to reduce GPU memory pressure.

USE_CPU_OFFLOAD = True


# Base seed.
#
# Each shot uses:
#
# BASE_SEED + shot_number

BASE_SEED = 20260810


# ============================================================
# IMAGE REFERENCES
# ============================================================

# Supported image formats.

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


# ============================================================
# LTX IMAGE CONDITIONING
# ============================================================

# Reference strength:
#
# 0.0 = weak/no reference influence
# 1.0 = very strong reference influence

REFERENCE_STRENGTH = 0.85


# Frame at which reference conditioning begins.

REFERENCE_START_FRAME = 0


# ============================================================
# OUTPUT ENCODING
# ============================================================

# H.264 quality.
#
# Lower CRF = higher quality and larger file.
#
# 18 = high-quality output.

OUTPUT_CRF = 18

OUTPUT_PRESET = "medium"

PIXEL_FORMAT = "yuv420p"


# ============================================================
# FINAL VIDEO NAME
# ============================================================

FINAL_VIDEO_NAME = (
    "story_final.mp4"
)


# ============================================================
# AUDIO
# ============================================================

# Audio is intentionally disabled for the current version.
#
# Future architecture:
#
# Story
#   ↓
# Qwen
#   ↓
# Video shots
#   +
# Automatic music / ambience / SFX
#   ↓
# Final MP4 with audio
#
# Keep None until the audio engine is implemented.

DEFAULT_AUDIO = None


# ============================================================
# CLEANUP
# ============================================================

# Remove temporary story clips before starting a new story.
#
# Final videos in work/output are preserved.

CLEAN_TEMPORARY_CLIPS = False


# ============================================================
# VALIDATION
# ============================================================

def validate_config():
    """
    Validate all central technical configuration values.
    """

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    if WIDTH <= 0:
        raise ValueError(
            "WIDTH must be greater than 0."
        )

    if HEIGHT <= 0:
        raise ValueError(
            "HEIGHT must be greater than 0."
        )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    if SOURCE_FPS <= 0:
        raise ValueError(
            "SOURCE_FPS must be greater than 0."
        )

    if FINAL_FPS <= 0:
        raise ValueError(
            "FINAL_FPS must be greater than 0."
        )

    # --------------------------------------------------------
    # Shot settings
    # --------------------------------------------------------

    if DEFAULT_SHOT_SECONDS <= 0:
        raise ValueError(
            "DEFAULT_SHOT_SECONDS must be greater than 0."
        )

    if FRAMES_PER_SHOT <= 0:
        raise ValueError(
            "FRAMES_PER_SHOT must be greater than 0."
        )

    # --------------------------------------------------------
    # Qwen
    # --------------------------------------------------------

    if not QWEN_MODEL_ID.strip():
        raise ValueError(
            "QWEN_MODEL_ID cannot be empty."
        )

    if QWEN_MAX_NEW_TOKENS <= 0:
        raise ValueError(
            "QWEN_MAX_NEW_TOKENS must be greater than 0."
        )

    if QWEN_DO_SAMPLE:

        if QWEN_TEMPERATURE <= 0:
            raise ValueError(
                "QWEN_TEMPERATURE must be greater than 0 "
                "when sampling is enabled."
            )

        if not 0.0 < QWEN_TOP_P <= 1.0:
            raise ValueError(
                "QWEN_TOP_P must be between 0 and 1."
            )

        if QWEN_TOP_K <= 0:
            raise ValueError(
                "QWEN_TOP_K must be greater than 0."
            )

    # --------------------------------------------------------
    # Reference conditioning
    # --------------------------------------------------------

    if not 0.0 <= REFERENCE_STRENGTH <= 1.0:
        raise ValueError(
            "REFERENCE_STRENGTH must be between 0 and 1."
        )

    if REFERENCE_START_FRAME < 0:
        raise ValueError(
            "REFERENCE_START_FRAME cannot be negative."
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    if OUTPUT_CRF < 0:
        raise ValueError(
            "OUTPUT_CRF cannot be negative."
        )

    if not OUTPUT_PRESET.strip():
        raise ValueError(
            "OUTPUT_PRESET cannot be empty."
        )

    if not PIXEL_FORMAT.strip():
        raise ValueError(
            "PIXEL_FORMAT cannot be empty."
        )

    if not FINAL_VIDEO_NAME.strip():
        raise ValueError(
            "FINAL_VIDEO_NAME cannot be empty."
        )


# ============================================================
# CONFIGURATION DISPLAY
# ============================================================

def print_config():
    """
    Validate and print the active configuration.
    """

    validate_config()

    print()
    print("=" * 70)
    print("AI VIDEO GENERATOR CONFIGURATION")
    print("=" * 70)

    print()
    print("PROJECT")
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Generate script    : {GENERATE_SCRIPT}")

    print()
    print("VIDEO")
    print(f"Resolution         : {WIDTH}x{HEIGHT}")
    print(f"Source FPS         : {SOURCE_FPS}")
    print(f"Final FPS          : {FINAL_FPS}")

    print()
    print("SHOT PLANNING")
    print(
        f"Default shot       : "
        f"{DEFAULT_SHOT_SECONDS:.2f} sec"
    )
    print(
        f"Source frames      : "
        f"{FRAMES_PER_SHOT}"
    )
    print(
        f"Source duration    : "
        f"{FRAMES_PER_SHOT / SOURCE_FPS:.3f} sec"
    )
    print(
        f"Shot count limit   : "
        f"{DEFAULT_SHOT_COUNT}"
    )

    print()
    print("QWEN STORY PLANNER")
    print(f"Model ID           : {QWEN_MODEL_ID}")
    print(f"Local model        : {QWEN_LOCAL_MODEL}")
    print(f"Use local model    : {USE_LOCAL_QWEN}")
    print(f"HF fallback        : {QWEN_ALLOW_HF_FALLBACK}")
    print(
        f"Max new tokens     : "
        f"{QWEN_MAX_NEW_TOKENS}"
    )
    print(f"Sampling            : {QWEN_DO_SAMPLE}")
    print(f"Temperature         : {QWEN_TEMPERATURE}")
    print(f"Top P               : {QWEN_TOP_P}")
    print(f"Top K               : {QWEN_TOP_K}")

    print()
    print("LTX")
    print(f"CPU offload         : {USE_CPU_OFFLOAD}")
    print(f"Base seed           : {BASE_SEED}")

    print()
    print("IMAGE REFERENCES")
    print(
        f"Reference strength  : "
        f"{REFERENCE_STRENGTH}"
    )
    print(
        f"Reference start     : "
        f"{REFERENCE_START_FRAME}"
    )

    print()
    print("OUTPUT")
    print(f"CRF                 : {OUTPUT_CRF}")
    print(f"Preset              : {OUTPUT_PRESET}")
    print(f"Pixel format        : {PIXEL_FORMAT}")
    print(f"Final filename      : {FINAL_VIDEO_NAME}")

    print()
    print("AUDIO")
    print(f"Default audio       : {DEFAULT_AUDIO}")

    print()
    print("CLEANUP")
    print(
        f"Clean temporary     : "
        f"{CLEAN_TEMPORARY_CLIPS}"
    )

    print("=" * 70)


# ============================================================
# VALIDATE WHEN IMPORTED
# ============================================================

validate_config()
