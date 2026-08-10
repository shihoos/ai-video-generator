from pathlib import Path


# ============================================================
# AI VIDEO GENERATOR — TECHNICAL CONFIGURATION
# ============================================================
#
# IMPORTANT:
# This file contains ONLY technical/project settings.
#
# Do NOT put:
#   - characters
#   - story
#   - environment
#   - camera angles
#   - visual style
#   - actions
#
# Those come from the story and optional reference files.
# ============================================================


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(
    "/kaggle/working/ai-video-generator"
)

GENERATE_SCRIPT = (
    PROJECT_ROOT
    / "kaggle"
    / "generate.py"
)


# ============================================================
# WORK DIRECTORIES
# ============================================================

WORK_DIR = (
    PROJECT_ROOT
    / "work"
)

CLIPS_DIR = (
    WORK_DIR
    / "story_clips"
)

OUTPUT_DIR = (
    WORK_DIR
    / "output"
)


# ============================================================
# ASSET DIRECTORIES
# ============================================================

ASSETS_DIR = (
    PROJECT_ROOT
    / "assets"
)

CHARACTER_DIR = (
    ASSETS_DIR
    / "characters"
)

REFERENCE_DIR = (
    ASSETS_DIR
    / "references"
)

AUDIO_DIR = (
    ASSETS_DIR
    / "audio"
)


# ============================================================
# VIDEO RESOLUTION
# ============================================================

# Final output resolution.
#
# 1280 x 720 = 720p

WIDTH = 1280
HEIGHT = 720


# ============================================================
# FRAME RATE
# ============================================================

# LTX source generation FPS.
#
# Keep this at 8 for the current T4 setup.
SOURCE_FPS = 8


# Final video FPS.
#
# 24 FPS gives a cinematic look.
#
# You can later change this to:
#
# FINAL_FPS = 25
#
# or:
#
# FINAL_FPS = 30
#
# without changing the LTX generation stage.

FINAL_FPS = 24


# ============================================================
# DEFAULT SHOT LENGTH
# ============================================================

# Default duration of a generated shot.
#
# IMPORTANT:
# LTX 0.9.8 expects:
#
# frames = N * 8 + 1
#
# Therefore:
#
# 5 seconds at 8 FPS:
#
# 5 * 8 = 40
# + 1
# = 41 frames
#
# 41 / 8 = 5.125 seconds

DEFAULT_SHOT_SECONDS = 5.0

FRAMES_PER_SHOT = (
    int(DEFAULT_SHOT_SECONDS * SOURCE_FPS) + 1
)


# ============================================================
# STORY / SHOT COUNT
# ============================================================

# IMPORTANT:
#
# There is NO fixed maximum number of shots.
#
# The story planner can produce:
#
# 4 shots
# 8 shots
# 10 shots
# 20 shots
# 50 shots
# etc.
#
# This value is only used when the planner needs
# a default estimate.

DEFAULT_SHOT_COUNT = None


# ============================================================
# LTX GENERATION
# ============================================================

USE_CPU_OFFLOAD = True


# ============================================================
# GENERATION SEED
# ============================================================

BASE_SEED = 20260810


# ============================================================
# OUTPUT ENCODING
# ============================================================

# H.264 quality.
#
# Lower CRF = higher quality / larger file.
#
# 18 is high quality.
OUTPUT_CRF = 18

OUTPUT_PRESET = "medium"

PIXEL_FORMAT = "yuv420p"


# ============================================================
# AUDIO
# ============================================================

# Optional background audio.
#
# Leave as None until audio is implemented.
#
# Example:
#
# DEFAULT_AUDIO = AUDIO_DIR / "cinematic_music.mp3"

DEFAULT_AUDIO = None


# ============================================================
# REFERENCE MATCHING
# ============================================================

# Allowed image formats.

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


# ============================================================
# CLEANUP
# ============================================================

# Delete temporary generated scene clips before starting
# a completely new generation.

CLEAN_TEMPORARY_CLIPS = True


# ============================================================
# HELPER
# ============================================================

def print_config():
    """Print active technical configuration."""

    print("=" * 70)
    print("VIDEO CONFIGURATION")
    print("=" * 70)

    print(f"Resolution       : {WIDTH}x{HEIGHT}")
    print(f"Source FPS       : {SOURCE_FPS}")
    print(f"Final FPS        : {FINAL_FPS}")
    print(f"Default shot     : {DEFAULT_SHOT_SECONDS:.2f} sec")
    print(f"Frames per shot  : {FRAMES_PER_SHOT}")
    print(f"CPU offload      : {USE_CPU_OFFLOAD}")
    print(f"Output CRF       : {OUTPUT_CRF}")

    print("=" * 70)
