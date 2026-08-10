import argparse
import gc
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


# ============================================================
# AI STORY VIDEO GENERATOR
# ============================================================
#
# STORY
#   ↓
# Qwen3 story understanding
#   ↓
# Characters / locations / actions
#   ↓
# Cinematic shot planning
#   ↓
# Reference matching
#   ↓
# LTX video generation
#   ↓
# Multiple clips
#   ↓
# FFmpeg
#   ↓
# 25 FPS final video
#
# Audio will be added as a separate engine later.
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/kaggle/working/ai-video-generator"
)

KAGGLE_DIR = PROJECT_ROOT / "kaggle"


if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(KAGGLE_DIR),
    )


# ============================================================
# PROJECT CONFIG
# ============================================================

from video_config import (
    PROJECT_ROOT,
    GENERATE_SCRIPT,
    CLIPS_DIR,
    OUTPUT_DIR,
    CHARACTER_DIR,
    REFERENCE_DIR,
    WIDTH,
    HEIGHT,
    SOURCE_FPS,
    FINAL_FPS,
    DEFAULT_SHOT_SECONDS,
    FRAMES_PER_SHOT,
    USE_CPU_OFFLOAD,
    BASE_SEED,
    OUTPUT_CRF,
    OUTPUT_PRESET,
    PIXEL_FORMAT,
    IMAGE_EXTENSIONS,
    CLEAN_TEMPORARY_CLIPS,
)


# ============================================================
# QWEN3 STORY PLANNER CONFIGURATION
# ============================================================

# Primary model.
#
# This is the recommended model for our current pipeline:
#
# Qwen3 4B Instruct 2507
#
# It is a text-generation model used only for:
#
# story
#   ↓
# cinematic storyboard
#
QWEN_MODEL_ID = (
    "Qwen/Qwen3-4B-Instruct-2507"
)


# ------------------------------------------------------------
# Kaggle Dataset location
# ------------------------------------------------------------
#
# Your Dataset should eventually contain:
#
# /kaggle/input/datasets/shihoos/
#     ai-video-model/
#         qwen3-4b-instruct-2507/
#
# ------------------------------------------------------------

QWEN_LOCAL_MODEL = (
    Path("/kaggle/input/datasets/shihoos")
    / "ai-video-model"
    / "qwen3-4b-instruct-2507"
)


# ------------------------------------------------------------
# Model selection
# ------------------------------------------------------------

# True:
#   Use the local Kaggle Dataset whenever available.
#
# False:
#   Use the Hugging Face model ID.
#
USE_LOCAL_QWEN = True


# ------------------------------------------------------------
# Hugging Face fallback
# ------------------------------------------------------------

QWEN_ALLOW_HF_FALLBACK = True


# ------------------------------------------------------------
# Planner generation
# ------------------------------------------------------------

PLANNER_MAX_NEW_TOKENS = 2500

PLANNER_DO_SAMPLE = False

PLANNER_TEMPERATURE = 0.0


# ============================================================
# T4 SAFE RENDERING
# ============================================================
#
# We already tested this successfully on your T4:
#
# 1280x720
# 25 frames
# 8 FPS
#
# Keep this until the complete story pipeline is stable.
#
# Later we can implement temporal continuation for longer
# shots instead of pushing 40-50 frames into one T4 inference.
# ============================================================

SAFE_TEST_FRAMES = 25


# ============================================================
# DEFAULT SAMPLE STORY
# ============================================================

DEFAULT_STORY = """
A lone samurai named Kenji walks slowly through an ancient
Japanese forest at dawn. Mist hangs between the tall trees
and red maple leaves drift through the air.

Kenji suddenly hears a faint sound behind him. He stops walking
and slowly turns his head.

He sees a mysterious warrior standing far away in the fog.
Kenji carefully reaches for his katana and watches the stranger.

The mysterious warrior takes one slow step forward. Kenji draws
his katana and takes a defensive stance as the wind moves
through the forest.

The two warriors stare at each other in silence while the
morning sunlight breaks through the mist.
"""


# ============================================================
# COMMAND HELPER
# ============================================================

def run_command(command):
    """
    Run a command and stop if it fails.
    """

    print()
    print("=" * 70)
    print(
        "$ "
        + " ".join(
            str(item)
            for item in command
        )
    )
    print("=" * 70)

    result = subprocess.run(
        command
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"Command failed with exit code "
            f"{result.returncode}"
        )


# ============================================================
# SAFE DIRECTORY
# ============================================================

def ensure_real_directory(path):
    """
    Ensure that path exists as a real directory.

    If a file or symlink occupies the expected path,
    remove it and recreate the directory.

    This prevents FileExistsError problems such as:

        work/output
        work/story_clips

    being files instead of directories.
    """

    path = Path(path)

    # --------------------------------------------------------
    # Symlink
    # --------------------------------------------------------

    if path.is_symlink():

        print(
            f"⚠️ Removing invalid symlink: {path}"
        )

        path.unlink()

    # --------------------------------------------------------
    # File
    # --------------------------------------------------------

    elif path.exists() and not path.is_dir():

        print(
            f"⚠️ Removing invalid file path: {path}"
        )

        path.unlink()

    # --------------------------------------------------------
    # Already a directory
    # --------------------------------------------------------

    elif path.exists() and path.is_dir():

        return path

    # --------------------------------------------------------
    # Create directory
    # --------------------------------------------------------

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


# ============================================================
# STRING CLEANING
# ============================================================

def clean_name(value):

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return value.strip()


# ============================================================
# REFERENCE MANAGER
# ============================================================

def index_reference_files():
    """
    Find all character and reference images.

    Supported folders:

        assets/characters/
        assets/references/

    The function recursively scans both folders.
    """

    references = []

    for directory, kind in [
        (
            CHARACTER_DIR,
            "character",
        ),
        (
            REFERENCE_DIR,
            "reference",
        ),
    ]:

        if not directory.exists():
            continue

        for path in directory.rglob("*"):

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            references.append(
                {
                    "name": path.stem,
                    "kind": kind,
                    "path": path,
                }
            )

    return references


def match_references(
    names,
    references,
):
    """
    Match story character names against
    filenames in the reference folders.
    """

    matches = []

    normalized_names = [
        clean_name(name)
        for name in names
    ]

    for reference in references:

        ref_name = clean_name(
            reference["name"]
        )

        for requested in normalized_names:

            if not requested:
                continue

            if (
                requested == ref_name
                or requested in ref_name
                or ref_name in requested
            ):

                matches.append(
                    reference
                )

                break

    return matches


# ============================================================
# QWEN MODEL RESOLUTION
# ============================================================

def resolve_planner_model():
    """
    Select the story planner model.

    Priority:

        1. Local Kaggle Dataset
        2. Hugging Face fallback

    This means the model will NOT be downloaded every
    Kaggle session once it exists in the Dataset.
    """

    print()
    print("=" * 70)
    print("STORY PLANNER MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Local Dataset
    # --------------------------------------------------------

    if (
        USE_LOCAL_QWEN
        and QWEN_LOCAL_MODEL.is_dir()
    ):

        print(
            "✅ Local Qwen3 model found"
        )

        print(
            f"Path: {QWEN_LOCAL_MODEL}"
        )

        return str(
            QWEN_LOCAL_MODEL
        )

    # --------------------------------------------------------
    # Local model missing
    # --------------------------------------------------------

    print(
        "⚠️ Local Qwen3 model not found:"
    )

    print(
        QWEN_LOCAL_MODEL
    )

    # --------------------------------------------------------
    # Hugging Face fallback
    # --------------------------------------------------------

    if QWEN_ALLOW_HF_FALLBACK:

        print(
            "Using Hugging Face fallback:"
        )

        print(
            QWEN_MODEL_ID
        )

        return QWEN_MODEL_ID

    raise FileNotFoundError(
        "Qwen story planner is unavailable.\n\n"
        f"Local model expected at:\n"
        f"{QWEN_LOCAL_MODEL}\n\n"
        f"Hugging Face fallback:\n"
        f"{QWEN_MODEL_ID}"
    )


PLANNER_MODEL = (
    resolve_planner_model()
)


# ============================================================
# LOAD STORY PLANNER
# ============================================================

def load_story_planner():

    print()
    print("=" * 70)
    print("LOADING STORY PLANNER")
    print("=" * 70)

    print(
        f"Model: {PLANNER_MODEL}"
    )

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )

    local_model = Path(
        PLANNER_MODEL
    ).is_dir()

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            PLANNER_MODEL,
            local_files_only=local_model,
        )
    )

    # --------------------------------------------------------
    # Model
    #
    # Keep Qwen on CPU.
    #
    # LTX needs the T4 GPU.
    # --------------------------------------------------------

    model = (
        AutoModelForCausalLM.from_pretrained(
            PLANNER_MODEL,
            torch_dtype="auto",
            device_map="cpu",
            local_files_only=local_model,
        )
    )

    model.eval()

    print(
        "✅ Story planner loaded on CPU"
    )

    return (
        tokenizer,
        model,
    )


# ============================================================
# UNLOAD STORY PLANNER
# ============================================================

def unload_story_planner(
    model,
    tokenizer,
):

    del model
    del tokenizer

    gc.collect()

    try:

        import torch

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    except Exception:

        pass

    print(
        "✅ Story planner unloaded"
    )


# ============================================================
# STORY ANALYSIS PROMPT
# ============================================================

def build_planner_prompt(
    story,
    reference_names,
):

    available_refs = ", ".join(
        reference_names
    )

    if not available_refs:

        available_refs = "none"

    return f"""
You are a professional film director,
cinematographer and storyboard planner.

Your job is to convert the user's story into
a coherent sequence of cinematic shots that
can be generated independently by an AI video
generation model and then assembled into one
continuous film.

The user's story is authoritative.

Do not change the story.

Do not invent major events.

Do not remove important events.

==================================================
CHARACTER CONTINUITY
==================================================

Identify every important character.

For each important character describe:

- name
- approximate age
- gender when stated or visually implied
- face
- hairstyle
- clothing
- body type
- important props
- distinctive appearance

Keep character appearance consistent across
all shots.

If a reference image exists, associate the
correct reference filename with that character.

Available reference filenames:

{available_refs}

If no reference exists, create a detailed
visual description from the story.

==================================================
LOCATION CONTINUITY
==================================================

Identify important locations.

Maintain:

- environment
- architecture
- weather
- time of day
- lighting
- atmosphere
- visual style

Do not randomly change locations.

==================================================
STORY BREAKDOWN
==================================================

Break the story into logical cinematic shots.

Each shot should represent ONE clear visual
moment or action.

Do not create unnecessary shots.

Do not split every sentence automatically.

Combine closely related actions when they
belong in the same cinematic moment.

Create a new shot when:

- the action changes significantly
- the camera perspective should change
- a new character enters
- the emotional beat changes
- the location changes
- an important reaction deserves a close-up

==================================================
CAMERA SHOT
==================================================

Choose the camera shot based on the story.

Possible choices include:

- extreme wide shot
- wide establishing shot
- full shot
- medium shot
- medium close-up
- close-up
- extreme close-up
- over-the-shoulder
- POV
- two-shot

Do not use camera changes only for variety.

Camera choice must serve the scene.

==================================================
CAMERA ANGLE
==================================================

Choose an appropriate angle when useful.

Possible choices:

- eye level
- low angle
- high angle
- over-the-shoulder
- POV
- profile
- rear angle
- three-quarter angle

==================================================
CAMERA MOVEMENT
==================================================

Use movement only when useful.

Possible choices:

- static
- slow push-in
- dolly
- tracking
- pan
- tilt
- crane
- orbit
- follow shot
- handheld

Avoid excessive camera movement.

==================================================
SHOT DURATION
==================================================

Use approximately 5 seconds as the normal
default duration.

Use:

3-4 seconds for a very simple visual beat.

5 seconds for a normal cinematic action.

6-8 seconds when an action genuinely needs
more time.

Do not create extra shots merely to increase
video duration.

==================================================
VISUAL PROMPTS
==================================================

Every shot must contain a complete visual
prompt suitable for an AI video generator.

The prompt must describe:

- character
- character appearance
- clothing
- props
- location
- action
- environment
- lighting
- camera shot
- camera angle
- camera movement
- mood
- realistic physical movement

Do not say:

"same character as before"

Instead repeat the important character
appearance in the visual prompt.

==================================================
VISUAL QUALITY
==================================================

Prefer:

- realistic cinematic photography
- natural anatomy
- stable facial structure
- consistent character identity
- realistic lighting
- realistic materials
- physically believable movement
- subtle environmental motion
- cinematic composition

Avoid:

- distorted anatomy
- duplicate people
- unnecessary characters
- random costume changes
- random location changes
- excessive motion
- unrealistic camera movement

==================================================
OUTPUT
==================================================

Return ONLY valid JSON.

Do not use Markdown.

Do not use ```json fences.

Use exactly this structure:

{{
  "title": "short cinematic title",

  "visual_style": "overall visual style",

  "characters": [
    {{
      "name": "character name",
      "description": "complete visual description",
      "reference": "matching reference filename or null"
    }}
  ],

  "locations": [
    {{
      "name": "location name",
      "description": "complete visual description"
    }}
  ],

  "shots": [
    {{
      "shot_number": 1,
      "duration_seconds": 5,
      "characters": ["character names"],
      "location": "location name",
      "action": "clear visual action",
      "camera_shot": "camera shot",
      "camera_angle": "camera angle",
      "camera_movement": "camera movement",
      "mood": "mood",
      "visual_prompt": "complete cinematic video-generation prompt"
    }}
  ]
}}

STORY:

{story}
"""


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):

    text = text.strip()

    # --------------------------------------------------------
    # Remove markdown fences
    # --------------------------------------------------------

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^```\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:

        return json.loads(
            text
        )

    except json.JSONDecodeError:

        pass

    # --------------------------------------------------------
    # Find JSON object
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if (
        start >= 0
        and end > start
    ):

        candidate = text[
            start:end + 1
        ]

        return json.loads(
            candidate
        )

    raise ValueError(
        "Story planner did not return valid JSON."
    )


# ============================================================
# PLAN STORY
# ============================================================

def plan_story(
    story,
    references,
):

    tokenizer, model = (
        load_story_planner()
    )

    reference_names = [
        item["name"]
        for item in references
    ]

    prompt = build_planner_prompt(
        story,
        reference_names,
    )

    # --------------------------------------------------------
    # Qwen3 chat template
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional film "
                "director and storyboard planner. "
                "Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    import torch

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=(
                PLANNER_MAX_NEW_TOKENS
            ),
            do_sample=(
                PLANNER_DO_SAMPLE
            ),
        )

    generated = output[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        plan = extract_json(
            response
        )

    except Exception:

        print()
        print(
            "❌ Qwen returned invalid JSON."
        )

        print()
        print(
            "Raw planner output:"
        )

        print(
            response
        )

        unload_story_planner(
            model,
            tokenizer,
        )

        raise

    unload_story_planner(
        model,
        tokenizer,
    )

    return plan


# ============================================================
# VALIDATE PLAN
# ============================================================

def validate_plan(plan):

    if not isinstance(
        plan,
        dict,
    ):

        raise ValueError(
            "Story plan is not a JSON object."
        )

    shots = plan.get(
        "shots"
    )

    if not isinstance(
        shots,
        list,
    ):

        raise ValueError(
            "Story plan contains no shots."
        )

    if not shots:

        raise ValueError(
            "Story planner produced zero shots."
        )

    # --------------------------------------------------------
    # Validate every shot
    # --------------------------------------------------------

    for index, shot in enumerate(
        shots,
        start=1,
    ):

        if not isinstance(
            shot,
            dict,
        ):

            raise ValueError(
                f"Shot {index} is invalid."
            )

        shot[
            "shot_number"
        ] = index

        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        duration = shot.get(
            "duration_seconds",
            DEFAULT_SHOT_SECONDS,
        )

        try:

            duration = float(
                duration
            )

        except Exception:

            duration = (
                DEFAULT_SHOT_SECONDS
            )

        # Allow planner to choose 3-8 seconds.
        duration = max(
            3.0,
            min(
                8.0,
                duration,
            ),
        )

        shot[
            "duration_seconds"
        ] = duration

        # ----------------------------------------------------
        # Required fields
        # ----------------------------------------------------

        shot.setdefault(
            "characters",
            [],
        )

        shot.setdefault(
            "location",
            "",
        )

        shot.setdefault(
            "action",
            "",
        )

        shot.setdefault(
            "camera_shot",
            "cinematic medium shot",
        )

        shot.setdefault(
            "camera_angle",
            "eye level",
        )

        shot.setdefault(
            "camera_movement",
            "slow controlled camera movement",
        )

        shot.setdefault(
            "mood",
            "cinematic",
        )

        shot.setdefault(
            "visual_prompt",
            shot["action"],
        )

        # ----------------------------------------------------
        # Normalize character list
        # ----------------------------------------------------

        if not isinstance(
            shot["characters"],
            list,
        ):

            shot["characters"] = []

    return plan


# ============================================================
# PRINT PLAN
# ============================================================

def print_plan(plan):

    print()
    print("=" * 70)
    print("🎬 STORY PLAN")
    print("=" * 70)

    print(
        "Title:",
        plan.get(
            "title",
            "Untitled",
        ),
    )

    characters = plan.get(
        "characters",
        [],
    )

    print(
        "Characters:",
        ", ".join(
            str(
                item.get(
                    "name",
                    "",
                )
            )
            for item in characters
            if isinstance(
                item,
                dict,
            )
        ),
    )

    shots = plan[
        "shots"
    ]

    print(
        f"Shots: {len(shots)}"
    )

    print()

    for shot in shots:

        print(
            f"SHOT {shot['shot_number']}"
        )

        print(
            f"Duration : "
            f"{shot['duration_seconds']} sec"
        )

        print(
            f"Location : "
            f"{shot['location']}"
        )

        print(
            f"Action   : "
            f"{shot['action']}"
        )

        print(
            f"Camera   : "
            f"{shot['camera_shot']}"
        )

        print(
            f"Angle    : "
            f"{shot['camera_angle']}"
        )

        print(
            f"Movement : "
            f"{shot['camera_movement']}"
        )

        print()


# ============================================================
# GENERATE SHOT
# ============================================================

def generate_shot(
    shot,
    plan,
    shot_index,
    references,
):

    shot_dir = (
        CLIPS_DIR
        / f"shot_{shot_index:03d}"
    )

    ensure_real_directory(
        shot_dir
    )

    # --------------------------------------------------------
    # Match character references
    # --------------------------------------------------------

    characters = shot.get(
        "characters",
        [],
    )

    matched = match_references(
        characters,
        references,
    )

    # --------------------------------------------------------
    # Build final LTX prompt
    # --------------------------------------------------------

    prompt = str(
        shot.get(
            "visual_prompt",
            "",
        )
    ).strip()

    prompt += (
        ". Maintain strict visual continuity "
        "with the established characters, "
        "clothing, environment and lighting. "
        f"Camera shot: "
        f"{shot['camera_shot']}. "
        f"Camera angle: "
        f"{shot['camera_angle']}. "
        f"Camera movement: "
        f"{shot['camera_movement']}. "
        f"Mood: "
        f"{shot['mood']}. "
        "Realistic cinematic lighting, "
        "natural physical motion, "
        "detailed realistic textures, "
        "physically believable movement, "
        "stable facial identity, "
        "stable anatomy, "
        "no blood, no gore."
    )

    # --------------------------------------------------------
    # Shot information
    # --------------------------------------------------------

    print()
    print("#" * 70)

    print(
        f"🎥 GENERATING SHOT "
        f"{shot_index}/{len(plan['shots'])}"
    )

    print(
        "#" * 70
    )

    print(
        f"Duration planned: "
        f"{shot['duration_seconds']} sec"
    )

    print(
        f"Safe T4 frames: "
        f"{SAFE_TEST_FRAMES}"
    )

    print(
        f"Camera: "
        f"{shot['camera_shot']}"
    )

    print(
        f"Angle: "
        f"{shot['camera_angle']}"
    )

    print(
        f"Movement: "
        f"{shot['camera_movement']}"
    )

    # --------------------------------------------------------
    # References
    # --------------------------------------------------------

    if matched:

        print(
            "References:"
        )

        for item in matched:

            print(
                f"  {item['kind']}: "
                f"{item['path']}"
            )

    else:

        print(
            "References: none"
        )

    # --------------------------------------------------------
    # LTX command
    # --------------------------------------------------------

    command = [
        sys.executable,

        str(
            GENERATE_SCRIPT
        ),

        "--prompt",
        prompt,

        "--width",
        str(WIDTH),

        "--height",
        str(HEIGHT),

        "--frames",
        str(
            SAFE_TEST_FRAMES
        ),

        "--fps",
        str(SOURCE_FPS),

        "--seed",
        str(
            BASE_SEED
            + shot_index
        ),

        "--output",
        str(shot_dir),
    ]

    # --------------------------------------------------------
    # CPU offload
    # --------------------------------------------------------

    if USE_CPU_OFFLOAD:

        command.append(
            "--offload"
        )

    # --------------------------------------------------------
    # Image conditioning
    # --------------------------------------------------------

    if matched:

        media_paths = [
            str(
                item["path"]
            )
            for item in matched
        ]

        start_frames = [
            "0"
            for _ in matched
        ]

        strengths = [
            "0.85"
            for _ in matched
        ]

        command.extend(
            [
                "--conditioning-media",
                *media_paths,

                "--conditioning-start-frames",
                *start_frames,

                "--conditioning-strengths",
                *strengths,
            ]
        )

    # --------------------------------------------------------
    # Run LTX
    # --------------------------------------------------------

    run_command(
        command
    )

    # --------------------------------------------------------
    # Find generated MP4
    # --------------------------------------------------------

    videos = list(
        shot_dir.glob(
            "*.mp4"
        )
    )

    if not videos:

        raise RuntimeError(
            f"No video generated for "
            f"shot {shot_index}"
        )

    videos.sort(
        key=lambda path:
        path.stat().st_mtime,
        reverse=True,
    )

    video = videos[0]

    print(
        f"✅ Shot generated: "
        f"{video}"
    )

    return video


# ============================================================
# CONCAT FILE
# ============================================================

def create_concat_file(
    videos,
):

    concat = (
        CLIPS_DIR
        / "concat.txt"
    )

    with concat.open(
        "w",
        encoding="utf-8",
    ) as file:

        for video in videos:

            escaped = (
                str(
                    video.resolve()
                )
                .replace(
                    "'",
                    "'\\''",
                )
            )

            file.write(
                f"file '{escaped}'\n"
            )

    return concat


# ============================================================
# FINAL ASSEMBLY
# ============================================================

def assemble(
    videos,
):

    ensure_real_directory(
        OUTPUT_DIR
    )

    concat = create_concat_file(
        videos
    )

    final_video = (
        OUTPUT_DIR
        / "story_final.mp4"
    )

    # --------------------------------------------------------
    # FFmpeg
    #
    # Individual LTX clips are generated at SOURCE_FPS.
    #
    # Final output is always converted to FINAL_FPS.
    #
    # Current target:
    #
    # 1280x720
    # 25 FPS
    # H.264
    # --------------------------------------------------------

    command = [
        "ffmpeg",

        "-y",

        "-f",
        "concat",

        "-safe",
        "0",

        "-i",
        str(concat),

        "-vf",
        (
            f"scale={WIDTH}:{HEIGHT}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={WIDTH}:{HEIGHT}:"
            "(ow-iw)/2:"
            "(oh-ih)/2,"
            f"fps={FINAL_FPS}"
        ),

        "-c:v",
        "libx264",

        "-preset",
        OUTPUT_PRESET,

        "-crf",
        str(
            OUTPUT_CRF
        ),

        "-pix_fmt",
        PIXEL_FORMAT,

        "-movflags",
        "+faststart",

        # Audio intentionally disabled for now.
        "-an",

        str(final_video),
    ]

    run_command(
        command
    )

    if not final_video.exists():

        raise RuntimeError(
            "Final video was not created."
        )

    size_mb = (
        final_video.stat().st_size
        / (1024 * 1024)
    )

    print()
    print("=" * 70)
    print("✅ FINAL STORY VIDEO")
    print("=" * 70)

    print(
        f"File       : "
        f"{final_video}"
    )

    print(
        f"Resolution : "
        f"{WIDTH}x{HEIGHT}"
    )

    print(
        f"FPS        : "
        f"{FINAL_FPS}"
    )

    print(
        f"Shots      : "
        f"{len(videos)}"
    )

    print(
        f"Size       : "
        f"{size_mb:.2f} MB"
    )

    return final_video


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate a multi-shot "
            "cinematic video from a story."
        )
    )

    parser.add_argument(
        "--story",
        type=str,
        default=None,
        help="Story text.",
    )

    parser.add_argument(
        "--story-file",
        type=str,
        default=None,
        help="Path to a text story file.",
    )

    parser.add_argument(
        "--keep-clips",
        action="store_true",
        help=(
            "Keep previous story clips."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # DIRECTORIES
    # ========================================================

    ensure_real_directory(
        CHARACTER_DIR
    )

    ensure_real_directory(
        REFERENCE_DIR
    )

    ensure_real_directory(
        OUTPUT_DIR
    )

    # ========================================================
    # TEMPORARY CLIPS
    # ========================================================

    if (
        CLEAN_TEMPORARY_CLIPS
        and not args.keep_clips
    ):

        if (
            CLIPS_DIR.exists()
            or CLIPS_DIR.is_symlink()
        ):

            print(
                f"🧹 Cleaning temporary clips: "
                f"{CLIPS_DIR}"
            )

            if CLIPS_DIR.is_symlink():

                CLIPS_DIR.unlink()

            elif CLIPS_DIR.is_dir():

                shutil.rmtree(
                    CLIPS_DIR
                )

            else:

                CLIPS_DIR.unlink()

    ensure_real_directory(
        CLIPS_DIR
    )

    # ========================================================
    # STORY
    # ========================================================

    if args.story_file:

        story_path = Path(
            args.story_file
        )

        if not story_path.exists():

            raise FileNotFoundError(
                story_path
            )

        story = (
            story_path.read_text(
                encoding="utf-8"
            )
        )

    elif args.story:

        story = args.story

    else:

        story = DEFAULT_STORY

    story = story.strip()

    if not story:

        raise ValueError(
            "Story is empty."
        )

    # ========================================================
    # REFERENCES
    # ========================================================

    references = (
        index_reference_files()
    )

    print()
    print("=" * 70)
    print("📚 REFERENCES")
    print("=" * 70)

    if references:

        for item in references:

            print(
                f"{item['kind']:10} "
                f"{item['name']:25} "
                f"{item['path']}"
            )

    else:

        print(
            "No character/reference images found."
        )

    # ========================================================
    # STORY PLANNING
    # ========================================================

    print()
    print("=" * 70)
    print("🧠 ANALYZING STORY")
    print("=" * 70)

    plan = plan_story(
        story,
        references,
    )

    plan = validate_plan(
        plan
    )

    print_plan(
        plan
    )

    # ========================================================
    # GENERATE SHOTS
    # ========================================================

    generated = []

    total_shots = len(
        plan["shots"]
    )

    print()
    print(
        f"🎬 Total shots planned: "
        f"{total_shots}"
    )

    for index, shot in enumerate(
        plan["shots"],
        start=1,
    ):

        video = generate_shot(
            shot,
            plan,
            index,
            references,
        )

        generated.append(
            video
        )

        # ----------------------------------------------------
        # Clean memory between shots.
        # ----------------------------------------------------

        gc.collect()

        try:

            import torch

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        except Exception:

            pass

    # ========================================================
    # ASSEMBLE
    # ========================================================

    final_video = assemble(
        generated
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("🚀 GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Final video:\n"
        f"{final_video}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
