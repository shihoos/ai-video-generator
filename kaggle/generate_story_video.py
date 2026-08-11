import argparse
import gc
import hashlib
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
# Story
#   ↓
# Qwen3 story / storyboard planner
#   ↓
# Characters / locations / actions / camera / duration
#   ↓
# Optional reference matching
#   ↓
# LTX shot generation
#   ↓
# Unlimited shot collection
#   ↓
# FFmpeg final assembly
#   ↓
# 25 FPS MP4
#
# Audio is intentionally not generated in this version.
# ============================================================


PROJECT_ROOT = Path("/kaggle/working/ai-video-generator")
KAGGLE_DIR = PROJECT_ROOT / "kaggle"

if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))


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
    QWEN_MODEL_ID,
    QWEN_LOCAL_MODEL,
    USE_LOCAL_QWEN,
    QWEN_ALLOW_HF_FALLBACK,
    QWEN_MAX_NEW_TOKENS,
    QWEN_DO_SAMPLE,
    QWEN_TEMPERATURE,
    QWEN_TOP_P,
    QWEN_TOP_K,
    REFERENCE_STRENGTH,
    REFERENCE_START_FRAME,
    FINAL_VIDEO_NAME,
)


# ============================================================
# GENERATION FINGERPRINT / RESUME
# ============================================================
#
# A run is only treated as "the same generation" if the story
# text AND the technical settings that affect shot output are
# identical to the previous run. This prevents silently
# reusing shots that were rendered under a different
# resolution, frame budget or model.
#
# If the fingerprint changes, previous clips are cleared.
# If it matches, completed shots are reused.
# ============================================================

FINGERPRINT_FILE = (
    CLIPS_DIR / ".generation_fingerprint.json"
)

STORY_PLAN_FILE = (
    CLIPS_DIR / "story_plan.json"
)

# Bump this manually if you change the planner prompt text
# (build_planner_prompt) in a way that should invalidate old
# shots even though the story itself is unchanged.
PLANNER_PROMPT_VERSION = 1


def compute_fingerprint(story):
    """Build a fingerprint covering the story and every
    technical setting that affects how a shot is rendered."""

    payload = {
        "story": story.strip(),
        "planner_prompt_version": PLANNER_PROMPT_VERSION,
        "width": WIDTH,
        "height": HEIGHT,
        "source_fps": SOURCE_FPS,
        "final_fps": FINAL_FPS,
        "frames_per_shot": FRAMES_PER_SHOT,
        "qwen_model_id": QWEN_MODEL_ID,
        "qwen_do_sample": QWEN_DO_SAMPLE,
        "base_seed": BASE_SEED,
        "reference_strength": REFERENCE_STRENGTH,
        "reference_start_frame": REFERENCE_START_FRAME,
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def prepare_story_workspace(story, force_clean=False):
    """
    Decide whether existing shot clips (and the saved storyboard)
    can be reused.

    Returns True if this is a resumed run (same fingerprint,
    safe to skip re-planning and reuse existing shots), False
    if the workspace was cleaned and generation starts fresh.
    """

    ensure_real_directory(CLIPS_DIR)

    current_fingerprint = compute_fingerprint(story)

    if force_clean:
        print()
        print("🧹 --force-clean requested. Wiping previous clips.")
        clean_directory(CLIPS_DIR)
        ensure_real_directory(CLIPS_DIR)
        FINGERPRINT_FILE.write_text(
            current_fingerprint,
            encoding="utf-8",
        )
        return False

    if not FINGERPRINT_FILE.exists():
        FINGERPRINT_FILE.write_text(
            current_fingerprint,
            encoding="utf-8",
        )
        return False

    previous_fingerprint = (
        FINGERPRINT_FILE.read_text(encoding="utf-8").strip()
    )

    if previous_fingerprint == current_fingerprint:
        print()
        print(
            "♻️ Matching generation fingerprint detected."
        )
        print(
            "Existing completed shots will be reused."
        )
        return True

    print()
    print(
        "🆕 Story or technical settings changed since last run."
    )
    print(
        "Cleaning previous story clips..."
    )

    clean_directory(CLIPS_DIR)
    ensure_real_directory(CLIPS_DIR)

    FINGERPRINT_FILE.write_text(
        current_fingerprint,
        encoding="utf-8",
    )

    return False


def load_saved_plan():
    """Load a previously saved storyboard, if one exists."""

    if not STORY_PLAN_FILE.exists():
        return None

    try:
        return json.loads(
            STORY_PLAN_FILE.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        print(
            "⚠️ Saved story plan could not be parsed. "
            "Re-planning with Qwen."
        )
        return None


def save_story_plan(plan):
    """Save the generated storyboard for inspection/debugging."""

    ensure_real_directory(CLIPS_DIR)

    STORY_PLAN_FILE.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"✅ Story plan saved: {STORY_PLAN_FILE}")


# ============================================================
# BASIC HELPERS
# ============================================================

def run_command(command):
    """Run a subprocess and fail loudly if it fails."""

    print()
    print("=" * 70)
    print("$ " + " ".join(str(item) for item in command))
    print("=" * 70)

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}"
        )


def ensure_real_directory(path):
    """
    Make sure path is a real directory.

    This specifically prevents the recurring Kaggle error where
    work/output or work/story_clips exists as a FILE.
    """

    path = Path(path)

    if path.is_symlink():
        path.unlink()

    elif path.exists() and not path.is_dir():
        path.unlink()

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not path.is_dir():
        raise RuntimeError(
            f"Expected directory but could not create: {path}"
        )

    return path


def clean_directory(path):
    """Remove a directory or file and recreate it as a directory."""

    path = Path(path)

    if path.is_symlink():
        path.unlink()

    elif path.exists() and path.is_dir():
        shutil.rmtree(path)

    elif path.exists():
        path.unlink()

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def check_requirements():
    """Check the files and tools required by this pipeline."""

    print("=" * 70)
    print("CHECKING STORY VIDEO PIPELINE")
    print("=" * 70)

    if not GENERATE_SCRIPT.exists():
        raise FileNotFoundError(
            f"LTX generator not found: {GENERATE_SCRIPT}"
        )

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg is not installed or not available in PATH."
        )

    print(f"✅ generate.py : {GENERATE_SCRIPT}")
    print("✅ FFmpeg available")

    ensure_real_directory(PROJECT_ROOT)
    ensure_real_directory(WORK_DIR := PROJECT_ROOT / "work")
    ensure_real_directory(CLIPS_DIR)
    ensure_real_directory(OUTPUT_DIR)
    ensure_real_directory(CHARACTER_DIR)
    ensure_real_directory(REFERENCE_DIR)

    print("✅ Working directories ready")


# ============================================================
# QWEN MODEL RESOLUTION
# ============================================================

def resolve_qwen_model():
    """
    Select the Qwen model.

    Priority:
        1. Local Kaggle Dataset
        2. Hugging Face fallback
        3. Error
    """

    local_exists = (
        QWEN_LOCAL_MODEL.exists()
        and QWEN_LOCAL_MODEL.is_dir()
    )

    if USE_LOCAL_QWEN and local_exists:
        print()
        print("✅ Local Qwen Dataset found")
        print(f"Qwen path: {QWEN_LOCAL_MODEL}")
        return str(QWEN_LOCAL_MODEL)

    if USE_LOCAL_QWEN and not local_exists:
        print()
        print("⚠️ Local Qwen Dataset not found")
        print(f"Expected: {QWEN_LOCAL_MODEL}")

    if QWEN_ALLOW_HF_FALLBACK:
        print()
        print("Using Hugging Face Qwen fallback:")
        print(QWEN_MODEL_ID)
        return QWEN_MODEL_ID

    raise FileNotFoundError(
        "Qwen local model was not found and HF fallback "
        "is disabled."
    )


# ============================================================
# REFERENCE INDEX
# ============================================================

def clean_name(value):
    """Normalize a character/reference name for matching."""

    value = str(value).lower().strip()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return " ".join(value.split())


def index_reference_files():
    """
    Index optional character and general reference images.

    Example:

        assets/characters/kenji.png

    If Qwen mentions "Kenji", the file can automatically be
    attached to that shot.
    """

    references = []

    for directory, kind in [
        (CHARACTER_DIR, "character"),
        (REFERENCE_DIR, "reference"),
    ]:
        if not directory.exists():
            continue

        for path in directory.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            references.append(
                {
                    "name": path.stem,
                    "kind": kind,
                    "path": path,
                }
            )

    return references


def match_references(names, references):
    """
    Match planner character/reference names to local files.

    Priority:
        1. Exact normalized match.
        2. Fuzzy (substring) match, but only when exactly one
           candidate exists. A warning is printed.
        3. If a fuzzy match is ambiguous (multiple candidates),
           attach nothing rather than guessing.
    """

    matches = []
    seen = set()

    normalized_names = [
        clean_name(name)
        for name in names
        if str(name).strip()
    ]

    # --------------------------------------------------------
    # Pass 1: exact matches
    # --------------------------------------------------------

    for requested in normalized_names:
        if not requested:
            continue

        for reference in references:
            ref_name = clean_name(reference["name"])

            if ref_name and ref_name == requested:
                key = str(reference["path"])

                if key not in seen:
                    matches.append(reference)
                    seen.add(key)

                break

    matched_names = {
        clean_name(item["name"]) for item in matches
    }

    # --------------------------------------------------------
    # Pass 2: fuzzy fallback only for names with no exact match
    # --------------------------------------------------------

    for requested in normalized_names:
        if not requested or requested in matched_names:
            continue

        candidates = [
            reference
            for reference in references
            if clean_name(reference["name"])
            and (
                requested in clean_name(reference["name"])
                or clean_name(reference["name"]) in requested
            )
        ]

        if len(candidates) == 1:
            candidate = candidates[0]
            key = str(candidate["path"])

            if key not in seen:
                print(
                    "⚠️ Fuzzy reference match: requested "
                    f"'{requested}' matched "
                    f"'{candidate['name']}'"
                )

                matches.append(candidate)
                seen.add(key)

        elif len(candidates) > 1:
            print(
                f"⚠️ Ambiguous reference '{requested}' matched "
                f"{len(candidates)} files. No reference attached."
            )

    return matches


# ============================================================
# STORY PLANNER PROMPT
# ============================================================

def build_planner_prompt(story, reference_names):
    """
    Build a strict cinematic storyboard-planning prompt.

    Qwen is asked to decide:
      - characters
      - locations
      - actions
      - shot count
      - shot duration
      - camera shot
      - camera movement
      - mood
      - visual prompt
    """

    available_refs = ", ".join(reference_names)

    if not available_refs:
        available_refs = "none"

    return f"""
You are the story director, cinematographer and storyboard
planner for a cinematic AI video generator.

Convert the user's story into a practical sequence of visual
shots that can be generated independently by a text-to-video
model.

USER STORY:
{story}

AVAILABLE LOCAL REFERENCE FILE NAMES:
{available_refs}

IMPORTANT RULES:

1. Preserve the user's story.
2. Do not invent major plot events.
3. Do not invent unnecessary characters.
4. Identify every important named character.
5. Keep character appearance consistent across all shots.
6. Keep clothing, props and environment consistent.
7. Identify locations and their visual characteristics.
8. Break the story into logical cinematic shots.
9. Each shot should contain one clear primary visual action.
10. Choose the camera framing automatically.
11. Choose camera movement only when it helps the action.
12. Use establishing shots when a new location is introduced.
13. Use wide shots for geography and group relationships.
14. Use medium shots for character actions.
15. Use close-ups for emotion, eyes, hands and important details.
16. Use over-the-shoulder shots for conversations or confrontations.
17. Use tracking/dolly/push/pan/tilt/crane movement only when
    visually appropriate.
18. Maintain continuity between neighboring shots.
19. Prefer approximately {DEFAULT_SHOT_SECONDS:.1f} seconds for
    normal shots, but choose a shorter or longer planned duration
    when the story action genuinely needs it.
20. Do not create unnecessary shots.
21. The video generator currently uses a T4-safe source frame
    budget, so the planned duration is a cinematic planning hint;
    the renderer controls the actual source frames.
22. If a local reference filename clearly matches a character,
    include that exact character name in the shot's characters
    list so the renderer can attach the reference.
23. If no reference exists, design the character visually from
    the story.
24. Do not mention technical implementation details in
    visual_prompt.
25. Return ONLY valid JSON.
26. Do not use Markdown fences.
27. Do not add commentary before or after the JSON.

Return exactly this structure:

{{
  "title": "short title",
  "visual_style": "overall cinematic visual style",
  "characters": [
    {{
      "name": "character name",
      "description": "stable visual appearance, clothing and props"
    }}
  ],
  "locations": [
    {{
      "name": "location name",
      "description": "stable visual environment"
    }}
  ],
  "shots": [
    {{
      "shot_number": 1,
      "duration_seconds": 5.0,
      "characters": ["character name"],
      "location": "location name",
      "action": "single clear visual action",
      "camera_shot": "wide|medium|close-up|extreme close-up|over-the-shoulder|low-angle|high-angle",
      "camera_movement": "static|slow push-in|slow pull-back|tracking|pan|tilt|dolly|crane",
      "mood": "visual emotional mood",
      "visual_prompt": "complete cinematic generation prompt"
    }}
  ]
}}

The number of shots is NOT fixed. Use as many shots as the
story actually needs.
"""


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):
    """
    Robustly extract the first valid JSON object from Qwen output.
    """

    text = text.strip()

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

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Qwen returned text containing JSON, but the JSON "
                f"could not be parsed: {exc}"
            ) from exc

    raise ValueError(
        "Qwen did not return a valid JSON object."
    )


# ============================================================
# QWEN LOAD / UNLOAD
# ============================================================

def load_story_planner():
    """
    Load Qwen on CPU so the T4 GPU remains available for LTX.
    """

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )

    planner_model = resolve_qwen_model()

    print()
    print("=" * 70)
    print("LOADING STORY PLANNER")
    print("=" * 70)
    print(f"Model: {planner_model}")

    tokenizer = AutoTokenizer.from_pretrained(
        planner_model,
        local_files_only=(
            planner_model == str(QWEN_LOCAL_MODEL)
        ),
    )

    model = AutoModelForCausalLM.from_pretrained(
        planner_model,
        torch_dtype="auto",
        device_map="cpu",
        local_files_only=(
            planner_model == str(QWEN_LOCAL_MODEL)
        ),
    )

    model.eval()

    print("✅ Story planner loaded on CPU")

    return tokenizer, model


def unload_story_planner(model, tokenizer):
    """Release Qwen before LTX starts."""

    print()
    print("UNLOADING STORY PLANNER")

    del model
    del tokenizer

    gc.collect()

    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    print("✅ Story planner unloaded")


# ============================================================
# STORY PLANNING
# ============================================================

def plan_story(story, references):
    """Generate the complete storyboard with Qwen."""

    tokenizer, model = load_story_planner()

    reference_names = [
        item["name"]
        for item in references
    ]

    prompt = build_planner_prompt(
        story,
        reference_names,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional film director and "
                "storyboard planner. Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    # Inputs are already CPU tensors and Qwen is deliberately
    # loaded on CPU so the T4 remains available for LTX.
    with __import__("torch").no_grad():
        generation_kwargs = {
            "max_new_tokens": QWEN_MAX_NEW_TOKENS,
            "do_sample": QWEN_DO_SAMPLE,
        }

        if QWEN_DO_SAMPLE:
            generation_kwargs.update(
                {
                    "temperature": QWEN_TEMPERATURE,
                    "top_p": QWEN_TOP_P,
                    "top_k": QWEN_TOP_K,
                }
            )

        output = model.generate(
            **inputs,
            **generation_kwargs,
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

    print()
    print("=" * 70)
    print("QWEN RAW PLANNER OUTPUT")
    print("=" * 70)
    print(response[:4000])

    plan = extract_json(response)

    unload_story_planner(
        model,
        tokenizer,
    )

    return plan


# ============================================================
# PLAN VALIDATION / NORMALIZATION
# ============================================================

VALID_CAMERA_SHOTS = {
    "wide",
    "medium",
    "close-up",
    "extreme close-up",
    "over-the-shoulder",
    "low-angle",
    "high-angle",
}

VALID_MOVEMENTS = {
    "static",
    "slow push-in",
    "slow pull-back",
    "tracking",
    "pan",
    "tilt",
    "dolly",
    "crane",
}


def normalize_duration(value):
    try:
        duration = float(value)
    except (TypeError, ValueError):
        duration = DEFAULT_SHOT_SECONDS

    return max(
        2.0,
        min(duration, 10.0),
    )


def normalize_shot(shot, index):
    """Make a planner shot safe for the renderer."""

    if not isinstance(shot, dict):
        raise ValueError(
            f"Shot {index} is not a JSON object."
        )

    characters = shot.get(
        "characters",
        [],
    )

    if not isinstance(characters, list):
        characters = [str(characters)]

    characters = [
        str(item).strip()
        for item in characters
        if str(item).strip()
    ]

    camera_shot = str(
        shot.get(
            "camera_shot",
            "medium",
        )
    ).strip().lower()

    if camera_shot not in VALID_CAMERA_SHOTS:
        camera_shot = "medium"

    camera_movement = str(
        shot.get(
            "camera_movement",
            "static",
        )
    ).strip().lower()

    if camera_movement not in VALID_MOVEMENTS:
        camera_movement = "static"

    location = str(
        shot.get(
            "location",
            "unspecified",
        )
    ).strip()

    action = str(
        shot.get(
            "action",
            "the characters remain naturally present",
        )
    ).strip()

    mood = str(
        shot.get(
            "mood",
            "cinematic",
        )
    ).strip()

    visual_prompt = str(
        shot.get(
            "visual_prompt",
            "",
        )
    ).strip()

    if not visual_prompt:
        visual_prompt = (
            f"Cinematic shot of {action} in {location}. "
            f"{mood} mood, realistic lighting, natural motion, "
            "detailed textures, physically believable movement."
        )

    return {
        "shot_number": index,
        "duration_seconds": normalize_duration(
            shot.get(
                "duration_seconds",
                DEFAULT_SHOT_SECONDS,
            )
        ),
        "characters": characters,
        "location": location,
        "action": action,
        "camera_shot": camera_shot,
        "camera_movement": camera_movement,
        "mood": mood,
        "visual_prompt": visual_prompt,
    }


def validate_and_normalize_plan(plan):
    """Validate required fields and normalize all shots."""

    if not isinstance(plan, dict):
        raise ValueError(
            "Story planner output must be a JSON object."
        )

    raw_shots = plan.get(
        "shots",
        [],
    )

    if not isinstance(raw_shots, list) or not raw_shots:
        raise ValueError(
            "Story planner returned no shots."
        )

    normalized_shots = []

    for index, shot in enumerate(
        raw_shots,
        start=1,
    ):
        normalized_shots.append(
            normalize_shot(
                shot,
                index,
            )
        )

    plan["shots"] = normalized_shots

    if not isinstance(
        plan.get("characters", []),
        list,
    ):
        plan["characters"] = []

    if not isinstance(
        plan.get("locations", []),
        list,
    ):
        plan["locations"] = []

    plan["title"] = str(
        plan.get(
            "title",
            "Untitled Story",
        )
    )

    plan["visual_style"] = str(
        plan.get(
            "visual_style",
            "cinematic realistic",
        )
    )

    return plan


# ============================================================
# PLAN PRINTING
# ============================================================

def print_plan(plan):
    print()
    print("=" * 70)
    print("🎬 STORY PLAN")
    print("=" * 70)

    print(
        f"Title      : {plan['title']}"
    )

    print(
        "Characters : "
        + ", ".join(
            item.get("name", "")
            for item in plan.get(
                "characters",
                [],
            )
            if isinstance(item, dict)
        )
    )

    print(
        f"Shots      : {len(plan['shots'])}"
    )

    print()

    for shot in plan["shots"]:
        print(
            f"SHOT {shot['shot_number']}"
        )
        print(
            f"Duration : "
            f"{shot['duration_seconds']:.1f} sec"
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
            f"Movement : "
            f"{shot['camera_movement']}"
        )
        print()


# ============================================================
# SHOT GENERATION
# ============================================================

def find_newest_video(directory):
    """Find the newest generated MP4."""

    videos = list(
        Path(directory).glob("*.mp4")
    )

    if not videos:
        raise RuntimeError(
            f"No MP4 video generated in {directory}"
        )

    videos.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    return videos[0]


def generate_shot(
    shot,
    plan,
    shot_index,
    references,
):
    """Generate one LTX shot."""

    shot_dir = (
        CLIPS_DIR
        / f"shot_{shot_index:03d}"
    )

    ensure_real_directory(shot_dir)

    # ----------------------------------------------------------
    # Resume: reuse this shot if it was already rendered under
    # the same generation fingerprint.
    # ----------------------------------------------------------

    existing_videos = list(shot_dir.glob("*.mp4"))

    if existing_videos:
        existing_videos.sort(
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        print()
        print(
            f"♻️ Shot {shot_index} already exists, "
            "reusing it (delete its folder or use "
            "--force-clean to regenerate)."
        )
        print(f"   {existing_videos[0]}")

        return existing_videos[0]

    characters = shot.get(
        "characters",
        [],
    )

    matched = match_references(
        characters,
        references,
    )

    # Add the global story style and continuity instructions.
    prompt = (
        f"{plan.get('visual_style', 'cinematic realistic')}. "
        f"{shot['visual_prompt']}. "
        "Maintain strict visual continuity with the established "
        "characters, clothing, props, environment and lighting. "
        f"Camera framing: {shot['camera_shot']}. "
        f"Camera movement: {shot['camera_movement']}. "
        f"Mood: {shot['mood']}. "
        "Realistic cinematic lighting, natural subtle movement, "
        "detailed realistic textures, physically believable motion, "
        "stable anatomy, stable face, no text, no subtitles, "
        "no blood, no gore."
    )

    print()
    print("#" * 70)
    print(
        f"🎥 GENERATING SHOT "
        f"{shot_index}/{len(plan['shots'])}"
    )
    print("#" * 70)

    print(
        f"Planned duration : "
        f"{shot['duration_seconds']:.1f} sec"
    )

    print(
        f"Source frames    : "
        f"{FRAMES_PER_SHOT}"
    )

    print(
        f"Source FPS       : "
        f"{SOURCE_FPS}"
    )

    print(
        f"Camera           : "
        f"{shot['camera_shot']}"
    )

    if matched:
        print("References:")

        for item in matched:
            print(
                f"  {item['kind']}: "
                f"{item['path']}"
            )
    else:
        print("References       : none")

    command = [
        sys.executable,
        str(GENERATE_SCRIPT),
        "--prompt",
        prompt,
        "--width",
        str(WIDTH),
        "--height",
        str(HEIGHT),
        "--frames",
        str(FRAMES_PER_SHOT),
        "--fps",
        str(SOURCE_FPS),
        "--seed",
        str(BASE_SEED + shot_index),
        "--output",
        str(shot_dir),
    ]

    if USE_CPU_OFFLOAD:
        command.append("--offload")

    # Optional image conditioning.
    if matched:
        media_paths = [
            str(item["path"])
            for item in matched
        ]

        start_frames = [
            str(REFERENCE_START_FRAME)
            for _ in matched
        ]

        strengths = [
            str(REFERENCE_STRENGTH)
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

    run_command(command)

    video = find_newest_video(
        shot_dir
    )

    print(
        f"✅ Shot generated: {video}"
    )

    return video


# ============================================================
# CONCAT FILE
# ============================================================

def create_concat_file(videos):
    """Create an FFmpeg concat file."""

    concat_file = (
        CLIPS_DIR / "concat.txt"
    )

    with concat_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        for video in videos:
            escaped = (
                str(video.resolve())
                .replace(
                    "'",
                    "'\\''",
                )
            )

            file.write(
                f"file '{escaped}'\n"
            )

    return concat_file


# ============================================================
# FINAL ASSEMBLY
# ============================================================

def assemble(videos):
    """
    Combine every generated shot.

    There is intentionally no fixed shot-count limit.
    """

    if not videos:
        raise RuntimeError(
            "No generated videos were supplied for assembly."
        )

    ensure_real_directory(
        OUTPUT_DIR
    )

    concat_file = create_concat_file(
        videos
    )

    final_video = (
        OUTPUT_DIR / FINAL_VIDEO_NAME
    )

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-vf",
        (
            f"scale={WIDTH}:{HEIGHT}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:"
            "(oh-ih)/2,"
            f"fps={FINAL_FPS}"
        ),
        "-c:v",
        "libx264",
        "-preset",
        OUTPUT_PRESET,
        "-crf",
        str(OUTPUT_CRF),
        "-pix_fmt",
        PIXEL_FORMAT,
        "-movflags",
        "+faststart",
        "-an",
        str(final_video),
    ]

    run_command(command)

    if not final_video.exists():
        raise RuntimeError(
            "FFmpeg finished but the final video was not created."
        )

    size_mb = (
        final_video.stat().st_size
        / (1024 * 1024)
    )

    print()
    print("=" * 70)
    print("✅ FINAL STORY VIDEO")
    print("=" * 70)
    print(f"File       : {final_video}")
    print(f"Resolution : {WIDTH}x{HEIGHT}")
    print(f"FPS        : {FINAL_FPS}")
    print(f"Shots      : {len(videos)}")
    print(f"Size       : {size_mb:.2f} MB")
    print("=" * 70)

    return final_video


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a multi-shot cinematic video "
            "from a story."
        )
    )

    parser.add_argument(
        "--story",
        type=str,
        required=True,
        help="Story text to convert into a cinematic video.",
    )

    parser.add_argument(
        "--force-clean",
        action="store_true",
        help=(
            "Ignore any existing shot clips and regenerate "
            "everything from scratch, even if the story and "
            "settings are unchanged."
        ),
    )

    args = parser.parse_args()

    story = args.story.strip()

    if not story:
        raise ValueError(
            "Story cannot be empty."
        )

    check_requirements()

    # CLEAN_TEMPORARY_CLIPS in video_config.py acts as a manual
    # always-clean override on top of --force-clean.
    force_clean = (
        args.force_clean
        or CLEAN_TEMPORARY_CLIPS
    )

    references = index_reference_files()

    print()

    if references:
        print(
            f"✅ Found {len(references)} "
            "character/reference image(s)."
        )

        for item in references:
            print(
                f"  - {item['kind']}: "
                f"{item['path']}"
            )
    else:
        print(
            "No character/reference images found."
        )

    # ----------------------------------------------------------
    # Resume decision: must happen before planning, since a
    # matching fingerprint means we can skip Qwen entirely and
    # reuse the previously saved storyboard.
    # ----------------------------------------------------------

    is_resume = prepare_story_workspace(
        story,
        force_clean=force_clean,
    )

    plan = load_saved_plan() if is_resume else None

    if plan is not None:
        print()
        print(
            "♻️ Reusing previously saved storyboard "
            "(skipping Qwen planning)."
        )
        plan = validate_and_normalize_plan(plan)
    else:
        print()
        print("=" * 70)
        print("PLANNING STORY")
        print("=" * 70)

        print(
            f"Story length: {len(story)} characters"
        )

        plan = plan_story(
            story,
            references,
        )

        plan = validate_and_normalize_plan(
            plan
        )

        save_story_plan(plan)

    print_plan(plan)

    ensure_real_directory(
        OUTPUT_DIR
    )

    generated_videos = []

    total_shots = len(
        plan["shots"]
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

        generated_videos.append(
            video
        )

    final_video = assemble(
        generated_videos
    )

    print()
    print("=" * 70)
    print("🚀 STORY VIDEO GENERATION COMPLETE")
    print("=" * 70)
    print(f"Final video: {final_video}")


if __name__ == "__main__":
    main()
