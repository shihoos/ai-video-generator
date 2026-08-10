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
# Local LLM story analysis
#   ↓
# Characters / locations / actions / shots / duration
#   ↓
# Reference matching
#   ↓
# LTX generation
#   ↓
# Scene clips
#   ↓
# 25 FPS final video
#
# Audio will be added as a separate engine later.
# ============================================================


PROJECT_ROOT = Path(
    "/kaggle/working/ai-video-generator"
)

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
)


# ============================================================
# STORY PLANNER MODEL
# ============================================================

PLANNER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PLANNER_MAX_NEW_TOKENS = 2500


# ============================================================
# T4 SAFE RENDERING
# ============================================================

# We KNOW this combination worked on your T4:
#
# 1280x720
# 25 frames
# 8 FPS
#
# Do not change this until the story pipeline is proven.

SAFE_TEST_FRAMES = 25


# ============================================================
# SAMPLE STORY
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
# UTILITIES
# ============================================================

def run_command(command):

    print()
    print("=" * 70)
    print("$ " + " ".join(str(x) for x in command))
    print("=" * 70)

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code "
            f"{result.returncode}"
        )


def clean_name(value):

    value = value.lower()

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
                matches.append(reference)
                break

    return matches


# ============================================================
# LOCAL LLM STORY PLANNER
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

    tokenizer = AutoTokenizer.from_pretrained(
        PLANNER_MODEL
    )

    model = AutoModelForCausalLM.from_pretrained(
        PLANNER_MODEL,
        torch_dtype="auto",
        device_map="cpu",
    )

    model.eval()

    print("✅ Story planner loaded on CPU")

    return tokenizer, model


def unload_story_planner(model, tokenizer):

    del model
    del tokenizer

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception:
        pass

    print("✅ Story planner unloaded")


# ============================================================
# STORY ANALYSIS PROMPT
# ============================================================

def build_planner_prompt(story, reference_names):

    available_refs = ", ".join(
        reference_names
    )

    if not available_refs:
        available_refs = "none"

    return f"""
You are a professional film director and storyboard planner.

Analyze the following story and convert it into a sequence
of cinematic video shots.

The final result will be generated by an AI video model.

IMPORTANT RULES:

1. Preserve the story exactly.
2. Do not invent major characters or events.
3. Split the story into logical cinematic shots.
4. Each shot must represent one clear visual action.
5. Choose an appropriate camera shot.
6. Choose camera movement when useful.
7. Identify characters appearing in each shot.
8. Identify important visual references.
9. Estimate a useful duration between 3 and 7 seconds.
10. Do not create unnecessary shots.
11. Maintain visual continuity between shots.
12. Return ONLY valid JSON.

Available reference filenames:

{available_refs}

Return this exact JSON structure:

{{
  "title": "short title",
  "visual_style": "overall visual style",
  "characters": [
    {{
      "name": "character name",
      "description": "visual description"
    }}
  ],
  "locations": [
    {{
      "name": "location",
      "description": "visual description"
    }}
  ],
  "shots": [
    {{
      "shot_number": 1,
      "duration_seconds": 5,
      "characters": ["name"],
      "location": "location",
      "action": "what happens visually",
      "camera_shot": "wide / medium / close-up / over-the-shoulder / POV / etc",
      "camera_movement": "static / dolly / tracking / pan / tilt / crane / etc",
      "mood": "mood",
      "visual_prompt": "complete cinematic prompt for the video model"
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

    # Remove markdown fences.
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

    # Find first JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:

        candidate = text[
            start:end + 1
        ]

        return json.loads(candidate)

    raise ValueError(
        "Story planner did not return valid JSON."
    )


# ============================================================
# PLAN STORY
# ============================================================

def plan_story(story, references):

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
                "You are a professional film "
                "storyboard planner. "
                "Return valid JSON only."
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

    with __import__("torch").no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=PLANNER_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
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

    plan = extract_json(
        response
    )

    unload_story_planner(
        model,
        tokenizer,
    )

    return plan


# ============================================================
# VALIDATE PLAN
# ============================================================

def validate_plan(plan):

    if not isinstance(plan, dict):
        raise ValueError(
            "Story plan is not a JSON object."
        )

    shots = plan.get("shots")

    if not isinstance(shots, list):
        raise ValueError(
            "Story plan contains no shots."
        )

    if not shots:
        raise ValueError(
            "Story planner produced zero shots."
        )

    for index, shot in enumerate(
        shots,
        start=1,
    ):

        shot["shot_number"] = index

        duration = shot.get(
            "duration_seconds",
            DEFAULT_SHOT_SECONDS,
        )

        try:
            duration = float(duration)

        except Exception:
            duration = DEFAULT_SHOT_SECONDS

        duration = max(
            3.0,
            min(
                7.0,
                duration,
            ),
        )

        shot["duration_seconds"] = duration

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

    print(
        "Characters:",
        ", ".join(
            item["name"]
            for item in plan.get(
                "characters",
                [],
            )
        ),
    )

    shots = plan["shots"]

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
            f"Movement : "
            f"{shot['camera_movement']}"
        )

        print()


# ============================================================
# GENERATE SCENE
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

    shot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    characters = shot.get(
        "characters",
        [],
    )

    matched = match_references(
        characters,
        references,
    )

    prompt = shot["visual_prompt"]

    # Add continuity instructions.
    prompt += (
        ". Maintain strict visual continuity "
        "with the established characters, "
        "clothing, environment and lighting. "
        f"Camera: {shot['camera_shot']}. "
        f"Camera movement: "
        f"{shot['camera_movement']}. "
        f"Mood: {shot['mood']}. "
        "Realistic cinematic lighting, "
        "natural motion, detailed textures, "
        "physically believable movement, "
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

    if matched:

        print("References:")

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
    # IMPORTANT:
    #
    # The current T4-safe test uses 25 frames.
    #
    # Later we will add long-shot continuation.
    # --------------------------------------------------------

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
        str(SAFE_TEST_FRAMES),

        "--fps",
        str(SOURCE_FPS),

        "--seed",
        str(
            BASE_SEED + shot_index
        ),

        "--output",
        str(shot_dir),
    ]

    if USE_CPU_OFFLOAD:
        command.append(
            "--offload"
        )

    # --------------------------------------------------------
    # Image conditioning
    # --------------------------------------------------------

    if matched:

        media_paths = [
            str(item["path"])
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

    run_command(command)

    videos = list(
        shot_dir.glob("*.mp4")
    )

    if not videos:

        raise RuntimeError(
            f"No video generated for shot "
            f"{shot_index}"
        )

    videos.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    video = videos[0]

    print(
        f"✅ Shot generated: {video}"
    )

    return video


# ============================================================
# CONCAT
# ============================================================

def create_concat_file(videos):

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
                str(video.resolve())
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

def assemble(videos):

    concat = create_concat_file(
        videos
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_video = (
        OUTPUT_DIR
        / "story_final.mp4"
    )

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
        f"File       : {final_video}"
    )

    print(
        f"Resolution : {WIDTH}x{HEIGHT}"
    )

    print(
        f"FPS        : {FINAL_FPS}"
    )

    print(
        f"Shots      : {len(videos)}"
    )

    print(
        f"Size       : {size_mb:.2f} MB"
    )

    return final_video


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate a multi-shot cinematic "
            "video from a story."
        )
    )

    parser.add_argument(
        "--story",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--story-file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--keep-clips",
        action="store_true",
        help="Keep previous story clips.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Requirements
    # --------------------------------------------------------

    if not GENERATE_SCRIPT.exists():

        raise FileNotFoundError(
            f"generate.py not found:\n"
            f"{GENERATE_SCRIPT}"
        )

    if shutil.which("ffmpeg") is None:

        raise RuntimeError(
            "FFmpeg is unavailable."
        )

    CHARACTER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CLIPS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Story
    # --------------------------------------------------------

    if args.story_file:

        story_path = Path(
            args.story_file
        )

        if not story_path.exists():

            raise FileNotFoundError(
                story_path
            )

        story = story_path.read_text(
            encoding="utf-8"
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

    # --------------------------------------------------------
    # References
    # --------------------------------------------------------

    references = index_reference_files()

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

    # --------------------------------------------------------
    # Story planning
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Clean clips
    # --------------------------------------------------------

    if (
        CLEAN_TEMPORARY_CLIPS
        and not args.keep_clips
        and CLIPS_DIR.exists()
    ):

        shutil.rmtree(
            CLIPS_DIR
        )

        CLIPS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Generate shots
    # --------------------------------------------------------

    generated = []

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

        # Keep GPU memory clean.
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception:
            pass

    # --------------------------------------------------------
    # Assemble
    # --------------------------------------------------------

    final_video = assemble(
        generated
    )

    print()
    print("=" * 70)
    print("🚀 GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Final video:\n{final_video}"
    )


if __name__ == "__main__":
    main()
