from pathlib import Path
import subprocess
import sys
import shutil


# ============================================================
# SAMURAI CINEMATIC VIDEO GENERATOR
# ============================================================

PROJECT_ROOT = Path("/kaggle/working/ai-video-generator")

GENERATE_SCRIPT = PROJECT_ROOT / "kaggle" / "generate.py"

BASE_OUTPUT = PROJECT_ROOT / "work" / "samurai_clips"
FINAL_OUTPUT = PROJECT_ROOT / "work" / "output"

WIDTH = 1280
HEIGHT = 720

# 25 frames at 8 FPS = 3.125 seconds per clip
FRAMES_PER_CLIP = 25
FPS = 8

BASE_SEED = 20260810


SCENES = [
    {
        "name": "scene_01_forest",
        "prompt": (
            "A cinematic historical Japanese samurai warrior walking slowly "
            "alone through a misty ancient Japanese forest at dawn, wearing "
            "traditional dark samurai armor, long katana sheathed at his side, "
            "red maple leaves drifting through the air, soft golden sunlight "
            "filtering through tall trees, atmospheric fog, realistic "
            "historical Japanese environment, dramatic cinematic composition, "
            "slow camera tracking movement, highly detailed realistic "
            "textures, natural subtle movement, serious calm expression, "
            "no blood, no gore"
        ),
    },
    {
        "name": "scene_02_alert",
        "prompt": (
            "The same historical Japanese samurai warrior from the previous "
            "scene standing in the same misty ancient Japanese forest at dawn, "
            "wearing the same dark samurai armor with the same long katana at "
            "his side, red maple leaves drifting through the fog, golden "
            "sunlight between tall trees, the warrior suddenly stops and "
            "slowly looks toward a distant sound, his hand moving toward the "
            "katana handle, tense but controlled body language, cinematic "
            "slow camera push, realistic historical Japanese environment, "
            "natural movement, highly detailed, serious dramatic mood, "
            "no blood, no gore"
        ),
    },
    {
        "name": "scene_03_draw_sword",
        "prompt": (
            "The same historical Japanese samurai warrior in the same misty "
            "Japanese forest at dawn, wearing the same traditional dark "
            "samurai armor, red maple leaves floating through the air, "
            "soft golden sunlight and atmospheric fog, the warrior slowly "
            "draws his long katana from its sheath and takes a defensive "
            "stance, controlled deliberate sword movement, cinematic camera "
            "push toward the warrior, realistic metal reflections, realistic "
            "historical Japanese environment, highly detailed, natural "
            "movement, serious cinematic atmosphere, no blood, no gore"
        ),
    },
    {
        "name": "scene_04_ready",
        "prompt": (
            "The same historical Japanese samurai warrior standing ready "
            "with his drawn katana in the same ancient misty Japanese forest "
            "at dawn, same dark samurai armor, same environment and red maple "
            "leaves, golden sunlight breaking through the trees, atmospheric "
            "fog moving naturally, the warrior calmly raises his katana into "
            "a defensive position while the camera slowly moves closer, "
            "dramatic cinematic composition, realistic historical Japanese "
            "environment, highly detailed armor and sword, natural subtle "
            "movement, powerful serious mood, no blood, no gore"
        ),
    },
]


def run_command(command):
    """Run a command and stop if it fails."""

    print("\n" + "=" * 70)
    print("$ " + " ".join(str(x) for x in command))
    print("=" * 70)

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}"
        )


def check_requirements():
    print("\n" + "=" * 70)
    print("CHECKING SAMURAI VIDEO GENERATOR")
    print("=" * 70)

    if not GENERATE_SCRIPT.exists():
        raise FileNotFoundError(
            f"generate.py not found:\n{GENERATE_SCRIPT}"
        )

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg is not installed or not available in PATH."
        )

    print(f"✅ generate.py: {GENERATE_SCRIPT}")
    print("✅ FFmpeg available")


def clean_previous_clips():
    print("\n" + "=" * 70)
    print("CLEANING PREVIOUS SAMURAI CLIPS")
    print("=" * 70)

    if BASE_OUTPUT.exists():
        shutil.rmtree(BASE_OUTPUT)

    BASE_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"✅ Clean clip directory: {BASE_OUTPUT}")


def find_generated_video(directory):
    """Return the newest MP4 generated in a directory."""

    videos = list(directory.glob("*.mp4"))

    if not videos:
        raise RuntimeError(
            f"No MP4 video was generated in:\n{directory}"
        )

    videos.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    return videos[0]


def generate_scene(scene, index):
    """Generate one cinematic scene."""

    scene_number = index + 1
    scene_output = BASE_OUTPUT / scene["name"]

    scene_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed = BASE_SEED + index

    print("\n")
    print("#" * 70)
    print(f"🎬 GENERATING SCENE {scene_number}/{len(SCENES)}")
    print("#" * 70)

    print(f"Scene : {scene['name']}")
    print(f"Seed  : {seed}")
    print(f"Size  : {WIDTH}x{HEIGHT}")
    print(f"Frames: {FRAMES_PER_CLIP}")
    print(f"FPS   : {FPS}")

    command = [
        sys.executable,
        str(GENERATE_SCRIPT),
        "--prompt",
        scene["prompt"],
        "--width",
        str(WIDTH),
        "--height",
        str(HEIGHT),
        "--frames",
        str(FRAMES_PER_CLIP),
        "--fps",
        str(FPS),
        "--seed",
        str(seed),
        "--output",
        str(scene_output),
        "--offload",
    ]

    run_command(command)

    video = find_generated_video(scene_output)

    print(f"✅ Scene {scene_number} generated:")
    print(video)

    return video


def create_concat_file(video_paths):
    """Create an FFmpeg concat input file."""

    concat_file = BASE_OUTPUT / "concat.txt"

    with concat_file.open("w", encoding="utf-8") as file:
        for video in video_paths:
            # FFmpeg concat format
            file.write(
                f"file '{video.resolve()}'\n"
            )

    print(f"✅ FFmpeg concat file created:")
    print(concat_file)

    return concat_file


def stitch_videos(video_paths):
    """Join all scenes into one final 720p MP4."""

    print("\n" + "=" * 70)
    print("🎞️ STITCHING SAMURAI SCENES")
    print("=" * 70)

    concat_file = create_concat_file(video_paths)

    final_video = (
        FINAL_OUTPUT
        / "samurai_cinematic_720p_12sec.mp4"
    )

    # Re-encode for consistent H.264 output.
    # This avoids problems caused by slightly different
    # stream parameters between generated clips.
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(final_video),
    ]

    run_command(command)

    if not final_video.exists():
        raise RuntimeError(
            "FFmpeg finished but final video was not created."
        )

    size_mb = final_video.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 70)
    print("✅ FINAL VIDEO CREATED")
    print("=" * 70)

    print(f"File : {final_video}")
    print(f"Size : {size_mb:.2f} MB")
    print("Resolution: 1280x720")
    print("Duration : approximately 12.5 seconds")
    print(f"Scenes   : {len(video_paths)}")

    return final_video


def main():
    print("\n")
    print("=" * 70)
    print("🎬 AI SAMURAI CINEMATIC VIDEO")
    print("=" * 70)

    check_requirements()

    clean_previous_clips()

    generated_videos = []

    for index, scene in enumerate(SCENES):
        video = generate_scene(
            scene,
            index,
        )

        generated_videos.append(video)

    final_video = stitch_videos(
        generated_videos
    )

    print("\n")
    print("=" * 70)
    print("🚀 SAMURAI VIDEO GENERATION COMPLETE")
    print("=" * 70)

    print(f"\nFinal video:")
    print(final_video)


if __name__ == "__main__":
    main()
