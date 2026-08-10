from pathlib import Path
import subprocess
import sys
import os


# ============================================================
# AI VIDEO PROJECT - KAGGLE STARTUP
# ============================================================

PROJECT_ROOT = Path("/kaggle/working/ai-video-generator")

DATASET_ROOT = Path(
    "/kaggle/input/datasets/shihoos/ai-video-model"
)

LTX_MODEL = DATASET_ROOT / "ltxv-2b-0.9.8-distilled.safetensors"

LTX_REPO = PROJECT_ROOT / "LTX-Video-0.9.8"

WORK_DIR = PROJECT_ROOT / "work"
CLIPS_DIR = WORK_DIR / "clips"
OUTPUT_DIR = WORK_DIR / "output"


def run(command):
    """Run a shell command and stop if it fails."""
    print(f"\n$ {command}")
    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"\n❌ Command failed: {command}")
        sys.exit(result.returncode)


def check_gpu():
    print("\n" + "=" * 60)
    print("GPU CHECK")
    print("=" * 60)

    run("nvidia-smi --query-gpu=name,memory.total --format=csv")


def check_model():
    print("\n" + "=" * 60)
    print("MODEL CHECK")
    print("=" * 60)

    if LTX_MODEL.exists():
        size_gb = LTX_MODEL.stat().st_size / (1024 ** 3)

        print("✅ LTX model found")
        print(f"Path: {LTX_MODEL}")
        print(f"Size: {size_gb:.2f} GB")
    else:
        print("❌ LTX model not found")
        print(f"Expected: {LTX_MODEL}")
        sys.exit(1)


def setup_directories():
    print("\n" + "=" * 60)
    print("DIRECTORY CHECK")
    print("=" * 60)

    for directory in [
        PROJECT_ROOT,
        WORK_DIR,
        CLIPS_DIR,
        OUTPUT_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    print("✅ Working directories ready")


def clone_ltx():
    print("\n" + "=" * 60)
    print("LTX-VIDEO CHECK")
    print("=" * 60)

    if LTX_REPO.exists():
        print("✅ LTX-Video repository already exists")
        return

    print("LTX-Video repository not found.")
    print("Cloning LTX-Video 0.9.8...")

    run(
        f"git clone --branch v0.9.8 "
        f"https://github.com/Lightricks/LTX-Video.git "
        f"{LTX_REPO}"
    )


def install_ltx():
    print("\n" + "=" * 60)
    print("LTX INSTALLATION CHECK")
    print("=" * 60)

    marker = LTX_REPO / "ltx_video.egg-info"

    if marker.exists():
        print("✅ LTX Python package already installed")
        return

    print("Installing LTX-Video package...")

    run(
        f"cd {LTX_REPO} && "
        f"pip install -e . --no-deps"
    )


def main():
    print("\n")
    print("=" * 60)
    print("🚀 AI VIDEO PROJECT STARTUP")
    print("=" * 60)

    check_gpu()
    check_model()
    setup_directories()
    clone_ltx()
    install_ltx()

    print("\n" + "=" * 60)
    print("✅ STARTUP CHECK COMPLETE")
    print("=" * 60)

    print(f"LTX model : {LTX_MODEL}")
    print(f"LTX repo  : {LTX_REPO}")
    print(f"Clips     : {CLIPS_DIR}")
    print(f"Output    : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
