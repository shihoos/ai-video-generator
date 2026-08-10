from pathlib import Path
import subprocess
import sys

# ============================================================
# AI VIDEO PROJECT - KAGGLE STARTUP
# ============================================================

PROJECT_ROOT = Path("/kaggle/working/ai-video-generator")

KAGGLE_DIR = PROJECT_ROOT / "kaggle"

if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))

from config import (
    LTX_MODEL,
    LTX_UPSCALER,
    LTX_COMMIT,
    LTX_REPO,
    LTX_CONFIG,
    WORK_DIR,
    CLIPS_DIR,
    FRAMES_DIR,
    OUTPUT_DIR,
)


def run(command):
    """Run a shell command and stop if it fails."""
    print(f"\n$ {command}")

    result = subprocess.run(
        command,
        shell=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"\n❌ Command failed: {command}")
        sys.exit(result.returncode)


# ============================================================
# GPU CHECK
# ============================================================

def check_gpu():
    print("\n" + "=" * 60)
    print("GPU CHECK")
    print("=" * 60)

    run(
        "nvidia-smi "
        "--query-gpu=index,name,memory.total "
        "--format=csv"
    )


# ============================================================
# PYTHON ENVIRONMENT
# ============================================================

def check_python_environment():
    print("\n" + "=" * 60)
    print("PYTHON ENVIRONMENT CHECK")
    print("=" * 60)

    packages = [
        "torch",
        "transformers",
        "diffusers",
        "huggingface_hub",
        "av",
    ]

    missing = []

    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")

            print(f"✅ {package}: {version}")

        except ImportError:
            print(f"⚠️ {package}: NOT INSTALLED")
            missing.append(package)

    # Install only missing dependencies.
    if missing:
        print("\nInstalling missing dependencies:")

        for package in missing:
            print(f"  → {package}")
            run(
                f"{sys.executable} -m pip install {package}"
            )

    # Verify PyTorch/CUDA
    import torch

    print(f"✅ CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("❌ CUDA is not available.")
        sys.exit(1)

    print(f"✅ CUDA version: {torch.version.cuda}")

    for index in range(torch.cuda.device_count()):
        print(
            f"✅ GPU {index}: "
            f"{torch.cuda.get_device_name(index)}"
        )


# ============================================================
# MODEL CHECK
# ============================================================

def check_models():
    print("\n" + "=" * 60)
    print("MODEL CHECK")
    print("=" * 60)

    if LTX_MODEL.exists():
        size_gb = LTX_MODEL.stat().st_size / (1024 ** 3)

        print("✅ LTX 2B model found")
        print(f"Path: {LTX_MODEL}")
        print(f"Size: {size_gb:.2f} GB")

    else:
        print("❌ LTX 2B model not found")
        print(f"Expected: {LTX_MODEL}")
        sys.exit(1)

    if LTX_UPSCALER.exists():
        size_mb = LTX_UPSCALER.stat().st_size / (1024 ** 2)

        print("✅ LTX spatial upscaler found")
        print(f"Path: {LTX_UPSCALER}")
        print(f"Size: {size_mb:.0f} MB")

    else:
        print("❌ LTX spatial upscaler not found")
        print(f"Expected: {LTX_UPSCALER}")
        sys.exit(1)


# ============================================================
# DIRECTORY CHECK
# ============================================================

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
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("✅ Working directories ready")


# ============================================================
# LTX REPOSITORY
# ============================================================

def setup_ltx_repository():
    print("\n" + "=" * 60)
    print("LTX-VIDEO REPOSITORY CHECK")
    print("=" * 60)

    if not LTX_REPO.exists():
        print("LTX-Video repository not found.")
        print("Cloning LTX-Video...")

        run(
            f"git clone "
            f"https://github.com/Lightricks/LTX-Video.git "
            f"{LTX_REPO}"
        )

    else:
        print("✅ LTX-Video repository already exists")

    # Get short commit hash
    result = subprocess.run(
        f"git -C {LTX_REPO} rev-parse --short HEAD",
        shell=True,
        capture_output=True,
        text=True,
    )

    current_commit = result.stdout.strip()

    print(f"Current LTX commit: {current_commit}")

    if current_commit != LTX_COMMIT:
        print(
            f"Switching to tested revision: {LTX_COMMIT}"
        )

        run(
            f"git -C {LTX_REPO} checkout {LTX_COMMIT}"
        )

    else:
        print(
            f"✅ LTX-Video revision correct: {LTX_COMMIT}"
        )

    if not LTX_CONFIG.exists():
        print("❌ Official LTX 0.9.8 config not found:")
        print(LTX_CONFIG)
        sys.exit(1)

    print(f"✅ LTX configuration found: {LTX_CONFIG}")


# ============================================================
# LTX PYTHON PACKAGE
# ============================================================

def install_ltx():
    print("\n" + "=" * 60)
    print("LTX PYTHON PACKAGE CHECK")
    print("=" * 60)

    test = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ltx_video; print(ltx_video.__file__)",
        ],
        capture_output=True,
        text=True,
    )

    if test.returncode == 0:
        print("✅ LTX Python package already installed")
        print(test.stdout.strip())
        return

    print("LTX Python package not installed.")
    print("Installing LTX-Video package...")

    run(
        f"cd {LTX_REPO} && "
        f"{sys.executable} -m pip install -e . --no-deps"
    )

    test = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ltx_video; print(ltx_video.__file__)",
        ],
        capture_output=True,
        text=True,
    )

    if test.returncode != 0:
        print("❌ LTX package installation failed.")
        print(test.stderr)
        sys.exit(1)

    print("✅ LTX Python package installed successfully")


# ============================================================
# FINAL STATUS
# ============================================================

def print_final_status():
    print("\n" + "=" * 60)
    print("✅ STARTUP CHECK COMPLETE")
    print("=" * 60)

    print(f"LTX model      : {LTX_MODEL}")
    print(f"LTX upscaler   : {LTX_UPSCALER}")
    print(f"LTX repo       : {LTX_REPO}")
    print(f"LTX revision   : {LTX_COMMIT}")
    print(f"LTX config     : {LTX_CONFIG}")
    print(f"Clips          : {CLIPS_DIR}")
    print(f"Output         : {OUTPUT_DIR}")

    print("\n🚀 LTX environment is ready.")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n")
    print("=" * 60)
    print("🚀 AI VIDEO PROJECT STARTUP")
    print("=" * 60)

    check_gpu()
    check_python_environment()
    check_models()
    setup_directories()
    setup_ltx_repository()
    install_ltx()
    print_final_status()


if __name__ == "__main__":
    main()
