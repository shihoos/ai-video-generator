from pathlib import Path
import subprocess
import sys
import shutil
import inspect


# ============================================================
# AI VIDEO PROJECT - KAGGLE START
# ============================================================

PROJECT_ROOT = Path(
    "/kaggle/working/ai-video-generator"
)

KAGGLE_DIR = PROJECT_ROOT / "kaggle"

# LTX repository cloned into Kaggle working storage.
LTX_REPO = PROJECT_ROOT / "LTX-Video-0.9.8"

# ============================================================
# PYTHON IMPORT PATH
# ============================================================

if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))


from config import (
    LTX_MODEL,
    LTX_UPSCALER,
    LTX_COMMIT,
    LTX_REPO as CONFIG_LTX_REPO,
    LTX_CONFIG,
    WORK_DIR,
    CLIPS_DIR,
    FRAMES_DIR,
    OUTPUT_DIR,
)

# ============================================================
# QWEN STORY PLANNER
# ============================================================
#
# The Qwen Dataset path now has a single source of truth in
# video_config.py. start.py no longer defines it separately.

from video_config import (
    QWEN_LOCAL_MODEL,
)


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_LTX_COMMIT = (
    "bdc8f017f0148a0f0bb9e3a5049d2d356423cee0"
)

EXPECTED_TRANSFORMERS_MAJOR = "5"


# ============================================================
# COMMAND HELPER
# ============================================================

def run(command, cwd=None):
    """Run a command and stop if it fails."""

    if isinstance(command, (list, tuple)):
        printable = " ".join(
            str(x) for x in command
        )
    else:
        printable = str(command)

    print()
    print("$ " + printable)

    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
    )

    if result.returncode != 0:
        print()
        print(
            f"❌ Command failed: {printable}"
        )

        sys.exit(result.returncode)

    return result


# ============================================================
# GPU CHECK
# ============================================================

def check_gpu():

    print()
    print("=" * 70)
    print("GPU CHECK")
    print("=" * 70)

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv",
        ],
        text=True,
    )

    if result.returncode != 0:
        print("❌ nvidia-smi failed")
        sys.exit(result.returncode)


# ============================================================
# PYTHON ENVIRONMENT
# ============================================================

def check_python_environment():

    print()
    print("=" * 70)
    print("PYTHON ENVIRONMENT CHECK")
    print("=" * 70)

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

            version = getattr(
                module,
                "__version__",
                "unknown",
            )

            print(
                f"✅ {package}: {version}"
            )

        except ImportError:

            print(
                f"⚠️ {package}: NOT INSTALLED"
            )

            missing.append(package)

    # --------------------------------------------------------
    # Only install AV if missing.
    #
    # NEVER automatically upgrade/downgrade:
    # torch
    # transformers
    # diffusers
    # huggingface_hub
    # --------------------------------------------------------

    for package in missing:

        if package == "av":

            print(
                "\nInstalling missing dependency: av"
            )

            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "av",
                ]
            )

        else:

            print(
                f"\n❌ Required package missing: "
                f"{package}"
            )

            print(
                "The startup script will not "
                "modify the ML environment."
            )

            sys.exit(1)

    # --------------------------------------------------------
    # Import and verify versions
    # --------------------------------------------------------

    import torch
    import transformers
    import huggingface_hub
    import diffusers

    print()
    print("-" * 70)

    print(
        "torch           :",
        torch.__version__,
    )

    print(
        "transformers    :",
        transformers.__version__,
    )

    print(
        "huggingface_hub :",
        huggingface_hub.__version__,
    )

    print(
        "diffusers       :",
        diffusers.__version__,
    )

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        print(
            "\n❌ CUDA is not available."
        )

        sys.exit(1)

    print(
        "\n✅ CUDA available"
    )

    print(
        "CUDA version:",
        torch.version.cuda,
    )

    for index in range(
        torch.cuda.device_count()
    ):

        print(
            f"GPU {index}: "
            f"{torch.cuda.get_device_name(index)}"
        )

    # --------------------------------------------------------
    # Transformers 5.x
    # --------------------------------------------------------

    if not transformers.__version__.startswith(
        EXPECTED_TRANSFORMERS_MAJOR + "."
    ):

        print(
            "\n❌ Unexpected Transformers version:"
        )

        print(
            transformers.__version__
        )

        print(
            "Expected Transformers 5.x."
        )

        sys.exit(1)

    print(
        "\n✅ Transformers 5.x detected"
    )

    print(
        "✅ Hugging Face Hub:",
        huggingface_hub.__version__,
    )

    print(
        "✅ Diffusers:",
        diffusers.__version__,
    )


# ============================================================
# MODEL CHECK
# ============================================================

def check_models():

    print()
    print("=" * 70)
    print("MODEL CHECK")
    print("=" * 70)

    # --------------------------------------------------------
    # LTX MODEL
    # --------------------------------------------------------

    if not LTX_MODEL.exists():

        print(
            "❌ LTX 2B model not found:"
        )

        print(
            LTX_MODEL
        )

        sys.exit(1)

    size_gb = (
        LTX_MODEL.stat().st_size
        / (1024 ** 3)
    )

    print(
        "✅ LTX 2B model found"
    )

    print(
        f"Path: {LTX_MODEL}"
    )

    print(
        f"Size: {size_gb:.2f} GB"
    )

    # --------------------------------------------------------
    # UPSCALER
    # --------------------------------------------------------

    if not LTX_UPSCALER.exists():

        print(
            "❌ LTX spatial upscaler not found:"
        )

        print(
            LTX_UPSCALER
        )

        sys.exit(1)

    size_mb = (
        LTX_UPSCALER.stat().st_size
        / (1024 ** 2)
    )

    print(
        "✅ LTX spatial upscaler found"
    )

    print(
        f"Path: {LTX_UPSCALER}"
    )

    print(
        f"Size: {size_mb:.0f} MB"
    )

    # --------------------------------------------------------
    # QWEN
    # --------------------------------------------------------

    if QWEN_LOCAL_MODEL.is_dir():

        print(
            "\n✅ Qwen3 story planner found"
        )

        print(
            f"Path: {QWEN_LOCAL_MODEL}"
        )

        # Count model files
        model_files = list(
            QWEN_LOCAL_MODEL.glob(
                "*.safetensors"
            )
        )

        print(
            f"Model weight files: "
            f"{len(model_files)}"
        )

    else:

        print(
            "\n⚠️ Qwen3 local model not found:"
        )

        print(
            QWEN_LOCAL_MODEL
        )

        print(
            "The story planner will use "
            "its configured fallback if available."
        )


# ============================================================
# SAFE DIRECTORY
# ============================================================

def ensure_real_directory(path):

    path = Path(path)

    # --------------------------------------------------------
    # Existing directory
    # --------------------------------------------------------

    if path.exists() and path.is_dir():
        return path

    # --------------------------------------------------------
    # Existing symlink
    # --------------------------------------------------------

    if path.is_symlink():

        print(
            f"⚠️ Removing invalid symlink: {path}"
        )

        path.unlink()

    # --------------------------------------------------------
    # Existing file
    # --------------------------------------------------------

    elif path.exists():

        print(
            f"⚠️ Removing invalid file: {path}"
        )

        path.unlink()

    # --------------------------------------------------------
    # Create directory
    # --------------------------------------------------------

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


# ============================================================
# DIRECTORY SETUP
# ============================================================

def setup_directories():

    print()
    print("=" * 70)
    print("DIRECTORY CHECK")
    print("=" * 70)

    directories = [
        PROJECT_ROOT,
        WORK_DIR,
        CLIPS_DIR,
        FRAMES_DIR,
        OUTPUT_DIR,
    ]

    for directory in directories:

        ensure_real_directory(
            directory
        )

        print(
            f"✅ {directory}"
        )

    print(
        "\n✅ Working directories ready"
    )


# ============================================================
# LTX REPOSITORY
# ============================================================

def setup_ltx_repository():

    print()
    print("=" * 70)
    print("LTX-VIDEO REPOSITORY CHECK")
    print("=" * 70)

    # --------------------------------------------------------
    # Clone repository if needed
    # --------------------------------------------------------

    if not LTX_REPO.exists():

        print(
            "LTX-Video repository not found."
        )

        print(
            "Cloning LTX-Video..."
        )

        run(
            [
                "git",
                "clone",
                "https://github.com/Lightricks/LTX-Video.git",
                str(LTX_REPO),
            ]
        )

    else:

        print(
            "✅ LTX repository exists:"
        )

        print(
            LTX_REPO
        )

    # --------------------------------------------------------
    # Verify Git repository
    # --------------------------------------------------------

    result = subprocess.run(
        [
            "git",
            "-C",
            str(LTX_REPO),
            "rev-parse",
            "--is-inside-work-tree",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        print(
            "❌ LTX directory is not "
            "a valid Git repository."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Fetch repository
    # --------------------------------------------------------

    run(
        [
            "git",
            "-C",
            str(LTX_REPO),
            "fetch",
            "--all",
            "--tags",
        ]
    )

    # --------------------------------------------------------
    # Checkout exact tested revision
    # --------------------------------------------------------

    print(
        "\nChecking out tested LTX revision:"
    )

    print(
        EXPECTED_LTX_COMMIT
    )

    run(
        [
            "git",
            "-C",
            str(LTX_REPO),
            "checkout",
            "--force",
            EXPECTED_LTX_COMMIT,
        ]
    )

    # --------------------------------------------------------
    # Verify exact commit
    # --------------------------------------------------------

    result = subprocess.run(
        [
            "git",
            "-C",
            str(LTX_REPO),
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    current_commit = result.stdout.strip()

    print(
        "\nLTX commit:"
    )

    print(
        current_commit
    )

    if current_commit != EXPECTED_LTX_COMMIT:

        print(
            "\n❌ LTX revision mismatch."
        )

        print(
            f"Expected: {EXPECTED_LTX_COMMIT}"
        )

        print(
            f"Found:    {current_commit}"
        )

        sys.exit(1)

    print(
        "\n✅ LTX 0.9.8 revision confirmed"
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    if not LTX_CONFIG.exists():

        print(
            "\n❌ LTX configuration not found:"
        )

        print(
            LTX_CONFIG
        )

        sys.exit(1)

    print(
        "✅ LTX configuration found:"
    )

    print(
        LTX_CONFIG
    )


# ============================================================
# LTX SOURCE IMPORT
# ============================================================

def setup_ltx_source():

    print()
    print("=" * 70)
    print("LTX SOURCE IMPORT CHECK")
    print("=" * 70)

    # Put LTX source FIRST on Python import path.
    if str(LTX_REPO) not in sys.path:

        sys.path.insert(
            0,
            str(LTX_REPO),
        )

    try:

        import ltx_video

        imported_path = Path(
            inspect.getfile(
                ltx_video
            )
        ).resolve()

    except Exception as exc:

        print(
            "\n❌ Could not import LTX:"
        )

        print(
            repr(exc)
        )

        sys.exit(1)

    expected_root = (
        LTX_REPO / "ltx_video"
    ).resolve()

    print(
        "LTX imported from:"
    )

    print(
        imported_path
    )

    # --------------------------------------------------------
    # Make sure Python did NOT import a pip-installed
    # version from site-packages.
    # --------------------------------------------------------

    try:

        imported_path.relative_to(
            expected_root
        )

    except ValueError:

        print(
            "\n❌ WRONG LTX IMPORT LOCATION"
        )

        print(
            f"Expected under:\n"
            f"{expected_root}"
        )

        print(
            f"Actually imported from:\n"
            f"{imported_path}"
        )

        sys.exit(1)

    print(
        "\n✅ LTX source import is correct"
    )


# ============================================================
# REMOVE OLD LTX PIP PACKAGE
# ============================================================

def remove_old_ltx_package():

    print()
    print("=" * 70)
    print("LTX PIP PACKAGE CLEANUP")
    print("=" * 70)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "show",
            "ltx-video",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        print(
            "✅ Separate ltx-video pip package "
            "is not installed."
        )

        return

    print(
        "⚠️ Old ltx-video pip package detected."
    )

    print(
        "Removing it."
    )

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "ltx-video",
        ]
    )

    print(
        "✅ Old ltx-video pip package removed"
    )


# ============================================================
# FINAL LTX VERIFICATION
# ============================================================

def verify_ltx():

    print()
    print("=" * 70)
    print("FINAL LTX VERIFICATION")
    print("=" * 70)

    if str(LTX_REPO) not in sys.path:

        sys.path.insert(
            0,
            str(LTX_REPO),
        )

    try:

        import ltx_video

        imported_path = Path(
            inspect.getfile(
                ltx_video
            )
        ).resolve()

    except Exception as exc:

        print(
            "❌ LTX import failed:"
        )

        print(
            repr(exc)
        )

        sys.exit(1)

    expected_root = (
        LTX_REPO / "ltx_video"
    ).resolve()

    print(
        "LTX imported from:"
    )

    print(
        imported_path
    )

    try:

        imported_path.relative_to(
            expected_root
        )

    except ValueError:

        print(
            "\n❌ LTX is being imported "
            "from the wrong location."
        )

        sys.exit(1)

    print(
        "\n✅ LTX source verification passed"
    )


# ============================================================
# QWEN VERIFICATION
# ============================================================

def verify_qwen():

    print()
    print("=" * 70)
    print("QWEN3 STORY PLANNER CHECK")
    print("=" * 70)

    if not QWEN_LOCAL_MODEL.is_dir():

        print(
            "⚠️ Local Qwen3 model not found."
        )

        print(
            QWEN_LOCAL_MODEL
        )

        print(
            "Story planner fallback may be used."
        )

        return

    required_files = [
        "config.json",
        "tokenizer_config.json",
    ]

    missing = []

    for filename in required_files:

        path = (
            QWEN_LOCAL_MODEL
            / filename
        )

        if not path.exists():

            missing.append(
                filename
            )

    if missing:

        print(
            "❌ Qwen3 directory exists but "
            "required files are missing:"
        )

        for filename in missing:
            print(
                f"  - {filename}"
            )

        sys.exit(1)

    print(
        "✅ Qwen3 local model ready"
    )

    print(
        f"Path: {QWEN_LOCAL_MODEL}"
    )


# ============================================================
# FINAL STATUS
# ============================================================

def print_final_status():

    print()
    print("=" * 70)
    print("✅ START CHECK COMPLETE")
    print("=" * 70)

    print(
        f"LTX model      : {LTX_MODEL}"
    )

    print(
        f"LTX upscaler   : {LTX_UPSCALER}"
    )

    print(
        f"LTX repository  : {LTX_REPO}"
    )

    print(
        f"LTX revision   : {EXPECTED_LTX_COMMIT}"
    )

    print(
        f"LTX config     : {LTX_CONFIG}"
    )

    print(
        f"Qwen3 model    : {QWEN_LOCAL_MODEL}"
    )

    print(
        f"Clips          : {CLIPS_DIR}"
    )

    print(
        f"Frames         : {FRAMES_DIR}"
    )

    print(
        f"Output         : {OUTPUT_DIR}"
    )

    print(
        "\n🚀 AI video environment is ready."
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("🚀 AI VIDEO PROJECT START")
    print("=" * 70)

    # 1. GPU
    check_gpu()

    # 2. Python environment
    check_python_environment()

    # 3. Models
    check_models()

    # 4. Directories
    setup_directories()

    # 5. LTX repository
    setup_ltx_repository()

    # 6. T4 patch
    print()
    print("=" * 70)
    print("APPLYING LTX T4 PATCH")
    print("=" * 70)

    run(
        [
            sys.executable,
            str(
                KAGGLE_DIR
                / "apply_ltx_t4_patch.py"
            ),
        ]
    )

    # 7. Direct LTX source import
    setup_ltx_source()

    # 8. Remove obsolete pip package
    remove_old_ltx_package()

    # 9. Verify LTX again
    verify_ltx()

    # 10. Qwen3
    verify_qwen()

    # 11. Final status
    print_final_status()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
