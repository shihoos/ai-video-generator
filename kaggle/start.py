from pathlib import Path
import subprocess
import sys
import shutil
import inspect


# ============================================================
# AI VIDEO PROJECT - KAGGLE STARTUP
# ============================================================

PROJECT_ROOT = Path(
    "/kaggle/working/ai-video-generator"
)

KAGGLE_DIR = PROJECT_ROOT / "kaggle"

# LTX repository inside the temporary Kaggle working directory.
LTX_REPO = PROJECT_ROOT / "LTX-Video-0.9.8"


# ============================================================
# PYTHON IMPORT PATH
# ============================================================

if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))


# Import project configuration.
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
# CONSTANTS
# ============================================================

EXPECTED_TRANSFORMERS_MAJOR = "5"
EXPECTED_LTX_COMMIT = "bdc8f017f0148a0f0bb9e3a5049d2d356423cee0"


# ============================================================
# COMMAND HELPER
# ============================================================

def run(command, cwd=None):
    """
    Run a command safely.

    command can be:
        ["git", "status"]

    or a string when shell execution is actually required.
    """

    if isinstance(command, (list, tuple)):
        printable = " ".join(str(x) for x in command)
    else:
        printable = str(command)

    print(f"\n$ {printable}")

    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
    )

    if result.returncode != 0:
        print(
            f"\n❌ Command failed with exit code "
            f"{result.returncode}"
        )
        sys.exit(result.returncode)

    return result


# ============================================================
# GPU CHECK
# ============================================================

def check_gpu():

    print("\n" + "=" * 60)
    print("GPU CHECK")
    print("=" * 60)

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
    # Only install AV if it is missing.
    #
    # DO NOT automatically upgrade/downgrade:
    # torch
    # transformers
    # diffusers
    # huggingface_hub
    # --------------------------------------------------------

    if missing:

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
                    "This startup script will not "
                    "automatically modify the ML environment."
                )

                sys.exit(1)

    # --------------------------------------------------------
    # Verify versions after optional AV installation.
    # --------------------------------------------------------

    import torch
    import transformers
    import huggingface_hub
    import diffusers

    print("\n" + "-" * 60)

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

    print(
        "\nCUDA available:",
        torch.cuda.is_available(),
    )

    if not torch.cuda.is_available():

        print("❌ CUDA is not available.")
        sys.exit(1)

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
    # Modern Transformers requirement
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
        "\n✅ Modern Transformers 5.x detected"
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

    print("\n" + "=" * 60)
    print("MODEL CHECK")
    print("=" * 60)

    # --------------------------------------------------------
    # LTX 2B
    # --------------------------------------------------------

    if LTX_MODEL.exists():

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

    else:

        print(
            "❌ LTX 2B model not found"
        )

        print(
            f"Expected: {LTX_MODEL}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Spatial upscaler
    # --------------------------------------------------------

    if LTX_UPSCALER.exists():

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

    else:

        print(
            "❌ LTX spatial upscaler not found"
        )

        print(
            f"Expected: {LTX_UPSCALER}"
        )

        sys.exit(1)


# ============================================================
# SAFE DIRECTORY CREATION
# ============================================================

def ensure_directory(directory):

    directory = Path(directory)

    # --------------------------------------------------------
    # If a FILE exists where a directory should be,
    # remove it.
    #
    # This permanently prevents errors such as:
    #
    # FileExistsError:
    # work/output
    #
    # --------------------------------------------------------

    if directory.exists() or directory.is_symlink():

        if directory.is_dir() and not directory.is_symlink():

            # Already a valid directory.
            return

        print(
            f"⚠️ Invalid directory path detected:"
        )

        print(
            f"   {directory}"
        )

        print(
            "   Removing invalid file/symlink..."
        )

        if directory.is_symlink() or directory.is_file():

            directory.unlink()

        else:

            shutil.rmtree(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# DIRECTORY CHECK
# ============================================================

def setup_directories():

    print("\n" + "=" * 60)
    print("DIRECTORY CHECK")
    print("=" * 60)

    directories = [
        PROJECT_ROOT,
        WORK_DIR,
        CLIPS_DIR,
        FRAMES_DIR,
        OUTPUT_DIR,
    ]

    for directory in directories:

        ensure_directory(directory)

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

    print("\n" + "=" * 60)
    print("LTX-VIDEO REPOSITORY CHECK")
    print("=" * 60)

    # --------------------------------------------------------
    # Clone if necessary
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
            f"✅ LTX repository exists:"
        )

        print(
            LTX_REPO
        )

    # --------------------------------------------------------
    # Verify it is actually a Git repository
    # --------------------------------------------------------

    git_check = subprocess.run(
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

    if git_check.returncode != 0:

        print(
            "❌ LTX directory exists but is not "
            "a valid Git repository."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Fetch required revision
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
    # Checkout exact tested LTX revision
    # --------------------------------------------------------

    print(
        "\nSwitching to tested LTX revision:"
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
        f"\nCurrent LTX commit:"
    )

    print(
        current_commit
    )

    if current_commit != EXPECTED_LTX_COMMIT:

        print(
            "\n❌ LTX revision verification failed."
        )

        print(
            f"Expected: {EXPECTED_LTX_COMMIT}"
        )

        print(
            f"Found:    {current_commit}"
        )

        sys.exit(1)

    print(
        "\n✅ LTX-Video 0.9.8 revision confirmed"
    )

    # --------------------------------------------------------
    # Verify configuration
    # --------------------------------------------------------

    if not LTX_CONFIG.exists():

        print(
            "\n❌ Official LTX configuration not found:"
        )

        print(
            LTX_CONFIG
        )

        sys.exit(1)

    print(
        f"✅ LTX configuration found:"
    )

    print(
        LTX_CONFIG
    )


# ============================================================
# LTX SOURCE PATH
# ============================================================

def setup_ltx_source():

    print("\n" + "=" * 60)
    print("LTX SOURCE IMPORT CHECK")
    print("=" * 60)

    # --------------------------------------------------------
    # Put repository directly on Python's import path.
    #
    # We intentionally do NOT use:
    #
    # pip install -e .
    #
    # This allows us to keep modern Transformers/HF versions
    # without the old ltx-video package dependency metadata.
    # --------------------------------------------------------

    if str(LTX_REPO) not in sys.path:

        sys.path.insert(
            0,
            str(LTX_REPO),
        )

    # --------------------------------------------------------
    # Import directly from repository
    # --------------------------------------------------------

    try:

        import ltx_video

        ltx_file = Path(
            inspect.getfile(ltx_video)
        ).resolve()

    except Exception as exc:

        print(
            "\n❌ Failed to import LTX directly "
            "from source."
        )

        print(
            repr(exc)
        )

        sys.exit(1)

    expected_path = (
        LTX_REPO / "ltx_video"
    ).resolve()

    print(
        "LTX imported from:"
    )

    print(
        ltx_file
    )

    # --------------------------------------------------------
    # Make absolutely sure we didn't accidentally import
    # an old pip installation.
    # --------------------------------------------------------

    try:

        ltx_file.relative_to(
            expected_path
        )

    except ValueError:

        print(
            "\n❌ WRONG LTX IMPORT LOCATION"
        )

        print(
            f"Expected under:"
        )

        print(
            expected_path
        )

        print(
            f"Actually imported from:"
        )

        print(
            ltx_file
        )

        sys.exit(1)

    print(
        "\n✅ LTX is being loaded directly "
        "from the 0.9.8 repository"
    )

    print(
        "✅ No LTX pip package is required"
    )


# ============================================================
# REMOVE OLD LTX PIP PACKAGE
# ============================================================

def remove_old_ltx_package():

    print("\n" + "=" * 60)
    print("LTX PIP PACKAGE CHECK")
    print("=" * 60)

    # --------------------------------------------------------
    # Check whether the old pip package is installed.
    # --------------------------------------------------------

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
        "⚠️ Separate ltx-video pip package detected."
    )

    print(
        "Removing it because this project uses "
        "the checked-out LTX source directly."
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

    # --------------------------------------------------------
    # Re-add LTX source because pip operations can affect
    # Python's environment.
    # --------------------------------------------------------

    if str(LTX_REPO) not in sys.path:

        sys.path.insert(
            0,
            str(LTX_REPO),
        )


# ============================================================
# VERIFY LTX AFTER PACKAGE CLEANUP
# ============================================================

def verify_ltx_after_cleanup():

    print("\n" + "=" * 60)
    print("FINAL LTX VERIFICATION")
    print("=" * 60)

    if str(LTX_REPO) not in sys.path:

        sys.path.insert(
            0,
            str(LTX_REPO),
        )

    try:

        import ltx_video

        ltx_file = Path(
            inspect.getfile(ltx_video)
        ).resolve()

    except Exception as exc:

        print(
            "❌ LTX import failed after cleanup."
        )

        print(
            repr(exc)
        )

        sys.exit(1)

    expected_path = (
        LTX_REPO / "ltx_video"
    ).resolve()

    try:

        ltx_file.relative_to(
            expected_path
        )

    except ValueError:

        print(
            "❌ LTX is being imported from "
            "the wrong location."
        )

        print(
            f"Expected: {expected_path}"
        )

        print(
            f"Found: {ltx_file}"
        )

        sys.exit(1)

    print(
        "✅ LTX import successful"
    )

    print(
        f"Source: {ltx_file}"
    )


# ============================================================
# FINAL STATUS
# ============================================================

def print_final_status():

    print("\n" + "=" * 60)
    print("✅ STARTUP CHECK COMPLETE")
    print("=" * 60)

    print(
        f"LTX model      : {LTX_MODEL}"
    )

    print(
        f"LTX upscaler   : {LTX_UPSCALER}"
    )

    print(
        f"LTX repo       : {LTX_REPO}"
    )

    print(
        f"LTX revision   : {EXPECTED_LTX_COMMIT}"
    )

    print(
        f"LTX config     : {LTX_CONFIG}"
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
        "\nModern environment:"
    )

    import torch
    import transformers
    import huggingface_hub
    import diffusers

    print(
        f"  torch           : {torch.__version__}"
    )

    print(
        f"  transformers    : {transformers.__version__}"
    )

    print(
        f"  huggingface_hub : {huggingface_hub.__version__}"
    )

    print(
        f"  diffusers       : {diffusers.__version__}"
    )

    print(
        "\n🚀 LTX environment is ready."
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 60)
    print("🚀 AI VIDEO PROJECT STARTUP")
    print("=" * 60)

    # 1. GPU
    check_gpu()

    # 2. Python environment
    check_python_environment()

    # 3. Models
    check_models()

    # 4. Working directories
    setup_directories()

    # 5. LTX repository
    setup_ltx_repository()

    # 6. Apply T4 compatibility patch
    print("\n" + "=" * 60)
    print("APPLYING LTX T4 PATCH")
    print("=" * 60)

    run(
        [
            sys.executable,
            str(
                KAGGLE_DIR
                / "apply_ltx_t4_patch.py"
            ),
        ]
    )

    # 7. Direct LTX source path
    setup_ltx_source()

    # 8. Remove obsolete pip-installed LTX package
    remove_old_ltx_package()

    # 9. Verify direct source import again
    verify_ltx_after_cleanup()

    # 10. Final status
    print_final_status()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
