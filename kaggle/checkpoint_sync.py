import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from video_config import CLIPS_DIR


# ============================================================
# CHECKPOINT SYNC
# ============================================================
#
# /kaggle/working is ephemeral: it is wiped when a session ends
# or restarts unless you explicitly commit it. Our shot-resume
# system (fingerprint + per-shot skip-if-exists) only survives
# an in-session crash — it does NOT survive closing the tab,
# hitting the weekly quota, or coming back next week.
#
# To actually resume across sessions, CLIPS_DIR must be synced
# to a persistent Kaggle Dataset:
#
#   pull  -> copy the Dataset's saved shots into CLIPS_DIR
#            (run this at the START of every session, before
#            generate_story_video.py)
#
#   push  -> copy CLIPS_DIR's current shots into the Dataset
#            (run this at the END of every session, or
#            periodically if you're worried about running out
#            of time mid-session)
#
# ONE-TIME SETUP (do this once, outside this script):
#   1. Create a new PRIVATE Kaggle Dataset from the Kaggle UI.
#      It can start empty — upload any placeholder file.
#      Note its slug, e.g. "yourusername/ai-video-checkpoints".
#   2. In your Kaggle notebook, go to Add-ons -> Kaggle API
#      credentials, or attach a Kaggle Secret containing your
#      kaggle.json, so the `kaggle` CLI is authenticated.
#      (If you're running this inside a Kaggle Notebook, the
#      kaggle package is already installed.)
#   3. Attach that dataset to your notebook as input data, the
#      same way you attached the Qwen model dataset. It will
#      then be readable at:
#      /kaggle/input/<your-dataset-slug>
# ============================================================


def run(command, check=True):
    print()
    print("$ " + " ".join(str(part) for part in command))

    result = subprocess.run(command)

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            + " ".join(str(part) for part in command)
        )

    return result.returncode


def pull(dataset_slug):
    """Copy previously saved shots from the Kaggle Dataset into
    the working CLIPS_DIR, so generate_story_video.py can resume."""

    dataset_dir_name = dataset_slug.split("/")[-1]
    source = Path("/kaggle/input") / dataset_dir_name

    if not source.exists():
        print()
        print(
            f"⚠️ Checkpoint dataset not found at {source}."
        )
        print(
            "Make sure it's attached to this notebook as input "
            "data (Add-ons -> Add Data)."
        )
        print(
            "Starting with an empty CLIPS_DIR — this is expected "
            "on your very first run."
        )
        return

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0

    for item in source.rglob("*"):
        if item.is_file():
            relative = item.relative_to(source)
            destination = CLIPS_DIR / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
            copied += 1

    print()
    print(f"✅ Pulled {copied} checkpoint file(s) into {CLIPS_DIR}")


def push(dataset_slug, message):
    """Push the current CLIPS_DIR (shots + fingerprint + story
    plan) to the Kaggle Dataset so the next session can resume."""

    if not CLIPS_DIR.exists() or not any(CLIPS_DIR.iterdir()):
        print()
        print(
            f"⚠️ {CLIPS_DIR} is empty or missing. Nothing to push."
        )
        return

    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "The 'kaggle' CLI is not available. Make sure the "
            "kaggle package is installed and your API "
            "credentials are configured."
        )

    # `kaggle datasets version` uploads the CURRENT contents of
    # the given folder as a new version of the dataset. It
    # requires a datapackage.json / dataset-metadata.json to
    # already exist for that dataset (created automatically the
    # first time you push through the Kaggle UI, or via
    # `kaggle datasets init`).
    metadata_file = CLIPS_DIR / "dataset-metadata.json"

    if not metadata_file.exists():
        print()
        print(
            "No dataset-metadata.json found in CLIPS_DIR yet. "
            "Initializing it now."
        )

        run(
            [
                "kaggle",
                "datasets",
                "init",
                "-p",
                str(CLIPS_DIR),
            ]
        )

        print(
            "⚠️ Edit "
            f"{metadata_file} "
            "and set \"id\" to your dataset slug "
            f"(e.g. \"{dataset_slug}\"), then re-run this push."
        )

        return

    run(
        [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(CLIPS_DIR),
            "-m",
            message,
            "--dir-mode",
            "zip",
        ]
    )

    print()
    print(
        f"✅ Pushed current shots in {CLIPS_DIR} to "
        f"Kaggle Dataset '{dataset_slug}'"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sync story_clips checkpoint state with a persistent "
            "Kaggle Dataset, since /kaggle/working does not "
            "survive across sessions."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Your checkpoint dataset slug, "
            "e.g. yourusername/ai-video-checkpoints"
        ),
    )

    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull saved shots from the Dataset into CLIPS_DIR.",
    )

    parser.add_argument(
        "--push",
        action="store_true",
        help="Push current CLIPS_DIR shots to the Dataset.",
    )

    parser.add_argument(
        "--message",
        default="Checkpoint update",
        help="Version message to use when pushing.",
    )

    args = parser.parse_args()

    if not args.pull and not args.push:
        parser.error("Specify --pull, --push, or both.")

    if args.pull:
        pull(args.dataset)

    if args.push:
        push(args.dataset, args.message)


if __name__ == "__main__":
    main()
