import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path("/kaggle/working/ai-video-generator")
KAGGLE_DIR = PROJECT_ROOT / "kaggle"

if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))

from config import (
    LTX_MODEL,
    LTX_UPSCALER,
    LTX_CONFIG,
    OUTPUT_DIR,
)


def build_test_config():
    """
    Create a temporary inference configuration based on the
    official LTX-Video 0.9.8 distilled configuration.

    The only changes are:
      - use persistent Kaggle model paths
      - disable prompt enhancement for the initial test
    """

    if not LTX_CONFIG.exists():
        raise FileNotFoundError(
            f"LTX configuration not found: {LTX_CONFIG}"
        )

    text = LTX_CONFIG.read_text()

    text = text.replace(
        'checkpoint_path: "ltxv-2b-0.9.8-distilled.safetensors"',
        f'checkpoint_path: "{LTX_MODEL}"',
    )

    text = text.replace(
        'spatial_upscaler_model_path: "ltxv-spatial-upscaler-0.9.8.safetensors"',
        f'spatial_upscaler_model_path: "{LTX_UPSCALER}"',
    )

    # Disable automatic Florence/Llama prompt enhancement
    text = text.replace(
        "prompt_enhancement_words_threshold: 120",
        "prompt_enhancement_words_threshold: 0",
    )

    test_config = PROJECT_ROOT / "work" / "ltx-inference.yaml"
    test_config.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_config.write_text(text)

    return test_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate video with LTX-Video 0.9.8"
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt for video generation",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Output width",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Output height",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=9,
        help="Number of frames",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Output frame rate",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=171198,
        help="Random seed",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output directory",
    )

    parser.add_argument(
        "--offload",
        action="store_true",
        help="Enable CPU offloading",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Verify models
    # --------------------------------------------------------

    if not LTX_MODEL.exists():
        raise FileNotFoundError(
            f"LTX model not found: {LTX_MODEL}"
        )

    if not LTX_UPSCALER.exists():
        raise FileNotFoundError(
            f"LTX spatial upscaler not found: {LTX_UPSCALER}"
        )

    # --------------------------------------------------------
    # Build temporary configuration
    # --------------------------------------------------------

    pipeline_config = build_test_config()

    # --------------------------------------------------------
    # Import official LTX inference
    # --------------------------------------------------------

    from ltx_video.inference import (
        infer,
        InferenceConfig,
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_path = (
        Path(args.output)
        if args.output
        else OUTPUT_DIR
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Build official LTX configuration
    # --------------------------------------------------------

    config = InferenceConfig(
        prompt=args.prompt,
        output_path=str(output_path),
        pipeline_config=str(pipeline_config),
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        frame_rate=args.fps,
        offload_to_cpu=args.offload,
    )

    print("\n" + "=" * 60)
    print("🎬 LTX VIDEO GENERATION")
    print("=" * 60)

    print(f"Prompt       : {args.prompt}")
    print(f"Resolution   : {args.width}x{args.height}")
    print(f"Frames       : {args.frames}")
    print(f"FPS          : {args.fps}")
    print(f"Seed         : {args.seed}")
    print(f"Model        : {LTX_MODEL}")
    print(f"Upscaler     : {LTX_UPSCALER}")
    print(f"Config       : {pipeline_config}")
    print(f"Output       : {output_path}")
    print(f"CPU offload  : {args.offload}")

    print("\nStarting LTX inference...\n")

    infer(config=config)

    print("\n" + "=" * 60)
    print("✅ GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_path}")


if __name__ == "__main__":
    main()
