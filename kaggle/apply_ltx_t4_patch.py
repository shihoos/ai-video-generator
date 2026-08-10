from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path("/kaggle/working/ai-video-generator")
LTX_REPO = PROJECT_ROOT / "LTX-Video-0.9.8"

LTX_COMMIT = "bdc8f01"

INFERENCE_FILE = (
    LTX_REPO
    / "ltx_video"
    / "inference.py"
)

PIPELINE_FILE = (
    LTX_REPO
    / "ltx_video"
    / "pipelines"
    / "pipeline_ltx_video.py"
)


# ============================================================
# COMMAND HELPER
# ============================================================

def run(command):
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
# RESTORE CLEAN LTX 0.9.8 SOURCE
# ============================================================

def restore_ltx_files():
    print("\n" + "=" * 60)
    print("RESTORING LTX 0.9.8 SOURCE")
    print("=" * 60)

    if not LTX_REPO.exists():
        raise FileNotFoundError(
            f"LTX repository not found: {LTX_REPO}"
        )

    # Always restore the two files that we modify.
    #
    # This guarantees that a previous failed/partial patch
    # cannot interfere with the current patch.
    run(
        f"git -C {LTX_REPO} checkout {LTX_COMMIT} -- "
        f"ltx_video/inference.py "
        f"ltx_video/pipelines/pipeline_ltx_video.py"
    )

    print(
        f"✅ LTX files restored from clean revision {LTX_COMMIT}"
    )


# ============================================================
# PATCH inference.py
# ============================================================

def patch_inference():
    print("\n" + "=" * 60)
    print("PATCHING inference.py")
    print("=" * 60)

    text = INFERENCE_FILE.read_text()

    # --------------------------------------------------------
    # FIX 1
    #
    # Keep the huge T5 text encoder on CPU.
    #
    # Original:
    #
    # transformer = transformer.to(device)
    # vae = vae.to(device)
    # text_encoder = text_encoder.to(device)
    #
    # The last line causes the 15 GB T4 OOM.
    # --------------------------------------------------------

    old = """    transformer = transformer.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
"""

    new = """    transformer = transformer.to(device)
    vae = vae.to(device)

    # T4 memory optimization:
    # Keep the large T5 text encoder on CPU.
    # Only its generated embeddings need to reach the GPU.
"""

    if old not in text:
        raise RuntimeError(
            "Expected text encoder placement code "
            "was not found in inference.py"
        )

    text = text.replace(
        old,
        new,
        1,
    )

    # --------------------------------------------------------
    # FIX 2
    #
    # Prevent pipeline.to(device) from moving the T5 encoder
    # onto the GPU.
    # --------------------------------------------------------

    old = """    pipeline = LTXVideoPipeline(**submodel_dict)
    pipeline = pipeline.to(device)
    return pipeline
"""

    new = """    pipeline = LTXVideoPipeline(**submodel_dict)

    # T4 memory optimization:
    # Do NOT call pipeline.to(device).
    # That would move the large T5 text encoder to GPU.
    pipeline._ltx_execution_device = torch.device(device)

    return pipeline
"""

    if old not in text:
        raise RuntimeError(
            "Expected pipeline.to(device) code "
            "was not found in inference.py"
        )

    text = text.replace(
        old,
        new,
        1,
    )

    # --------------------------------------------------------
    # FIX 3
    #
    # The multi-scale spatial upscaler should use the explicit
    # execution device.
    # --------------------------------------------------------

    old = """        latent_upsampler = create_latent_upsampler(
            spatial_upscaler_model_path, pipeline.device
        )
"""

    new = """        latent_upsampler = create_latent_upsampler(
            spatial_upscaler_model_path, device
        )
"""

    if old not in text:
        raise RuntimeError(
            "Expected spatial upscaler device code "
            "was not found in inference.py"
        )

    text = text.replace(
        old,
        new,
        1,
    )

    INFERENCE_FILE.write_text(text)

    print("✅ inference.py patched")


# ============================================================
# PATCH pipeline_ltx_video.py
# ============================================================

def patch_pipeline():
    print("\n" + "=" * 60)
    print("PATCHING pipeline_ltx_video.py")
    print("=" * 60)

    text = PIPELINE_FILE.read_text()

    # --------------------------------------------------------
    # FIX 4
    #
    # Add an explicit execution-device property.
    #
    # Because the T5 encoder remains on CPU, normal Diffusers
    # device detection can no longer be relied upon.
    # --------------------------------------------------------

    marker = """class LTXVideoPipeline(DiffusionPipeline):
"""

    property_block = """class LTXVideoPipeline(DiffusionPipeline):

    @property
    def _execution_device(self):
        # T4 memory optimization:
        # The T5 text encoder intentionally remains on CPU.
        # Therefore normal Diffusers device detection is not
        # suitable for this pipeline.
        if hasattr(self, "_ltx_execution_device"):
            return self._ltx_execution_device

        return self.device

"""

    if "def _execution_device(self):" in text:
        print(
            "✅ Explicit LTX execution device already present"
        )
    else:
        if marker not in text:
            raise RuntimeError(
                "LTXVideoPipeline class was not found."
            )

        text = text.replace(
            marker,
            property_block,
            1,
        )

        print(
            "✅ Explicit LTX execution device added"
        )

    # --------------------------------------------------------
    # FIX 5
    #
    # CRITICAL T4 MEMORY FIX
    #
    # Original:
    #
    # if self.text_encoder is not None:
    #     self.text_encoder = self.text_encoder.to(
    #         self._execution_device
    #     )
    #
    # This moves the huge T5 encoder onto the T4.
    #
    # New:
    #
    # When --offload is enabled, leave T5 on CPU.
    # --------------------------------------------------------

    old = """        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(self._execution_device)
"""

    new = """        if self.text_encoder is not None and not offload_to_cpu:
            self.text_encoder = self.text_encoder.to(
                self._execution_device
            )
"""

    if old not in text:
        raise RuntimeError(
            "Expected text encoder GPU transfer code "
            "was not found in pipeline_ltx_video.py"
        )

    text = text.replace(
        old,
        new,
        1,
    )

    print(
        "✅ T5 remains on CPU when --offload is enabled"
    )

    # --------------------------------------------------------
    # FIX 6
    #
    # ATTENTION MASK DEVICE FIX
    #
    # With the T5 encoder on CPU, the embeddings and attention
    # masks can end up on different devices.
    #
    # Later LTX performs torch.cat() on the masks.
    #
    # Therefore both masks must be explicitly moved to the
    # LTX execution device.
    # --------------------------------------------------------

    old = """        negative_prompt_attention_mask = negative_prompt_attention_mask.view(
            bs_embed * num_images_per_prompt, -1
        )
    else:
        negative_prompt_embeds = None
        negative_prompt_attention_mask = None

    return (
"""

    new = """        negative_prompt_attention_mask = negative_prompt_attention_mask.view(
            bs_embed * num_images_per_prompt, -1
        )

        # T4 memory/offload compatibility:
        # The T5 encoder remains on CPU, but the resulting
        # attention masks must be on the same device as the
        # LTX prompt embeddings.
        negative_prompt_attention_mask = (
            negative_prompt_attention_mask.to(device)
        )

    else:
        negative_prompt_embeds = None
        negative_prompt_attention_mask = None

    # Ensure prompt attention masks are on the LTX execution
    # device before they are concatenated later in the pipeline.
    if prompt_attention_mask is not None:
        prompt_attention_mask = prompt_attention_mask.to(device)

    if negative_prompt_attention_mask is not None:
        negative_prompt_attention_mask = (
            negative_prompt_attention_mask.to(device)
        )

    return (
"""

    if old not in text:
        raise RuntimeError(
            "Expected negative prompt attention-mask block "
            "was not found in pipeline_ltx_video.py"
        )

    text = text.replace(
        old,
        new,
        1,
    )

    print(
        "✅ Prompt attention masks moved to execution device"
    )

    PIPELINE_FILE.write_text(text)

    print("✅ pipeline_ltx_video.py patched")


# ============================================================
# VERIFY PATCH
# ============================================================

def verify_patch():
    print("\n" + "=" * 60)
    print("VERIFYING T4 PATCH")
    print("=" * 60)

    inference_text = INFERENCE_FILE.read_text()
    pipeline_text = PIPELINE_FILE.read_text()

    checks = [
        (
            "T5 remains off GPU during --offload",
            (
                "if self.text_encoder is not None and not offload_to_cpu:"
                in pipeline_text
            ),
        ),
        (
            "Explicit LTX execution device",
            (
                "pipeline._ltx_execution_device = torch.device(device)"
                in inference_text
            ),
        ),
        (
            "T5 no longer explicitly moved to GPU in inference.py",
            (
                "text_encoder = text_encoder.to(device)"
                not in inference_text
            ),
        ),
        (
            "Upscaler uses explicit execution device",
            (
                "spatial_upscaler_model_path, device"
                in inference_text
            ),
        ),
        (
            "Prompt attention mask moved to execution device",
            (
                "prompt_attention_mask = prompt_attention_mask.to(device)"
                in pipeline_text
            ),
        ),
        (
            "Negative attention mask moved to execution device",
            (
                "negative_prompt_attention_mask = ("
                in pipeline_text
            ),
        ),
    ]

    all_passed = True

    for description, result in checks:
        if result:
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            all_passed = False

    if not all_passed:
        print(
            "\n❌ One or more LTX T4 patch checks failed."
        )
        sys.exit(1)

    print(
        "\n✅ All T4 memory/device patch checks passed"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("LTX-VIDEO T4 MEMORY + DEVICE PATCH")
    print("=" * 60)

    if not LTX_REPO.exists():
        raise FileNotFoundError(
            f"LTX repository not found: {LTX_REPO}"
        )

    if not INFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"LTX inference file not found: {INFERENCE_FILE}"
        )

    if not PIPELINE_FILE.exists():
        raise FileNotFoundError(
            f"LTX pipeline file not found: {PIPELINE_FILE}"
        )

    # Always start from clean bdc8f01 source.
    restore_ltx_files()

    # Apply all fixes.
    patch_inference()
    patch_pipeline()

    # Verify all fixes.
    verify_patch()

    print("\n" + "=" * 60)
    print("✅ LTX T4 PATCH COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
