from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path("/kaggle/working/ai-video-generator")
LTX_REPO = PROJECT_ROOT / "LTX-Video-0.9.8"

LTX_COMMIT = "bdc8f01"

INFERENCE_FILE = (
    LTX_REPO / "ltx_video" / "inference.py"
)

PIPELINE_FILE = (
    LTX_REPO
    / "ltx_video"
    / "pipelines"
    / "pipeline_ltx_video.py"
)


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
# RESTORE EXACT LTX 0.9.8 SOURCE
# ============================================================

def restore_ltx_files():
    print("\n" + "=" * 60)
    print("RESTORING LTX 0.9.8 SOURCE")
    print("=" * 60)

    if not LTX_REPO.exists():
        raise FileNotFoundError(
            f"LTX repository not found: {LTX_REPO}"
        )

    # Restore only the two files that our runtime patch modifies.
    # This removes any incomplete patch from a previous Kaggle run.
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
    # Keep transformer + VAE on GPU.
    # Keep the huge T5 text encoder on CPU.
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
            "Expected original text encoder placement code "
            "was not found in inference.py"
        )

    text = text.replace(old, new, 1)

    # --------------------------------------------------------
    # Prevent pipeline.to(device) from moving T5 to GPU.
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

    text = text.replace(old, new, 1)

    # --------------------------------------------------------
    # Multi-scale upscaler uses explicit execution device.
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

    text = text.replace(old, new, 1)

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
    # Add explicit execution device.
    # --------------------------------------------------------

    marker = """class LTXVideoPipeline(DiffusionPipeline):
"""

    property_block = """class LTXVideoPipeline(DiffusionPipeline):

    @property
    def _execution_device(self):
        # T4 memory optimization:
        # The T5 text encoder intentionally remains on CPU.
        # Therefore the normal Diffusers device detection is
        # not suitable for this pipeline.
        if hasattr(self, "_ltx_execution_device"):
            return self._ltx_execution_device

        return self.device

"""

    if marker not in text:
        raise RuntimeError(
            "LTXVideoPipeline class was not found."
        )

    text = text.replace(
        marker,
        property_block,
        1,
    )

    # --------------------------------------------------------
    # CRITICAL MEMORY FIX
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
    # When --offload is used, leave T5 on CPU.
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

    text = text.replace(old, new, 1)

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
            "if self.text_encoder is not None and not offload_to_cpu:"
            in pipeline_text,
        ),
        (
            "Explicit LTX execution device",
            "pipeline._ltx_execution_device = torch.device(device)"
            in inference_text,
        ),
        (
            "T5 no longer explicitly moved to GPU in inference",
            "text_encoder = text_encoder.to(device)"
            not in inference_text,
        ),
        (
            "Upscaler uses explicit device",
            "spatial_upscaler_model_path, device"
            in inference_text,
        ),
    ]

    for description, result in checks:
        if result:
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            sys.exit(1)

    print("\n✅ All T4 memory patch checks passed")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("LTX-VIDEO T4 MEMORY PATCH")
    print("=" * 60)

    restore_ltx_files()
    patch_inference()
    patch_pipeline()
    verify_patch()

    print("\n" + "=" * 60)
    print("✅ LTX T4 MEMORY PATCH COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
