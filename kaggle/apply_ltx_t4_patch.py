from pathlib import Path


PROJECT_ROOT = Path("/kaggle/working/ai-video-generator")
LTX_REPO = PROJECT_ROOT / "LTX-Video-0.9.8"

INFERENCE_FILE = (
    LTX_REPO / "ltx_video" / "inference.py"
)

PIPELINE_FILE = (
    LTX_REPO
    / "ltx_video"
    / "pipelines"
    / "pipeline_ltx_video.py"
)


def replace_once(text, old, new, description):
    """
    Replace one exact block.

    Returns:
        (new_text, changed)
    """

    if new in text:
        print(f"✅ Already patched: {description}")
        return text, False

    if old not in text:
        raise RuntimeError(
            f"Could not find expected LTX code for: {description}"
        )

    print(f"🔧 Applying: {description}")

    return text.replace(old, new, 1), True


# ============================================================
# PATCH INFERENCE.PY
# ============================================================

def patch_inference():
    text = INFERENCE_FILE.read_text()

    changed = False

    # --------------------------------------------------------
    # Keep transformer + VAE on GPU.
    # Keep the large T5 text encoder on CPU.
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

    text, did_change = replace_once(
        text,
        old,
        new,
        "keep T5 text encoder on CPU",
    )

    changed = changed or did_change

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

    text, did_change = replace_once(
        text,
        old,
        new,
        "prevent pipeline.to(device)",
    )

    changed = changed or did_change

    # --------------------------------------------------------
    # Multi-scale upscaler should use the actual execution
    # device rather than pipeline.device.
    # --------------------------------------------------------

    old = """        latent_upsampler = create_latent_upsampler(
            spatial_upscaler_model_path, pipeline.device
        )
"""

    new = """        latent_upsampler = create_latent_upsampler(
            spatial_upscaler_model_path, device
        )
"""

    text, did_change = replace_once(
        text,
        old,
        new,
        "use explicit execution device for upscaler",
    )

    changed = changed or did_change

    INFERENCE_FILE.write_text(text)

    return changed


# ============================================================
# PATCH PIPELINE_LTX_VIDEO.PY
# ============================================================

def patch_pipeline():
    text = PIPELINE_FILE.read_text()

    changed = False

    # --------------------------------------------------------
    # Add explicit execution device property.
    #
    # Diffusers normally determines the device from modules.
    # Our T5 intentionally remains on CPU, so we explicitly
    # remember which device the LTX transformer should use.
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

    if "def _execution_device(self):" in text:
        print("✅ Already patched: explicit LTX execution device")
    else:
        if marker not in text:
            raise RuntimeError(
                "Could not find LTXVideoPipeline class."
            )

        print(
            "🔧 Applying: explicit LTX execution device"
        )

        text = text.replace(
            marker,
            property_block,
            1,
        )

        changed = True

    # --------------------------------------------------------
    # CRITICAL T4 FIX
    #
    # Original:
    #
    # if self.text_encoder is not None:
    #     self.text_encoder = self.text_encoder.to(
    #         self._execution_device
    #     )
    #
    # This moves the huge T5 encoder onto the 15 GB T4.
    #
    # New:
    #
    # only move it when CPU offloading is NOT requested.
    # With --offload, it remains on CPU.
    # --------------------------------------------------------

    old = """        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(self._execution_device)
"""

    new = """        if self.text_encoder is not None and not offload_to_cpu:
            self.text_encoder = self.text_encoder.to(
                self._execution_device
            )
"""

    text, did_change = replace_once(
        text,
        old,
        new,
        "keep T5 encoder on CPU during inference",
    )

    changed = changed or did_change

    PIPELINE_FILE.write_text(text)

    return changed


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("LTX-VIDEO T4 MEMORY PATCH")
    print("=" * 60)

    if not INFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {INFERENCE_FILE}"
        )

    if not PIPELINE_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {PIPELINE_FILE}"
        )

    inference_changed = patch_inference()
    pipeline_changed = patch_pipeline()

    print()

    if inference_changed or pipeline_changed:
        print("✅ LTX T4 memory patch applied")
    else:
        print("✅ LTX T4 memory patch already applied")

    print("=" * 60)


if __name__ == "__main__":
    main()
