from pathlib import Path


PROJECT_ROOT = Path("/kaggle/working/ai-video-generator")
LTX_REPO = PROJECT_ROOT / "LTX-Video-0.9.8"

INFERENCE_FILE = LTX_REPO / "ltx_video" / "inference.py"
PIPELINE_FILE = LTX_REPO / "ltx_video" / "pipelines" / "pipeline_ltx_video.py"


def replace_once(text, old, new, description):
    if new in text:
        return text, False

    if old not in text:
        raise RuntimeError(
            f"Could not find expected LTX code for: {description}"
        )

    return text.replace(old, new, 1), True


def patch_inference():
    text = INFERENCE_FILE.read_text()

    # --------------------------------------------------------
    # 1. Keep the huge T5 text encoder on CPU.
    # --------------------------------------------------------

    old = """    transformer = transformer.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
"""

    new = """    transformer = transformer.to(device)
    vae = vae.to(device)

    # T4 memory fix:
    # Keep the large T5 text encoder on CPU.
    # It will be used from CPU and its embeddings are moved to GPU.
    # Moving the full text encoder to a 15 GB T4 causes OOM.
"""

    text, changed = replace_once(
        text,
        old,
        new,
        "keep text encoder on CPU",
    )

    # --------------------------------------------------------
    # 2. Do NOT call pipeline.to(device).
    #
    # That would move the CPU text encoder back to GPU.
    # Transformer and VAE were already moved explicitly above.
    # --------------------------------------------------------

    old = """    pipeline = LTXVideoPipeline(**submodel_dict)
    pipeline = pipeline.to(device)
    return pipeline
"""

    new = """    pipeline = LTXVideoPipeline(**submodel_dict)

    # T4 memory fix:
    # Do not call pipeline.to(device), because that would move
    # the CPU-resident text encoder onto the GPU.
    pipeline._ltx_execution_device = torch.device(device)

    return pipeline
"""

    text, changed2 = replace_once(
        text,
        old,
        new,
        "prevent pipeline.to(device) from moving text encoder",
    )

    # --------------------------------------------------------
    # 3. Use the explicit execution device for the multi-scale
    # upscaler rather than pipeline.device.
    # --------------------------------------------------------

    old = """        latent_upsampler = create_latent_upsampler(
            spatial_upscaler_model_path, pipeline.device
        )
"""

    new = """        latent_upsampler = create_latent_upsampler(
            spatial_upscaler_model_path, device
        )
"""

    text, changed3 = replace_once(
        text,
        old,
        new,
        "use explicit GPU device for spatial upscaler",
    )

    INFERENCE_FILE.write_text(text)

    return changed or changed2 or changed3


def patch_pipeline():
    text = PIPELINE_FILE.read_text()

    # Add an explicit execution-device property to LTXVideoPipeline.
    old = """class LTXVideoPipeline(DiffusionPipeline):
    r\"\"\"
"""

    new = """class LTXVideoPipeline(DiffusionPipeline):
    @property
    def _execution_device(self):
        # T4 memory fix:
        # The text encoder intentionally stays on CPU, so the normal
        # Diffusers device detection cannot be used here.
        # The actual LTX execution device is stored explicitly.
        if hasattr(self, "_ltx_execution_device"):
            return self._ltx_execution_device

        return self.device

    r\"\"\"
"""

    text, changed = replace_once(
        text,
        old,
        new,
        "explicit LTX execution device",
    )

    # --------------------------------------------------------
    # Do not move the text encoder to GPU when CPU offload is
    # requested.
    # --------------------------------------------------------

    old = """        # 3. Encode input prompt
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(self._execution_device)
        (
"""

    new = """        # 3. Encode input prompt
        #
        # T4 memory fix:
        # When offload_to_cpu is enabled, keep the large T5 encoder
        # on CPU. encode_prompt() runs the encoder on its current
        # device and moves only the resulting embeddings to GPU.
        if self.text_encoder is not None and not offload_to_cpu:
            self.text_encoder = self.text_encoder.to(self._execution_device)
        (
"""

    text, changed2 = replace_once(
        text,
        old,
        new,
        "keep text encoder on CPU during prompt encoding",
    )

    # --------------------------------------------------------
    # If the encoder isn't using a CPU-offload hook, the original
    # code's explicit CPU move is still useful. With our direct
    # CPU placement it is harmless.
    # --------------------------------------------------------

    PIPELINE_FILE.write_text(text)

    return changed or changed2


def main():
    print("=" * 60)
    print("LTX-VIDEO T4 MEMORY PATCH")
    print("=" * 60)

    if not INFERENCE_FILE.exists():
        raise FileNotFoundError(INFERENCE_FILE)

    if not PIPELINE_FILE.exists():
        raise FileNotFoundError(PIPELINE_FILE)

    inference_changed = patch_inference()
    pipeline_changed = patch_pipeline()

    if inference_changed or pipeline_changed:
        print("✅ LTX T4 memory patch applied")
    else:
        print("✅ LTX T4 memory patch already applied")

    print("=" * 60)


if __name__ == "__main__":
    main()
