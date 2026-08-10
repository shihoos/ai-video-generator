from pathlib import Path
import re
import subprocess
import sys

PROJECT_ROOT = Path("/kaggle/working/ai-video-generator")
LTX_REPO = PROJECT_ROOT / "LTX-Video-0.9.8"
INFERENCE_FILE = LTX_REPO / "ltx_video" / "inference.py"
PIPELINE_FILE = LTX_REPO / "ltx_video" / "pipelines" / "pipeline_ltx_video.py"
REVISION = "bdc8f01"


def fail(message: str):
    raise RuntimeError(message)


def run(cmd):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Command failed: {cmd}")


def replace_once(text, old, new, description):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one occurrence for {description}, found {count}."
        )
    return text.replace(old, new, 1)


def restore():
    run(
        f"git -C {LTX_REPO} checkout {REVISION} -- "
        f"ltx_video/inference.py ltx_video/pipelines/pipeline_ltx_video.py"
    )


def patch_inference():
    text = INFERENCE_FILE.read_text(encoding="utf-8")

    # Keep the fixes already established for the T4:
    # - no whole-pipeline .to(cuda)
    # - no direct T5 .to(cuda)
    old = """    pipeline = LTXVideoPipeline(**submodel_dict)
    pipeline = pipeline.to(device)
    return pipeline
"""
    new = """    pipeline = LTXVideoPipeline(**submodel_dict)

    # T4: keep T5 on CPU; remember the actual generation device.
    pipeline._ltx_execution_device = torch.device(device)

    return pipeline
"""
    if old in text:
        text = replace_once(
            text, old, new, "whole-pipeline device transfer"
        )

    old_t5 = """    text_encoder = text_encoder.to(device)
"""
    if old_t5 in text:
        text = text.replace(old_t5, "", 1)

    # CRITICAL MULTI-SCALE FIX:
    # The clean LTX 0.9.8 inference code creates the latent
    # upsampler using `pipeline.device`. Because we deliberately
    # keep T5 on CPU, Diffusers can report pipeline.device as CPU.
    # The first-pass latents, however, are CUDA tensors.
    #
    # Therefore the upsampler must be created on the explicit
    # LTX generation device, not pipeline.device.
    old_upsampler = """        latent_upsampler = create_latent_upsampler(
            spatial_upscaler_model_path, pipeline.device
        )
"""
    new_upsampler = """        # T4 / multi-scale device fix:
        # `pipeline.device` can resolve to CPU because T5 is kept
        # on CPU for memory reasons. The first-pass latents are
        # generated on the explicit CUDA execution device.
        # Create the latent upsampler on that same device.
        latent_upsampler = create_latent_upsampler(
            spatial_upscaler_model_path, device
        )
"""
    if old_upsampler in text:
        text = replace_once(
            text,
            old_upsampler,
            new_upsampler,
            "multi-scale latent upsampler device",
        )
        print("✅ Latent upsampler now uses explicit generation device")
    elif new_upsampler not in text:
        raise RuntimeError(
            "Could not find the multi-scale latent upsampler creation block."
        )

    INFERENCE_FILE.write_text(text, encoding="utf-8")


def patch_pipeline():
    text = PIPELINE_FILE.read_text(encoding="utf-8")

    # T5 CPU offload.
    old_t5 = """        # 3. Encode input prompt
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(self._execution_device)
"""
    new_t5 = """        # 3. Encode input prompt
        if self.text_encoder is not None:
            if not offload_to_cpu:
                self.text_encoder = self.text_encoder.to(
                    self._execution_device
                )
"""
    if old_t5 in text:
        text = replace_once(text, old_t5, new_t5, "T5 CPU offload")

    # Resolve generation-local device without relying on Diffusers'
    # CPU offload storage device.
    pattern = re.compile(
        r"^(?P<i>[ \t]+)device = self\._execution_device\s*$",
        re.MULTILINE,
    )
    replacement = (
        r"\g<i># T4: use explicit generation device when available.\n"
        r'\g<i>device = getattr(self, "_ltx_execution_device", '
        r"self._execution_device)"
    )
    text, count = pattern.subn(replacement, text)

    if count == 0 and 'device = getattr(self, "_ltx_execution_device"' not in text:
        raise RuntimeError(
            "Could not find the LTX execution-device assignment."
        )

    # Transformer must follow the same generation device.
    old_transformer = (
        "        self.transformer = self.transformer.to(self._execution_device)\n"
    )
    new_transformer = (
        "        self.transformer = self.transformer.to(device)\n"
    )
    if old_transformer in text:
        text = replace_once(
            text,
            old_transformer,
            new_transformer,
            "transformer generation device",
        )

    # Prompt masks must match CUDA transformer tensors.
    old_masks = """        prompt_embeds_batch = torch.cat(
            [negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0
        )
        prompt_attention_mask_batch = torch.cat(
            [
                negative_prompt_attention_mask,
                prompt_attention_mask,
                prompt_attention_mask,
            ],
            dim=0,
        )
"""
    new_masks = """        prompt_embeds_batch = torch.cat(
            [negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0
        )

        prompt_attention_mask = prompt_attention_mask.to(device)
        negative_prompt_attention_mask = (
            negative_prompt_attention_mask.to(device)
        )

        prompt_attention_mask_batch = torch.cat(
            [
                negative_prompt_attention_mask,
                prompt_attention_mask,
                prompt_attention_mask,
            ],
            dim=0,
        )
"""
    if old_masks in text:
        text = replace_once(
            text,
            old_masks,
            new_masks,
            "attention-mask device alignment",
        )

    # Also align masks immediately after CPU T5 encoding, because the
    # clean code can create them on the T5/device path before concatenation.
    old_encode = """        ) = self.encode_prompt(
            prompt,
            True,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            text_encoder_max_tokens=text_encoder_max_tokens,
        )
        if offload_to_cpu and self.text_encoder is not None:
            self.text_encoder = self.text_encoder.cpu()
"""
    new_encode = """        ) = self.encode_prompt(
            prompt,
            True,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            text_encoder_max_tokens=text_encoder_max_tokens,
        )

        if prompt_attention_mask is not None:
            prompt_attention_mask = prompt_attention_mask.to(device)
        if negative_prompt_attention_mask is not None:
            negative_prompt_attention_mask = (
                negative_prompt_attention_mask.to(device)
            )

        if offload_to_cpu and self.text_encoder is not None:
            self.text_encoder = self.text_encoder.cpu()
"""
    if old_encode in text:
        text = replace_once(
            text,
            old_encode,
            new_encode,
            "post-T5 attention-mask device alignment",
        )

    # MULTI-SCALE FIX:
    # Make _upsample_latents self-healing even if an upstream change
    # or Diffusers device reporting puts the upsampler elsewhere.
    old_upsample = """    def _upsample_latents(
        self, latest_upsampler: LatentUpsampler, latents: torch.Tensor
    ):
        assert latents.device == latest_upsampler.device
        latents = un_normalize_latents(
            latents, self.vae, vae_per_channel_normalize=True
        )
        upsampled_latents = latest_upsampler(latents)
        upsampled_latents = normalize_latents(
            upsampled_latents, self.vae, vae_per_channel_normalize=True
        )
        return upsampled_latents
"""
    new_upsample = """    def _upsample_latents(
        self, latest_upsampler: LatentUpsampler, latents: torch.Tensor
    ):
        # T4 / multi-scale device fix:
        # The first-pass latents are produced on the real generation
        # device. The latent upsampler must execute on that same device.
        target_device = latents.device

        if latest_upsampler.device != target_device:
            latest_upsampler = latest_upsampler.to(target_device)
            latest_upsampler.eval()

        if latest_upsampler.device != target_device:
            raise RuntimeError(
                "Latent upsampler could not be moved to the latent device: "
                f"latents={target_device}, upsampler={latest_upsampler.device}"
            )

        latents = un_normalize_latents(
            latents, self.vae, vae_per_channel_normalize=True
        )
        upsampled_latents = latest_upsampler(latents)
        upsampled_latents = normalize_latents(
            upsampled_latents, self.vae, vae_per_channel_normalize=True
        )

        # Release the upsampler from GPU after the first-pass upsample.
        # The second diffusion pass uses the main LTX transformer instead.
        if target_device.type == "cuda":
            latest_upsampler = latest_upsampler.cpu()
            torch.cuda.empty_cache()

        return upsampled_latents
"""
    if old_upsample in text:
        text = replace_once(
            text,
            old_upsample,
            new_upsample,
            "multi-scale latent upsampler device alignment",
        )
        print("✅ Multi-scale upsampler now follows latent device")
    elif new_upsample not in text:
        raise RuntimeError(
            "Could not find LTXMultiScalePipeline._upsample_latents()."
        )

    PIPELINE_FILE.write_text(text, encoding="utf-8")


def verify():
    inference = INFERENCE_FILE.read_text(encoding="utf-8")
    pipeline = PIPELINE_FILE.read_text(encoding="utf-8")

    for path in (INFERENCE_FILE, PIPELINE_FILE):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            print(result.stderr)
            raise RuntimeError(f"Syntax check failed: {path}")

    checks = {
        "No whole-pipeline CUDA transfer": "pipeline = pipeline.to(device)" not in inference,
        "T5 direct CUDA transfer removed": "text_encoder = text_encoder.to(device)" not in inference,
        "Explicit LTX generation device": "_ltx_execution_device = torch.device(device)" in inference,
        "Upsampler created on explicit device": "spatial_upscaler_model_path, device" in inference,
        "Generation device override": 'device = getattr(self, "_ltx_execution_device"' in pipeline,
        "Transformer uses generation device": "self.transformer = self.transformer.to(device)" in pipeline,
        "Upsampler follows latent device": "target_device = latents.device" in pipeline,
        "Upsampler device assertion replaced safely": "could not be moved to the latent device" in pipeline,
        "Prompt mask device alignment": "prompt_attention_mask = prompt_attention_mask.to(device)" in pipeline,
        "Negative mask device alignment": "negative_prompt_attention_mask = (" in pipeline and ".to(device)" in pipeline,
    }

    failed = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(("✅ " if ok else "❌ ") + name)

    if failed:
        raise RuntimeError("Patch verification failed: " + ", ".join(failed))

    print("✅ Python syntax checks passed")
    print("✅ ALL T4 MULTI-SCALE DEVICE CHECKS PASSED")


def main():
    print("\n" + "=" * 60)
    print("LTX-VIDEO T4 FINAL MULTI-SCALE DEVICE PATCH")
    print("=" * 60)

    restore()
    patch_inference()
    patch_pipeline()
    verify()

    print("\n🚀 LTX T4 multi-scale patch complete.")


if __name__ == "__main__":
    main()
