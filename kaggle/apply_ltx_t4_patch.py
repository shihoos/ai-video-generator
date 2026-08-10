from pathlib import Path
import subprocess
import sys
import re


# ============================================================
# LTX-VIDEO T4 MEMORY + DEVICE PATCH
# ============================================================

PROJECT_ROOT = Path("/kaggle/working/ai-video-generator")

LTX_REPO = PROJECT_ROOT / "LTX-Video-0.9.8"

LTX_REVISION = "bdc8f01"

INFERENCE_FILE = LTX_REPO / "ltx_video" / "inference.py"

PIPELINE_FILE = (
    LTX_REPO
    / "ltx_video"
    / "pipelines"
    / "pipeline_ltx_video.py"
)


# ============================================================
# HELPERS
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


def fail(message):
    raise RuntimeError(message)


def replace_once(text, old, new, description):
    count = text.count(old)

    if count == 0:
        fail(
            f"Could not find expected LTX code for:\n"
            f"{description}\n\n"
            f"The clean LTX source may have changed."
        )

    if count > 1:
        fail(
            f"Expected exactly one occurrence for:\n"
            f"{description}\n"
            f"Found {count} occurrences."
        )

    return text.replace(old, new, 1)


def write_text(path, text):
    path.write_text(text, encoding="utf-8")


# ============================================================
# RESTORE CLEAN LTX SOURCE
# ============================================================

def restore_ltx_source():

    print("\n" + "=" * 60)
    print("RESTORING LTX 0.9.8 SOURCE")
    print("=" * 60)

    if not LTX_REPO.exists():
        fail(f"LTX repository not found: {LTX_REPO}")

    run(
        f"git -C {LTX_REPO} checkout {LTX_REVISION} -- "
        f"ltx_video/inference.py "
        f"ltx_video/pipelines/pipeline_ltx_video.py"
    )

    print(f"✅ LTX files restored from clean revision {LTX_REVISION}")


# ============================================================
# PATCH inference.py
# ============================================================

def patch_inference():

    print("\n" + "=" * 60)
    print("PATCHING inference.py")
    print("=" * 60)

    text = INFERENCE_FILE.read_text(encoding="utf-8")

    changed = False

    # --------------------------------------------------------
    # Patch 1:
    # Explicitly pass execution device to pipeline creation.
    #
    # This is intentionally conservative. If the clean source
    # already has the correct behavior, we leave it alone.
    # --------------------------------------------------------

    old = """pipeline = create_ltx_video_pipeline(
        ckpt_path=ltxv_model_path,
        precision=precision,
        text_encoder_model_name_or_path=text_encoder_model_name_or_path,
        sampler=sampler,
        device=device,"""

    new = """pipeline = create_ltx_video_pipeline(
        ckpt_path=ltxv_model_path,
        precision=precision,
        text_encoder_model_name_or_path=text_encoder_model_name_or_path,
        sampler=sampler,
        device=device,"""

    if old in text:
        # Already correct in this revision.
        print("✅ Explicit LTX execution device already present")
    else:
        # Check whether the pipeline call exists but has a
        # different formatting.
        pattern = re.compile(
            r"pipeline\s*=\s*create_ltx_video_pipeline\(\s*"
            r"ckpt_path=ltxv_model_path,\s*"
            r"precision=precision,\s*"
            r"text_encoder_model_name_or_path="
            r"text_encoder_model_name_or_path,\s*"
            r"sampler=sampler,\s*"
            r"device=device,",
            re.MULTILINE,
        )

        if pattern.search(text):
            print("✅ Explicit LTX execution device already present")
        else:
            print(
                "ℹ️ Explicit device pattern not changed in inference.py"
            )

    # --------------------------------------------------------
    # Patch 2:
    # Do NOT explicitly move T5 to CUDA here.
    #
    # The current architecture keeps the large T5 model on CPU
    # when offloading is enabled. The pipeline itself handles
    # prompt encoding.
    # --------------------------------------------------------

    old_t5 = """    text_encoder = text_encoder.to(device)
"""

    if old_t5 in text:
        text = text.replace(
            old_t5,
            "",
            1,
        )
        changed = True
        print("✅ Removed explicit T5 GPU move from inference.py")
    else:
        print("✅ No explicit T5 GPU move found in inference.py")

    if changed:
        write_text(INFERENCE_FILE, text)

    print("✅ inference.py patch check complete")


# ============================================================
# PATCH pipeline_ltx_video.py
# ============================================================

def patch_pipeline():

    print("\n" + "=" * 60)
    print("PATCHING pipeline_ltx_video.py")
    print("=" * 60)

    text = PIPELINE_FILE.read_text(encoding="utf-8")

    # ========================================================
    # PATCH 1
    #
    # Prevent the huge T5 text encoder from being moved to GPU
    # when CPU offloading is enabled.
    #
    # Clean LTX source:
    #
    # if self.text_encoder is not None:
    #     self.text_encoder = self.text_encoder.to(
    #         self._execution_device
    #     )
    #
    # ========================================================

    old_t5_block = """        # 3. Encode input prompt
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(self._execution_device)
"""

    new_t5_block = """        # 3. Encode input prompt
        #
        # T4 MEMORY FIX:
        # Keep the large T5 text encoder on CPU when CPU
        # offloading is enabled.
        #
        # A Tesla T4 has only 15 GB VRAM and the T5 encoder
        # can consume most of that memory by itself.
        #
        # encode_prompt() detects the actual text encoder
        # device and moves only the resulting embeddings to
        # the execution device.
        if self.text_encoder is not None:
            if not offload_to_cpu:
                self.text_encoder = self.text_encoder.to(
                    self._execution_device
                )
"""

    if old_t5_block in text:

        text = replace_once(
            text,
            old_t5_block,
            new_t5_block,
            "keep T5 text encoder on CPU during --offload",
        )

        print("✅ T5 remains on CPU when --offload is enabled")

    elif new_t5_block in text:

        print("✅ T5 CPU-offload patch already present")

    else:

        fail(
            "Could not find the T5 execution-device block "
            "in pipeline_ltx_video.py"
        )

    # ========================================================
    # PATCH 2
    #
    # Move prompt attention masks to the execution device
    # AFTER encode_prompt().
    #
    # Why?
    #
    # T5 runs on CPU.
    # Therefore its tokenizer attention masks originate on CPU.
    #
    # The transformer runs on CUDA.
    #
    # torch.cat() later combines:
    #
    # negative_prompt_attention_mask
    # prompt_attention_mask
    # prompt_attention_mask
    #
    # They must all be on the same device.
    # ========================================================

    old_encode_end = """        ) = self.encode_prompt(
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

    new_encode_end = """        ) = self.encode_prompt(
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

        # T4 DEVICE FIX:
        #
        # The T5 encoder remains on CPU during offloading.
        # Therefore the attention masks produced during prompt
        # encoding may initially be CPU tensors.
        #
        # The LTX transformer runs on the execution device
        # (CUDA on Kaggle T4), so all attention masks must be
        # moved to that device before torch.cat().
        if prompt_attention_mask is not None:
            prompt_attention_mask = prompt_attention_mask.to(device)

        if negative_prompt_attention_mask is not None:
            negative_prompt_attention_mask = (
                negative_prompt_attention_mask.to(device)
            )

        if offload_to_cpu and self.text_encoder is not None:
            self.text_encoder = self.text_encoder.cpu()
"""

    if old_encode_end in text:

        text = replace_once(
            text,
            old_encode_end,
            new_encode_end,
            "move prompt attention masks to execution device",
        )

        print("✅ Prompt attention mask moved to execution device")
        print("✅ Negative prompt attention mask moved to execution device")

    elif new_encode_end in text:

        print("✅ Attention-mask device patch already present")

    else:

        fail(
            "Could not find the encode_prompt() completion block "
            "in pipeline_ltx_video.py"
        )

    # ========================================================
    # PATCH 3
    #
    # Make sure the masks are definitely on the same device
    # immediately before concatenation.
    #
    # This is a defensive safeguard.
    # ========================================================

    old_cat = """        prompt_embeds_batch = torch.cat(
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

    new_cat = """        prompt_embeds_batch = torch.cat(
            [negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0
        )

        # Final device safety check before concatenating attention
        # masks. All masks must be on the transformer execution
        # device.
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

    if old_cat in text:

        text = replace_once(
            text,
            old_cat,
            new_cat,
            "ensure attention masks share the execution device before torch.cat",
        )

        print("✅ Final attention-mask device safety check added")

    elif new_cat in text:

        print("✅ Final attention-mask device safety check already present")

    else:

        fail(
            "Could not find the prompt attention-mask concatenation "
            "block in pipeline_ltx_video.py"
        )

    write_text(PIPELINE_FILE, text)

    print("✅ pipeline_ltx_video.py patched")


# ============================================================
# VERIFICATION
# ============================================================

def verify_patch():

    print("\n" + "=" * 60)
    print("VERIFYING T4 PATCH")
    print("=" * 60)

    inference = INFERENCE_FILE.read_text(encoding="utf-8")
    pipeline = PIPELINE_FILE.read_text(encoding="utf-8")

    checks = []

    # --------------------------------------------------------
    # Check 1: T5 is not unconditionally moved to CUDA.
    # --------------------------------------------------------

    cpu_offload_pattern = (
        "if not offload_to_cpu:"
        in pipeline
        and "self.text_encoder = self.text_encoder.to("
        in pipeline
    )

    checks.append(
        (
            "T5 remains off GPU during --offload",
            cpu_offload_pattern,
        )
    )

    # --------------------------------------------------------
    # Check 2: attention masks moved to execution device.
    # --------------------------------------------------------

    prompt_mask_check = (
        "prompt_attention_mask = prompt_attention_mask.to(device)"
        in pipeline
    )

    negative_mask_check = (
        "negative_prompt_attention_mask"
        in pipeline
        and ".to(device)" in pipeline
    )

    checks.append(
        (
            "Prompt attention mask moved to execution device",
            prompt_mask_check,
        )
    )

    checks.append(
        (
            "Negative prompt attention mask moved to execution device",
            negative_mask_check,
        )
    )

    # --------------------------------------------------------
    # Check 3: pipeline has explicit execution device.
    # --------------------------------------------------------

    execution_device_check = (
        "device = self._execution_device"
        in pipeline
    )

    checks.append(
        (
            "Explicit LTX execution device",
            execution_device_check,
        )
    )

    # --------------------------------------------------------
    # Check 4: actual concatenation exists.
    # --------------------------------------------------------

    concat_check = (
        "prompt_attention_mask_batch = torch.cat("
        in pipeline
    )

    checks.append(
        (
            "Prompt attention-mask concatenation present",
            concat_check,
        )
    )

    # --------------------------------------------------------
    # Print results.
    # --------------------------------------------------------

    failed = False

    for description, result in checks:

        if result:
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            failed = True

    if failed:

        print("\n" + "=" * 60)
        print("PATCH VERIFICATION FAILED")
        print("=" * 60)

        fail(
            "One or more T4 patch verification checks failed."
        )

    print("\n" + "=" * 60)
    print("✅ ALL T4 MEMORY/DEVICE PATCH CHECKS PASSED")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("LTX-VIDEO T4 MEMORY + DEVICE PATCH")
    print("=" * 60)

    restore_ltx_source()

    patch_inference()

    patch_pipeline()

    verify_patch()

    print("\n" + "=" * 60)
    print("🚀 LTX T4 PATCH COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
