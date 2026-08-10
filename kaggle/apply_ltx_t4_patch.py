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

    # ========================================================
    # FIX 1: Never call pipeline.to(device)
    #
    # Diffusers pipeline.to(device) moves EVERY registered
    # component, including the large T5 encoder, to CUDA.
    # On a 15 GB Tesla T4 this causes OOM.
    #
    # The transformer and VAE are moved explicitly by the
    # pipeline-construction code; T5 remains on CPU.
    # ========================================================

    old_pipeline_to = """    pipeline = LTXVideoPipeline(**submodel_dict)
    pipeline = pipeline.to(device)
    return pipeline
"""

    new_pipeline_to = """    pipeline = LTXVideoPipeline(**submodel_dict)

    # T4 MEMORY FIX:
    #
    # Do NOT call pipeline.to(device).
    # Diffusers would move every registered component to CUDA,
    # including the large T5 text encoder.
    #
    # The transformer and VAE are placed explicitly, while T5
    # remains on CPU for CPU-offload operation.
    pipeline._ltx_execution_device = torch.device(device)

    return pipeline
"""

    if old_pipeline_to in text:
        text = text.replace(old_pipeline_to, new_pipeline_to, 1)
        print("✅ Removed pipeline.to(device) to prevent T5 GPU OOM")
    elif new_pipeline_to in text:
        print("✅ pipeline.to(device) T4 fix already present")
    else:
        raise RuntimeError(
            "Could not find the LTX pipeline construction block "
            "containing pipeline.to(device)."
        )

    # ========================================================
    # FIX 2: Remove any direct T5 GPU transfer in inference.py
    # ========================================================

    old_t5 = """    text_encoder = text_encoder.to(device)
"""

    if old_t5 in text:
        text = text.replace(old_t5, "", 1)
        print("✅ Removed explicit T5 GPU move from inference.py")
    else:
        print("✅ No explicit T5 GPU move found in inference.py")

    # ========================================================
    # SAFETY CHECKS
    # ========================================================

    if "pipeline = pipeline.to(device)" in text:
        raise RuntimeError(
            "pipeline.to(device) is still present in inference.py"
        )

    if "text_encoder = text_encoder.to(device)" in text:
        raise RuntimeError(
            "text_encoder.to(device) is still present in inference.py"
        )

    if "pipeline._ltx_execution_device = torch.device(device)" not in text:
        raise RuntimeError(
            "Explicit LTX execution device was not configured."
        )

    INFERENCE_FILE.write_text(text, encoding="utf-8")

    print("✅ inference.py patched")


# ============================================================
# PATCH pipeline_ltx_video.py
# ============================================================

def patch_pipeline():

    print("\n" + "=" * 60)
    print("PATCHING pipeline_ltx_video.py")
    print("=" * 60)

    text = PIPELINE_FILE.read_text(encoding="utf-8")

    # ========================================================
    # FIX 1: Keep T5 on CPU during offload
    # ========================================================

    old_t5_block = """        # 3. Encode input prompt
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(self._execution_device)
"""

    new_t5_block = """        # 3. Encode input prompt
        #
        # T4 MEMORY FIX:
        # Keep the large T5 text encoder on CPU when CPU
        # offloading is enabled. Moving it to a 15 GB T4 causes
        # CUDA out-of-memory errors.
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
        raise RuntimeError(
            "Could not find the T5 execution-device block "
            "in pipeline_ltx_video.py"
        )

    # ========================================================
    # FIX 2: FORCE THE ACTUAL LTX GENERATION DEVICE
    #
    # IMPORTANT:
    # Diffusers' self._execution_device becomes CPU when CPU
    # offload is active. LTX 0.9.8 then does:
    #
    #     device = self._execution_device
    #
    # and passes that device to prepare_latents(), while
    # generate.py supplies a CUDA generator.
    #
    # That produces:
    #
    #     Cannot generate a cpu tensor from a generator of type cuda
    #
    # We patch the assignment itself. The replacement preserves
    # the original indentation, so nested methods/blocks remain
    # syntactically valid.
    # ========================================================

    execution_pattern = re.compile(
        r"^(?P<indent>[ \t]+)device = self\._execution_device\s*$",
        re.MULTILINE,
    )

    execution_replacement = (
        r'\g<indent># LTX T4: use the real generation device, not '
        r'Diffusers offload storage device.\n'
        r'\g<indent>device = getattr(\n'
        r'\g<indent>    self,\n'
        r'\g<indent>    "_ltx_execution_device",\n'
        r'\g<indent>    self._execution_device,\n'
        r'\g<indent>)'
    )

    text, execution_count = execution_pattern.subn(
        execution_replacement,
        text,
    )

    if execution_count == 0:
        # If a previous compatible patch is already present, leave it.
        if 'device = getattr(\n            self,\n            "_ltx_execution_device",' in text:
            print("✅ LTX generation-device patch already present")
        else:
            raise RuntimeError(
                "Could not find the LTX `device = self._execution_device` "
                "assignment in pipeline_ltx_video.py."
            )
    else:
        print(
            f"✅ Patched {execution_count} LTX generation-device assignment(s)"
        )

    # ========================================================
    # FIX 2B: TRANSFORMER MUST FOLLOW THE SAME GENERATION DEVICE
    #
    # The clean LTX code moves the transformer to
    # self._execution_device. Under CPU offload that can be CPU,
    # even though generation tensors are CUDA.
    #
    # Use the already-resolved local `device` instead.
    # ========================================================

    old_transformer_device = (
        "        self.transformer = self.transformer.to(self._execution_device)\n"
    )

    new_transformer_device = (
        "        self.transformer = self.transformer.to(device)\n"
    )

    if old_transformer_device in text:
        text = replace_once(
            text,
            old_transformer_device,
            new_transformer_device,
            "move transformer to the actual LTX generation device",
        )
        print("✅ Transformer now follows the explicit generation device")
    elif new_transformer_device in text:
        print("✅ Transformer generation-device patch already present")
    else:
        raise RuntimeError(
            "Could not find the LTX transformer device assignment."
        )

    # ========================================================
    # FIX 3: Attention masks
    #
    # T5 is on CPU, so masks can originate on CPU.
    # Transformer/generation is on CUDA.
    # Move both masks before concatenation.
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
        # T5 remains on CPU during offload, so attention masks
        # can initially be CPU tensors. The transformer runs on
        # the execution device, therefore both masks must be on
        # that device before concatenation.
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
        raise RuntimeError(
            "Could not find the encode_prompt() completion block "
            "in pipeline_ltx_video.py"
        )

    # ========================================================
    # FIX 4: Final safety before torch.cat()
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

        # Final device safety check before concatenating masks.
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
        raise RuntimeError(
            "Could not find the prompt attention-mask concatenation "
            "block in pipeline_ltx_video.py"
        )

    # ========================================================
    # FIX 5: Generator/device consistency
    #
    # The caller creates a CUDA generator. prepare_latents()
    # must therefore receive CUDA as its device.
    #
    # The clean LTX code already passes `device` to
    # prepare_latents(). We verify that rather than modifying
    # the upstream implementation unnecessarily.
    # ========================================================

    prepare_latents_pattern = re.compile(
        r"self\.prepare_latents\([\s\S]{0,3000}?"
        r"device=device,[\s\S]{0,3000}?"
        r"generator=generator,",
        re.MULTILINE,
    )

    if prepare_latents_pattern.search(text):
        print("✅ prepare_latents uses the local execution device")
        print("✅ prepare_latents uses the generation generator")
    else:
        raise RuntimeError(
            "Could not verify prepare_latents device/generator handling."
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

    # ========================================================
    # Syntax check FIRST.
    #
    # This prevents a broken patch from reaching generation.
    # ========================================================

    compile_cmd = (
        f"{sys.executable} -m py_compile "
        f"{PIPELINE_FILE} {INFERENCE_FILE}"
    )

    result = subprocess.run(
        compile_cmd,
        shell=True,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(
            "Patched LTX Python source failed py_compile. "
            "Generation will not be started."
        )

    print("✅ Patched LTX source passes Python syntax check")

    # ========================================================
    # T5 CPU-offload check.
    # ========================================================

    t5_cpu_check = (
        "if not offload_to_cpu:"
        in pipeline
        and "self.text_encoder = self.text_encoder.to("
        in pipeline
    )

    checks.append(
        (
            "T5 remains off GPU during --offload",
            t5_cpu_check,
        )
    )

    # ========================================================
    # pipeline.to(device) must not move the whole pipeline.
    # ========================================================

    checks.append(
        (
            "pipeline.to(device) removed from inference",
            "pipeline = pipeline.to(device)" not in inference,
        )
    )

    # ========================================================
    # Direct T5 transfer must not remain.
    # ========================================================

    checks.append(
        (
            "Direct T5 GPU transfer removed from inference",
            "text_encoder = text_encoder.to(device)" not in inference,
        )
    )

    # ========================================================
    # Explicit execution device configured.
    # ========================================================

    checks.append(
        (
            "Explicit LTX execution device configured",
            "pipeline._ltx_execution_device = torch.device(device)"
            in inference,
        )
    )

    # ========================================================
    # Actual generation assignment must use the explicit device.
    # ========================================================

    generation_device_check = (
        'device = getattr(\n'
        '            self,\n'
        '            "_ltx_execution_device",'
        in pipeline
    )

    checks.append(
        (
            "LTX generation device overrides Diffusers offload device",
            generation_device_check,
        )
    )

    # ========================================================
    # No raw execution-device assignment should remain.
    #
    # This is intentionally checked after patching. A clean
    # source can contain the assignment; the patched source must
    # not use it for the generation-local `device`.
    # ========================================================

    checks.append(
        (
            "No unpatched Diffusers execution-device assignment remains",
            "device = self._execution_device" not in pipeline,
        )
    )

    # ========================================================
    # Transformer follows local generation device.
    # ========================================================

    checks.append(
        (
            "Transformer uses the local generation device",
            "self.transformer = self.transformer.to(device)"
            in pipeline,
        )
    )

    # ========================================================
    # Attention masks.
    # ========================================================

    checks.append(
        (
            "Prompt attention mask moved to generation device",
            "prompt_attention_mask = prompt_attention_mask.to(device)"
            in pipeline,
        )
    )

    checks.append(
        (
            "Negative attention mask moved to generation device",
            "negative_prompt_attention_mask = ("
            in pipeline
            and ".to(device)" in pipeline,
        )
    )

    checks.append(
        (
            "Attention-mask concatenation present",
            "prompt_attention_mask_batch = torch.cat("
            in pipeline,
        )
    )

    # ========================================================
    # prepare_latents() must receive both device and generator.
    # ========================================================

    prepare_latents_check = bool(
        re.search(
            r"self\.prepare_latents\([\s\S]{0,3000}?"
            r"device=device,[\s\S]{0,3000}?"
            r"generator=generator,",
            pipeline,
            re.MULTILINE,
        )
    )

    checks.append(
        (
            "prepare_latents receives the local generation device",
            prepare_latents_check,
        )
    )

    # ========================================================
    # Print all checks.
    # ========================================================

    failed = False

    for description, result in checks:
        if result:
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            failed = True

    if failed:
        print("\n" + "=" * 60)
        print("❌ PATCH VERIFICATION FAILED")
        print("=" * 60)
        raise RuntimeError(
            "One or more LTX T4 patch checks failed."
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
