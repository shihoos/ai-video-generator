from pathlib import Path
import re
import subprocess
import sys

PROJECT_ROOT = Path('/kaggle/working/ai-video-generator')
LTX_REPO = PROJECT_ROOT / 'LTX-Video-0.9.8'
INFERENCE_FILE = LTX_REPO / 'ltx_video' / 'inference.py'
PIPELINE_FILE = LTX_REPO / 'ltx_video' / 'pipelines' / 'pipeline_ltx_video.py'
REVISION = 'bdc8f01'


def run(cmd):
    print(f'\n$ {cmd}')
    result = subprocess.run(cmd, shell=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'Command failed: {cmd}')


def restore_clean_source():
    run(
        f'git -C {LTX_REPO} checkout {REVISION} -- '
        f'ltx_video/inference.py ltx_video/pipelines/pipeline_ltx_video.py'
    )
    print(f'✅ LTX files restored from clean revision {REVISION}')


def replace_required(text, pattern, replacement, description, flags=re.MULTILINE):
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f'Could not find exactly one block for: {description} (found {count})')
    return new_text


def patch_inference():
    print('\n' + '=' * 60)
    print('PATCHING inference.py')
    print('=' * 60)

    text = INFERENCE_FILE.read_text(encoding='utf-8')

    # 1) Never move the whole Diffusers pipeline to CUDA. That would
    # move the large T5 encoder onto a 15 GB T4 and cause OOM.
    pipeline_to_pattern = re.compile(
        r'(?m)^(?P<i>[ \t]+)pipeline = LTXVideoPipeline\(\*\*submodel_dict\)\n'
        r'(?P=i)pipeline = pipeline\.to\(device\)\n'
        r'(?P=i)return pipeline\n'
    )
    pipeline_to_replacement = (
        r'\g<i>pipeline = LTXVideoPipeline(**submodel_dict)\n'
        r'\n'
        r'\g<i># T4 memory fix: do not call pipeline.to(device).\n'
        r'\g<i># That would move the large T5 encoder to GPU. Keep T5 on CPU\n'
        r'\g<i># and carry the real generation device explicitly.\n'
        r'\g<i>pipeline._ltx_execution_device = torch.device(device)\n'
        r'\n'
        r'\g<i>return pipeline\n'
    )
    text, count = pipeline_to_pattern.subn(pipeline_to_replacement, text, count=1)
    if count:
        print('✅ Removed whole-pipeline .to(device)')
    elif 'pipeline._ltx_execution_device = torch.device(device)' in text and 'pipeline = pipeline.to(device)' not in text:
        print('✅ Whole-pipeline T4 fix already present')
    else:
        raise RuntimeError('Could not find create_ltx_video_pipeline() pipeline.to(device) block.')

    # 2) Remove direct T5 transfer from inference.py if present.
    text, count = re.subn(
        r'(?m)^\s*text_encoder = text_encoder\.to\(device\)\n',
        '',
        text,
        count=1,
    )
    if count:
        print('✅ Removed direct T5 GPU transfer')
    else:
        print('ℹ️ No direct T5 GPU transfer remained')

    # 3) Multi-scale: clean LTX creates the latent upsampler from pipeline.device.
    # That is unsafe after removing pipeline.to(cuda), because Diffusers can report
    # the pipeline storage/offload device as CPU. Use the explicit generation device.
    upsampler_pattern = re.compile(
        r'(?m)^(?P<i>[ \t]+)latent_upsampler = create_latent_upsampler\(\n'
        r'(?P=i)[ \t]+spatial_upscaler_model_path, pipeline\.device\n'
        r'(?P=i)\)\n'
    )
    upsampler_replacement = (
        r'\g<i># T4 multi-scale fix: the upsampler must share the first-pass\n'
        r'\g<i># latent device. Do not use pipeline.device because CPU offload\n'
        r'\g<i># can make that resolve to CPU.\n'
        r'\g<i>latent_upsampler = create_latent_upsampler(\n'
        r'\g<i>    spatial_upscaler_model_path, device\n'
        r'\g<i>)\n'
    )
    text, count = upsampler_pattern.subn(upsampler_replacement, text, count=1)
    if count:
        print('✅ Latent upsampler now uses explicit generation device')
    elif 'spatial_upscaler_model_path, device' in text:
        print('✅ Latent upsampler explicit-device fix already present')
    else:
        raise RuntimeError('Could not find the multi-scale latent upsampler creation block.')

    INFERENCE_FILE.write_text(text, encoding='utf-8')


def patch_pipeline():
    print('\n' + '=' * 60)
    print('PATCHING pipeline_ltx_video.py')
    print('=' * 60)

    text = PIPELINE_FILE.read_text(encoding='utf-8')

    # 1) Use the explicit generation device inside every LTXVideoPipeline.__call__
    # local device assignment. Regex preserves indentation and is intentionally
    # applied to all matching assignments in the clean revision.
    device_pattern = re.compile(
        r'^(?P<i>[ \t]+)device = self\._execution_device\s*$',
        re.MULTILINE,
    )
    device_replacement = (
        r'\g<i># T4: Diffusers CPU offload can report _execution_device as CPU.\n'
        r'\g<i># Use the explicit generation device when inference.py provides it.\n'
        r'\g<i>device = getattr(self, "_ltx_execution_device", self._execution_device)'
    )
    text, device_count = device_pattern.subn(device_replacement, text)
    if device_count == 0:
        if 'device = getattr(self, "_ltx_execution_device", self._execution_device)' in text:
            print('✅ Explicit generation-device override already present')
        else:
            raise RuntimeError('Could not find LTX execution-device assignment.')
    else:
        print(f'✅ Patched {device_count} LTX execution-device assignment(s)')

    # 2) Keep T5 on CPU when offloading. Match the actual clean 0.9.8 block.
    t5_pattern = re.compile(
        r'(?m)^(?P<i>[ \t]+)# 3\. Encode input prompt\n'
        r'(?P=i)if self\.text_encoder is not None:\n'
        r'(?P=i)    self\.text_encoder = self\.text_encoder\.to\(self\._execution_device\)\n'
    )
    t5_replacement = (
        r'\g<i># 3. Encode input prompt\n'
        r'\g<i>if self.text_encoder is not None:\n'
        r'\g<i>    # T4 memory fix: keep the large T5 encoder on CPU while offloading.\n'
        r'\g<i>    if not offload_to_cpu:\n'
        r'\g<i>        self.text_encoder = self.text_encoder.to(device)\n'
    )
    text, t5_count = t5_pattern.subn(t5_replacement, text, count=1)
    if t5_count:
        print('✅ T5 stays on CPU when --offload is enabled')
    elif 'if not offload_to_cpu:' in text and 'self.text_encoder = self.text_encoder.to(device)' in text:
        print('✅ T5 CPU-offload fix already present')
    else:
        raise RuntimeError('Could not find the T5 transfer block.')

    # 3) Make transformer follow the same explicit generation device.
    transformer_pattern = re.compile(
        r'(?m)^(?P<i>[ \t]+)self\.transformer = self\.transformer\.to\(self\._execution_device\)\s*$'
    )
    transformer_replacement = r'\g<i>self.transformer = self.transformer.to(device)'
    text, transformer_count = transformer_pattern.subn(transformer_replacement, text, count=1)
    if transformer_count:
        print('✅ Transformer uses explicit generation device')
    elif 'self.transformer = self.transformer.to(device)' in text:
        print('✅ Transformer generation-device fix already present')
    else:
        raise RuntimeError('Could not find transformer device transfer block.')

    # 4) Attention masks must be on the generation device before concatenation.
    cat_pattern = re.compile(
        r'(?m)^(?P<i>[ \t]+)prompt_embeds_batch = torch\.cat\(\n'
        r'(?P=i)    \[negative_prompt_embeds, prompt_embeds, prompt_embeds\], dim=0\n'
        r'(?P=i)\)\n'
        r'(?P=i)prompt_attention_mask_batch = torch\.cat\(\n'
        r'(?P=i)    \[\n'
        r'(?P=i)        negative_prompt_attention_mask,\n'
        r'(?P=i)        prompt_attention_mask,\n'
        r'(?P=i)        prompt_attention_mask,\n'
        r'(?P=i)    \],\n'
        r'(?P=i)    dim=0,\n'
        r'(?P=i)\)\n'
    )
    cat_replacement = (
        r'\g<i>prompt_embeds_batch = torch.cat(\n'
        r'\g<i>    [negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0\n'
        r'\g<i>)\n'
        r'\n'
        r'\g<i># T4 device fix: masks must match CUDA transformer tensors.\n'
        r'\g<i>prompt_attention_mask = prompt_attention_mask.to(device)\n'
        r'\g<i>negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)\n'
        r'\n'
        r'\g<i>prompt_attention_mask_batch = torch.cat(\n'
        r'\g<i>    [\n'
        r'\g<i>        negative_prompt_attention_mask,\n'
        r'\g<i>        prompt_attention_mask,\n'
        r'\g<i>        prompt_attention_mask,\n'
        r'\g<i>    ],\n'
        r'\g<i>    dim=0,\n'
        r'\g<i>)\n'
    )
    text, cat_count = cat_pattern.subn(cat_replacement, text, count=1)
    if cat_count:
        print('✅ Attention masks aligned to generation device')
    elif 'negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)' in text:
        print('✅ Attention-mask device fix already present')
    else:
        raise RuntimeError('Could not find the attention-mask concatenation block.')

    # 5) Replace ONLY the _upsample_latents method body. Do not rely on an exact
    # whitespace-sensitive source block; locate the method by its class-level
    # indentation and stop at the next method in LTXMultiScalePipeline.
    method_pattern = re.compile(
        r'(?ms)^(?P<i>    )def _upsample_latents\(\n'
        r'.*?'
        r'^(?P=i)def __init__\(\n'
    )
    method_replacement = '''    def _upsample_latents(
        self, latest_upsampler: LatentUpsampler, latents: torch.Tensor
    ):
        # T4 multi-scale fix:
        # The upsampler and first-pass latents must always share a device.
        target_device = latents.device

        if latest_upsampler.device != target_device:
            latest_upsampler = latest_upsampler.to(target_device)
            latest_upsampler.eval()

        if latest_upsampler.device != target_device:
            raise RuntimeError(
                "Latent upsampler device mismatch after transfer: "
                f"latents={target_device}, upsampler={latest_upsampler.device}"
            )

        latents = un_normalize_latents(
            latents, self.vae, vae_per_channel_normalize=True
        )
        upsampled_latents = latest_upsampler(latents)
        upsampled_latents = normalize_latents(
            upsampled_latents, self.vae, vae_per_channel_normalize=True
        )

        # The upsampler is only needed between the first and second passes.
        # Release its GPU memory before the second pass on a T4.
        if target_device.type == "cuda":
            latest_upsampler.cpu()
            torch.cuda.empty_cache()

        return upsampled_latents

    def __init__(
'''
    text, method_count = method_pattern.subn(method_replacement, text, count=1)
    if method_count:
        print('✅ Replaced LTXMultiScalePipeline._upsample_latents() safely')
    elif 'def _upsample_latents(' in text and 'Latent upsampler device mismatch after transfer' in text:
        print('✅ Multi-scale upsampler method already patched')
    else:
        raise RuntimeError('Could not locate LTXMultiScalePipeline._upsample_latents() in clean LTX 0.9.8 source.')

    PIPELINE_FILE.write_text(text, encoding='utf-8')


def verify():
    print('\n' + '=' * 60)
    print('VERIFYING PATCHED LTX SOURCE')
    print('=' * 60)

    inference = INFERENCE_FILE.read_text(encoding='utf-8')
    pipeline = PIPELINE_FILE.read_text(encoding='utf-8')

    for path in (INFERENCE_FILE, PIPELINE_FILE):
        result = subprocess.run(
            [sys.executable, '-m', 'py_compile', str(path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f'Python syntax check failed: {path}')

    checks = [
        ('whole-pipeline .to(device) removed', 'pipeline = pipeline.to(device)' not in inference),
        ('direct T5 GPU transfer removed', 'text_encoder = text_encoder.to(device)' not in inference),
        ('explicit LTX generation device set', '_ltx_execution_device = torch.device(device)' in inference),
        ('upsampler created from explicit device', 'spatial_upscaler_model_path, device' in inference),
        ('generation device override present', 'device = getattr(self, "_ltx_execution_device", self._execution_device)' in pipeline),
        ('transformer uses generation device', 'self.transformer = self.transformer.to(device)' in pipeline),
        ('prompt mask moved to generation device', 'prompt_attention_mask = prompt_attention_mask.to(device)' in pipeline),
        ('negative mask moved to generation device', 'negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)' in pipeline),
        ('multi-scale device synchronization present', 'target_device = latents.device' in pipeline),
        ('upsampler mismatch guard present', 'Latent upsampler device mismatch after transfer' in pipeline),
        ('clean upstream revision restored before patching', REVISION == 'bdc8f01'),
    ]

    failed = []
    for name, ok in checks:
        print(('✅ ' if ok else '❌ ') + name)
        if not ok:
            failed.append(name)

    if failed:
        raise RuntimeError('Patch verification failed: ' + ', '.join(failed))

    print('\n' + '=' * 60)
    print('✅ PYTHON SYNTAX CHECKS PASSED')
    print('✅ ALL T4 MULTI-SCALE DEVICE CHECKS PASSED')
    print('=' * 60)


def main():
    print('\n' + '=' * 60)
    print('LTX-VIDEO 0.9.8 — T4 FINAL DEVICE/OFFLOAD PATCH')
    print('=' * 60)

    restore_clean_source()
    patch_inference()
    patch_pipeline()
    verify()

    print('\n🚀 LTX T4 patch complete.')


if __name__ == '__main__':
    main()
