"""
video_generator.py
Runs Wan2.2 (TI2V-5B) via diffusers for short text-to-video generation.
Fits comfortably on a 3090 24GB with model CPU offload enabled.
The pipeline is loaded once and cached for the process lifetime.
"""

import logging
import os
import random
from functools import lru_cache

import torch
from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanPipeline,
    WanTransformer3DModel,
)
from diffusers.utils import export_to_video

from .config import settings

logger = logging.getLogger(__name__)

# Reduces fragmentation — important for large models on 24GB cards
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

MAX_NATIVE_FRAMES = 121  # ~5s @ 24fps — the model's trained clip length
REQUIRED_DIM_MULTIPLE = 32  # Wan2.2-VAE compression (4x16x16 + 2x patchify)


def _validate_resolution(width: int, height: int) -> None:
    if width % REQUIRED_DIM_MULTIPLE != 0 or height % REQUIRED_DIM_MULTIPLE != 0:
        raise ValueError(
            f"Wan requires width/height divisible by {REQUIRED_DIM_MULTIPLE}, "
            f"got {width}x{height}. Adjust settings.video_width/video_height."
        )


@lru_cache(maxsize=1)
def _load_pipeline(flow_shift: float) -> WanPipeline:
    """
    Load the Wan2.2 TI2V-5B pipeline for a given flow_shift.
    The VAE is kept in float32 for decode quality; the transformer and text
    encoder run in bfloat16. CPU offload keeps peak VRAM well under 24GB.

    Cached per flow_shift (there are only two sensible values: 5.0 for
    720p-class resolutions, 3.0 for lower) instead of caching with zero
    arguments against mutable `settings` — a zero-arg cache would silently
    keep serving a pipeline built with a stale flow_shift/resolution
    config if settings.video_height changes later in the same process.
    """
    dtype = torch.bfloat16

    logger.info("Loading Wan2.2-TI2V-5B from '%s' (flow_shift=%.1f) …", REPO_ID, flow_shift)
    logger.info("This will download ~15GB on first run — subsequent runs use cache.")

    vae = AutoencoderKLWan.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    transformer = WanTransformer3DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    pipe = WanPipeline.from_pretrained(
        REPO_ID,
        vae=vae,
        transformer=transformer,
        torch_dtype=dtype,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=flow_shift
    )

    # Offload modules to GPU only when active — keeps this well within 24GB
    pipe.enable_model_cpu_offload()

    # Decode the VAE in spatial/temporal tiles instead of all at once.
    # This avoids VRAM spikes during decode of high-res / many-frame clips.
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    logger.info("Wan2.2 pipeline ready.")
    return pipe


def generate_video(prompt: str, output_path: str) -> str:
    """
    Generate a single short video clip (up to ~5s) from a text prompt
    using Wan2.2 (TI2V-5B).

    Args:
        prompt: The detailed video-generation prompt.
        output_path: Where to write the resulting .mp4 file.

    Returns:
        The output_path, once the video has been written to disk.
    """

    logger.info("Generating video with prompt: %s", prompt)

    _validate_resolution(settings.video_width, settings.video_height)

    # flow_shift: Wan's official configs use ~5.0 for 720p-class resolutions,
    # ~3.0 for lower resolutions like 480p. Computed here (not baked into a
    # cached zero-arg loader) so it always reflects current settings.
    flow_shift = 5.0 if settings.video_height >= 720 else 3.0
    pipe = _load_pipeline(flow_shift)

    seed = settings.seed if settings.seed >= 0 else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Wan requires num_frames = 4*k + 1, capped at the model's native
    # trained clip length (~121 frames / 5s). Asking for more pushes the
    # model outside its training distribution and hurts quality badly.
    requested_frames = settings.video_seconds * settings.fps
    if requested_frames > MAX_NATIVE_FRAMES:
        logger.warning(
            "Requested %.1fs (%d frames) exceeds Wan's native ~%.1fs clip "
            "length; capping to %d frames (~%.1fs). Chain multiple segments "
            "via image-to-video continuation for longer clips.",
            settings.video_seconds,
            requested_frames,
            MAX_NATIVE_FRAMES / settings.fps,
            MAX_NATIVE_FRAMES,
            MAX_NATIVE_FRAMES / settings.fps,
        )
    raw_frames = min(requested_frames, MAX_NATIVE_FRAMES)
    num_frames = ((raw_frames - 1) // 4) * 4 + 1

    logger.info(
        "Generating video (seed=%d, steps=%d, guidance=%.1f, frames=%d)…",
        seed,
        settings.num_inference_steps,
        settings.guidance_scale,
        num_frames,
    )

    try:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=settings.negative_prompt,
                width=settings.video_width,
                height=settings.video_height,
                num_frames=num_frames,
                num_inference_steps=settings.num_inference_steps,
                guidance_scale=settings.guidance_scale,
                generator=generator,
            )
    except torch.OutOfMemoryError:
        logger.error(
            "CUDA OOM while generating (prompt=%r, %dx%d, %d frames, %d steps). "
            "Try enable_sequential_cpu_offload(), a lower resolution, or fewer frames.",
            prompt, settings.video_width, settings.video_height,
            num_frames, settings.num_inference_steps,
        )
        raise

    frames = result.frames[0]
    export_to_video(frames, output_path, fps=settings.fps)

    logger.info(
        "Video generated: %dx%d px, %d frames, saved to %s",
        settings.video_width,
        settings.video_height,
        num_frames,
        output_path,
    )
    return output_path