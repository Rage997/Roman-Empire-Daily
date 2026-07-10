"""
video_generator.py

Wan 2.2 image-to-video and text-to-video generation via Diffusers.

Requirements:
    pip install torch diffusers transformers accelerate imageio imageio-ffmpeg pillow
"""

import logging
import os
import random
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Union

import torch
from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
    WanPipeline,
    WanTransformer3DModel,
)
from diffusers.utils import export_to_video
from PIL import Image


try:
    from .config import settings
except ImportError:
    class Settings:
        video_width = 832
        video_height = 480
        video_seconds = 3.0
        fps = 24
        num_inference_steps = 20
        guidance_scale = 6.0
        negative_prompt = ""
        seed = -1

    settings = Settings()


REPO_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
MAX_NATIVE_FRAMES = 121
REQUIRED_DIM_MULTIPLE = 32

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(console_handler)


def _validate_resolution(width: int, height: int) -> None:
    if width % REQUIRED_DIM_MULTIPLE != 0:
        raise ValueError(
            f"Width must be divisible by {REQUIRED_DIM_MULTIPLE}, got {width}."
        )

    if height % REQUIRED_DIM_MULTIPLE != 0:
        raise ValueError(
            f"Height must be divisible by {REQUIRED_DIM_MULTIPLE}, got {height}."
        )


def _resolve_frame_count(video_seconds: float, fps: int) -> int:
    """
    Wan requires num_frames = 4*k + 1.
    """
    requested_frames = max(1, int(video_seconds * fps))
    capped_frames = min(requested_frames, MAX_NATIVE_FRAMES)
    frame_count = ((capped_frames - 1) // 4) * 4 + 1

    logger.info(
        "Frames: %d requested -> %d generated (%.2fs at %d FPS)",
        requested_frames,
        frame_count,
        frame_count / fps,
        fps,
    )

    return frame_count


def _make_generator() -> torch.Generator:
    seed = settings.seed if settings.seed >= 0 else random.randint(0, 2**32 - 1)
    logger.info("Seed: %d", seed)
    return torch.Generator(device="cpu").manual_seed(seed)


def _prepare_image(
    image: Union[str, Path, Image.Image],
) -> tuple[Image.Image, int, int]:
    if isinstance(image, (str, Path)):
        image_path = Path(image)

        if not image_path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {image_path}")

        logger.info("Loading input image: %s", image_path.resolve())

        with Image.open(image_path) as loaded_image:
            image = loaded_image.convert("RGB")

    if not isinstance(image, Image.Image):
        raise TypeError(
            "image must be a file path, pathlib.Path, or PIL.Image.Image"
        )

    target_width = settings.video_width
    target_height = settings.video_height

    _validate_resolution(target_width, target_height)

    original_width = image.width
    original_height = image.height

    source_aspect = image.width / image.height
    target_aspect = target_width / target_height

    if source_aspect >= target_aspect:
        width = target_width
        height = round((target_width / source_aspect) / REQUIRED_DIM_MULTIPLE)
        height *= REQUIRED_DIM_MULTIPLE
    else:
        height = target_height
        width = round((target_height * source_aspect) / REQUIRED_DIM_MULTIPLE)
        width *= REQUIRED_DIM_MULTIPLE

    width = max(REQUIRED_DIM_MULTIPLE, width)
    height = max(REQUIRED_DIM_MULTIPLE, height)

    _validate_resolution(width, height)

    image = image.resize((width, height), Image.LANCZOS)

    logger.info(
        "Input image prepared: %dx%d -> %dx%d",
        original_width,
        original_height,
        width,
        height,
    )

    return image, width, height

def _extract_frames(result) -> list:
    """
    Normalize Diffusers video output to a list of PIL.Image.Image frames.

    Depending on the Diffusers version, result.frames can be:
        - list[list[PIL.Image.Image]]
        - list[PIL.Image.Image]
        - numpy.ndarray with shape:
            [batch, frames, height, width, channels]
            [frames, height, width, channels]
    """
    import numpy as np

    if not hasattr(result, "frames"):
        raise RuntimeError(
            f"Unexpected pipeline result type: {type(result)}. "
            "Expected an object with a .frames attribute."
        )

    frames = result.frames

    if frames is None:
        raise RuntimeError("Wan returned no frames.")

    logger.info("Raw result.frames type: %s", type(frames))

    if isinstance(frames, np.ndarray):
        logger.info("Raw result.frames shape: %s", frames.shape)

        if frames.size == 0:
            raise RuntimeError("Wan returned an empty frame array.")

        # Typical Diffusers output:
        # [batch, num_frames, height, width, channels]
        if frames.ndim == 5:
            frames = frames[0]

        # Now expected:
        # [num_frames, height, width, channels]
        if frames.ndim != 4:
            raise RuntimeError(
                f"Unexpected NumPy frame shape: {frames.shape}. "
                "Expected [frames, height, width, channels]."
            )

        # Convert float [0, 1] or uint8 arrays into PIL images.
        if frames.dtype != np.uint8:
            if frames.max() <= 1.0:
                frames = (frames * 255).clip(0, 255).astype(np.uint8)
            else:
                frames = frames.clip(0, 255).astype(np.uint8)

        frames = [Image.fromarray(frame) for frame in frames]

    elif isinstance(frames, (list, tuple)):
        if len(frames) == 0:
            raise RuntimeError("Wan returned no frames.")

        # Typical list output:
        # [[frame_0, frame_1, ...]]
        if isinstance(frames[0], (list, tuple)):
            frames = frames[0]

        if len(frames) == 0:
            raise RuntimeError("Wan returned an empty frame sequence.")

        frames = list(frames)

    else:
        raise RuntimeError(
            f"Unsupported result.frames type: {type(frames)}. "
            "Expected a list, tuple, or numpy.ndarray."
        )

    if not frames:
        raise RuntimeError("Wan returned an empty frame sequence.")

    first_frame = frames[0]

    if not isinstance(first_frame, Image.Image):
        raise RuntimeError(
            f"Expected PIL.Image.Image frames after normalization, "
            f"got {type(first_frame)}."
        )

    logger.info(
        "Normalized %d frames (%dx%d, %s)",
        len(frames),
        first_frame.width,
        first_frame.height,
        first_frame.mode,
    )

    return frames


def _export_video(frames: list, output_path: Union[str, Path]) -> str:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Writing %d frames at %d FPS to: %s",
        len(frames),
        settings.fps,
        output_path,
    )

    export_to_video(
        video_frames=frames,
        output_video_path=str(output_path),
        fps=settings.fps,
    )

    if not output_path.is_file():
        raise FileNotFoundError(
            f"Video export finished but output file does not exist: {output_path}"
        )

    file_size = output_path.stat().st_size

    if file_size == 0:
        raise RuntimeError(f"Generated video is empty: {output_path}")

    logger.info(
        "Video generated successfully: %s (%.2f MB)",
        output_path,
        file_size / (1024 * 1024),
    )

    return str(output_path)


@lru_cache(maxsize=2)
def _load_i2v_pipeline(flow_shift: float) -> WanImageToVideoPipeline:
    """
    Load only the image-to-video pipeline.

    Do not load the text-to-video pipeline here. They each load their own
    VAE and transformer, which wastes RAM and VRAM.
    """
    dtype = torch.bfloat16

    logger.info("=" * 72)
    logger.info("Loading Wan 2.2 image-to-video pipeline")
    logger.info("Repository: %s", REPO_ID)
    logger.info("Flow shift: %.1f", flow_shift)
    logger.info("CUDA available: %s", torch.cuda.is_available())

    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
        logger.info(
            "GPU memory: %.1f GB",
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
        )

    started_at = time.time()

    logger.info("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    logger.info("Loading transformer...")
    transformer = WanTransformer3DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    logger.info("Building image-to-video pipeline...")
    pipe = WanImageToVideoPipeline.from_pretrained(
        REPO_ID,
        vae=vae,
        transformer=transformer,
        torch_dtype=dtype,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=flow_shift,
    )

    if torch.cuda.is_available():
        logger.info("Enabling model CPU offload...")
        pipe.enable_model_cpu_offload()
    else:
        logger.warning("CUDA unavailable. CPU generation will be extremely slow.")

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    logger.info(
        "Image-to-video pipeline loaded in %.1f seconds",
        time.time() - started_at,
    )
    logger.info("=" * 72)

    return pipe


@lru_cache(maxsize=2)
def _load_t2v_pipeline(flow_shift: float) -> WanPipeline:
    dtype = torch.bfloat16

    logger.info("=" * 72)
    logger.info("Loading Wan 2.2 text-to-video pipeline")
    logger.info("Repository: %s", REPO_ID)

    started_at = time.time()

    logger.info("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    logger.info("Loading transformer...")
    transformer = WanTransformer3DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    logger.info("Building text-to-video pipeline...")
    pipe = WanPipeline.from_pretrained(
        REPO_ID,
        vae=vae,
        transformer=transformer,
        torch_dtype=dtype,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=flow_shift,
    )

    if torch.cuda.is_available():
        logger.info("Enabling model CPU offload...")
        pipe.enable_model_cpu_offload()
    else:
        logger.warning("CUDA unavailable. CPU generation will be extremely slow.")

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    logger.info(
        "Text-to-video pipeline loaded in %.1f seconds",
        time.time() - started_at,
    )
    logger.info("=" * 72)

    return pipe


def generate_video_from_image(
    image: Union[str, Path, Image.Image],
    prompt: str,
    output_path: Union[str, Path],
) -> str:
    """
    Generate a video from a still image.

    Args:
        image: Image path or PIL.Image.Image.
        prompt: Motion prompt.
        output_path: MP4 destination path.

    Returns:
        Absolute output MP4 path.
    """
    logger.info("=" * 72)
    logger.info("Starting Wan image-to-video generation")
    logger.info("Prompt: %s", prompt)
    logger.info("=" * 72)

    image, width, height = _prepare_image(image)

    flow_shift = 5.0 if height >= 720 else 3.0
    pipe = _load_i2v_pipeline(flow_shift)

    num_frames = _resolve_frame_count(
        settings.video_seconds,
        settings.fps,
    )

    generator = _make_generator()

    logger.info(
        "Parameters: %dx%d | %d frames | %d steps | guidance %.1f",
        width,
        height,
        num_frames,
        settings.num_inference_steps,
        settings.guidance_scale,
    )

    started_at = time.time()

    try:
        with torch.inference_mode():
            result = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=settings.negative_prompt,
                num_frames=num_frames,
                num_inference_steps=settings.num_inference_steps,
                guidance_scale=settings.guidance_scale,
                generator=generator,
            )
    except torch.OutOfMemoryError as error:
        raise RuntimeError(
            "CUDA ran out of memory. Lower video_width/video_height, "
            "video_seconds, or num_inference_steps."
        ) from error

    logger.info(
        "Inference completed in %.1f seconds",
        time.time() - started_at,
    )

    frames = _extract_frames(result)
    return _export_video(frames, output_path)


def generate_video(
    prompt: str,
    output_path: Union[str, Path],
) -> str:
    """
    Generate a video from text.

    Args:
        prompt: Video prompt.
        output_path: MP4 destination path.

    Returns:
        Absolute output MP4 path.
    """
    logger.info("=" * 72)
    logger.info("Starting Wan text-to-video generation")
    logger.info("Prompt: %s", prompt)
    logger.info("=" * 72)

    _validate_resolution(
        settings.video_width,
        settings.video_height,
    )

    flow_shift = 5.0 if settings.video_height >= 720 else 3.0
    pipe = _load_t2v_pipeline(flow_shift)

    num_frames = _resolve_frame_count(
        settings.video_seconds,
        settings.fps,
    )

    generator = _make_generator()

    logger.info(
        "Parameters: %dx%d | %d frames | %d steps | guidance %.1f",
        settings.video_width,
        settings.video_height,
        num_frames,
        settings.num_inference_steps,
        settings.guidance_scale,
    )

    started_at = time.time()

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
    except torch.OutOfMemoryError as error:
        raise RuntimeError(
            "CUDA ran out of memory. Lower video_width/video_height, "
            "video_seconds, or num_inference_steps."
        ) from error

    logger.info(
        "Inference completed in %.1f seconds",
        time.time() - started_at,
    )

    frames = _extract_frames(result)
    return _export_video(frames, output_path)