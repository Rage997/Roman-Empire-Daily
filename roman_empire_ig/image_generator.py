"""
image_generator.py
Runs FLUX.2-dev (4-bit NF4 quantized) via diffusers.
Fits comfortably on a 3090 24GB (~18GB peak VRAM usage).
The pipeline is loaded once and cached for the process lifetime.
"""

import logging
import os
import random
from functools import lru_cache

import torch
from diffusers import Flux2Pipeline, AutoModel
from transformers import Mistral3ForConditionalGeneration
from PIL import Image

from .config import settings

logger = logging.getLogger(__name__)

# Reduces fragmentation — important for large models on 24GB cards
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ID = "diffusers/FLUX.2-dev-bnb-4bit"


@lru_cache(maxsize=1)
def _load_pipeline() -> Flux2Pipeline:
    """
    Load the FLUX.2-dev 4-bit quantized pipeline.
    Components are loaded on CPU first, then offloaded to GPU on demand.
    Peak VRAM usage: ~18GB on a 3090.
    """
    dtype = torch.bfloat16

    logger.info("Loading FLUX.2-dev (4-bit NF4) from '%s' …", REPO_ID)
    logger.info("This will download ~20GB on first run — subsequent runs use cache.")

    # Load text encoder (Mistral) on CPU — it's large and only needed briefly
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map="cpu",
    )

    # Load the DiT transformer on CPU first to avoid VRAM spike during load
    transformer = AutoModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
    )

    # Assemble full pipeline
    pipe = Flux2Pipeline.from_pretrained(
        REPO_ID,
        text_encoder=text_encoder,
        transformer=transformer,
        torch_dtype=dtype,
    )

    # Offload modules to GPU only when active — stays within 24GB
    pipe.enable_model_cpu_offload()

    logger.info("FLUX.2-dev pipeline ready.")
    return pipe


def generate_image(prompt: str) -> Image.Image:
    """
    Generate an image from a text prompt using FLUX.2-dev.

    Args:
        prompt: The detailed image-generation prompt.

    Returns:
        A PIL Image at the configured resolution.
    """

    logger.info("Generating image with prompt: %s", prompt)

    pipe = _load_pipeline()

    seed = settings.seed if settings.seed >= 0 else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    logger.info(
        "Generating image (seed=%d, steps=%d, guidance=%.1f)…",
        seed,
        settings.num_inference_steps,
        settings.guidance_scale,
    )

    result = pipe(
        prompt=prompt,
        width=settings.image_width,
        height=settings.image_height,
        num_inference_steps=settings.num_inference_steps,
        guidance_scale=settings.guidance_scale,
        generator=generator,
    )

    image: Image.Image = result.images[0]
    logger.info("Image generated: %dx%d px", image.width, image.height)
    return image