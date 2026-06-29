"""
pipeline.py
Orchestrates the full end-to-end flow:
  1. Generate a Roman fact + image prompt  (Claude)
  2. Generate the image                    (FLUX / diffusers)
  3. Compose text overlay                  (Pillow)
  4. Save locally
  5. Post to Instagram (optional)
"""

import logging
import re
import unicodedata
from datetime import datetime

from .composer import compose, save
from .config import settings
from .fact_generator import generate_fact_and_prompt
from .image_generator import generate_image
from .poster import build_caption, post

logger = logging.getLogger(__name__)


def _slugify(text: str, max_len: int = 60) -> str:
    """Turn arbitrary text into a safe filename stem."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:max_len]


def run(topic_hint: str = "") -> None:
    """
    Execute one full pipeline cycle.

    Args:
        topic_hint: Optional Roman topic to steer fact generation
                    (e.g. "aqueducts", "gladiators", "Julius Caesar").
    """
    logger.info("═══ Pipeline start ═══")

    # 1. Generate fact + prompt
    content = generate_fact_and_prompt(topic_hint=topic_hint)

    # 2. Generate image
    raw_image = generate_image(prompt=content.image_prompt)

    # 3. Compose overlay
    final_image = compose(image=raw_image, fact=content.fact)

    # 4. Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{_slugify(content.fact)}"
    output_path = save(image=final_image, stem=stem)

    # 5. Post to Instagram
    caption = build_caption(fact=content.fact)
    media_id = post(image_path=output_path, caption=caption)
    if media_id != "dry-run":
        logger.info("Posted to Instagram (media_id=%s).", media_id)

    logger.info("═══ Pipeline done — output: %s ═══", output_path)
