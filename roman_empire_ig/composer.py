"""
composer.py
Takes a raw generated image and a fact string, then:
  - Draws a semi-transparent dark bar at the bottom
  - Renders the fact text over it (word-wrapped)
  - Returns a final PIL Image ready to save or post
"""

import logging
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import settings

logger = logging.getLogger(__name__)

# Fallback to a built-in PIL font if no TTF is available
_FALLBACK_FONT_SIZE = 40


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a nice serif font; fall back to PIL's default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/System/Library/Fonts/Times.ttc",  # macOS fallback
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    logger.warning("No TTF font found — using PIL default (will look basic).")
    return ImageFont.load_default()


def compose(image: Image.Image, fact: str) -> Image.Image:
    """
    Overlay the fact text on the image with a dark semi-transparent bar.

    Args:
        image: The raw generated PIL Image.
        fact:  The Roman Empire fact string.

    Returns:
        Composited PIL Image.
    """
    if not settings.text_overlay:
        return image

    img = image.copy().convert("RGBA")
    w, h = img.size

    font = _load_font(settings.font_size)

    # Word-wrap the fact to fit within 90% of image width
    # Approximate character width based on font size
    chars_per_line = max(20, int(w * 0.85 / (settings.font_size * 0.55)))
    lines = textwrap.wrap(fact, width=chars_per_line)
    wrapped_text = "\n".join(lines)

    # Measure text block height
    dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = dummy_draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=8)
    text_h = bbox[3] - bbox[1]
    padding = 32
    bar_h = text_h + padding * 2

    # Draw dark overlay bar
    overlay = Image.new("RGBA", (w, bar_h), (0, 0, 0, 0))
    bar_color = (0, 0, 0, int(255 * settings.overlay_opacity))
    ImageDraw.Draw(overlay).rectangle([(0, 0), (w, bar_h)], fill=bar_color)

    bar_y = h - bar_h
    img.paste(overlay, (0, bar_y), overlay)

    # Parse hex font color
    hex_color = settings.font_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    font_rgba = (r, g, b, 255)

    # Draw text centered horizontally, vertically centered in the bar
    draw = ImageDraw.Draw(img)
    text_x = w // 2
    text_y = bar_y + padding
    draw.multiline_text(
        (text_x, text_y),
        wrapped_text,
        font=font,
        fill=font_rgba,
        anchor="ma",  # middle-top anchor
        align="center",
        spacing=8,
    )

    logger.info("Text overlay applied (%d lines).", len(lines))
    return img.convert("RGB")


def save(image: Image.Image, stem: str) -> Path:
    """Save the final image to the configured output directory."""
    out_dir = settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.jpg"
    image.save(path, format="JPEG", quality=95)
    logger.info("Saved → %s", path)
    return path
