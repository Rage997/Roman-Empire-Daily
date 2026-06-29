"""
poster.py
Publishes a finished image to Instagram via the official Graph API.
Requires a Facebook/Instagram Business account with a connected app.

Set INSTAGRAM_ACCESS_TOKEN and INSTAGRAM_ACCOUNT_ID in your .env.
Set AUTO_POST=false to skip posting and just generate images locally.
"""

import logging
from pathlib import Path

import requests

from .config import settings

logger = logging.getLogger(__name__)

_BASE_URL = "https://graph.facebook.com/v19.0"


class InstagramPosterError(RuntimeError):
    pass


def _api(method: str, endpoint: str, **kwargs) -> dict:
    url = f"{_BASE_URL}/{endpoint}"
    resp = requests.request(method, url, timeout=30, **kwargs)
    data = resp.json()
    if "error" in data:
        raise InstagramPosterError(f"Graph API error: {data['error']}")
    return data


def post(image_path: Path, caption: str) -> str:
    """
    Upload and publish a single image to Instagram.

    Args:
        image_path: Local path to the JPEG image.
        caption:    Post caption (the Roman fact + hashtags).

    Returns:
        The published Instagram media ID.
    """
    if not settings.auto_post:
        logger.info("AUTO_POST is disabled — skipping Instagram upload.")
        return "dry-run"

    if not settings.instagram_access_token or not settings.instagram_account_id:
        raise InstagramPosterError(
            "INSTAGRAM_ACCESS_TOKEN and INSTAGRAM_ACCOUNT_ID must be set in .env "
            "to enable posting."
        )

    # Step 1: create a media container (Instagram needs a public URL)
    # For simplicity we upload via a publicly reachable image URL.
    # In production: upload image_path to S3/GCS, get a public URL, pass it here.
    raise NotImplementedError(
        "To post to Instagram you need to host the image at a public URL first.\n"
        "Upload the file to S3/GCS/Cloudflare R2, then pass the URL to the "
        "Graph API container endpoint. See: "
        "https://developers.facebook.com/docs/instagram-api/guides/content-publishing"
    )


def build_caption(fact: str) -> str:
    """Combine the fact with evergreen Roman history hashtags."""
    hashtags = (
        "#RomanEmpire #AncientRome #RomanHistory #Rome #AncientHistory "
        "#RomanFacts #HistoryFacts #ClassicalAntiquity #DidYouKnow #History"
    )
    return f"{fact}\n\n{hashtags}"
