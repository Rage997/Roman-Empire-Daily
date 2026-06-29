"""
Basic unit tests — mock the external APIs so tests run without GPU or API keys.
Run with: pytest
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from roman_empire_ig.composer import compose
from roman_empire_ig.fact_generator import RomanContent, generate_fact_and_prompt


# ── fact_generator ────────────────────────────────────────────────────────────

MOCK_FACT = "Roman soldiers received part of their pay in salt — the origin of the word 'salary'."
MOCK_PROMPT = "A Roman legionary holding a small leather pouch of white salt crystals…"


@pytest.fixture()
def mock_anthropic(monkeypatch):
    """Patch the Anthropic client so no real API call is made."""
    mock_message = MagicMock()
    mock_message.content = [
        MagicMock(text=json.dumps({"fact": MOCK_FACT, "prompt": MOCK_PROMPT}))
    ]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message

    with patch("roman_empire_ig.fact_generator.anthropic.Anthropic", return_value=mock_client):
        yield mock_client


def test_generate_fact_and_prompt_returns_content(mock_anthropic):
    content = generate_fact_and_prompt()
    assert isinstance(content, RomanContent)
    assert content.fact == MOCK_FACT
    assert content.image_prompt == MOCK_PROMPT


def test_generate_fact_with_topic_hint(mock_anthropic):
    content = generate_fact_and_prompt(topic_hint="gladiators")
    # Just verify the call went through without error
    assert content.fact


def test_generate_fact_raises_on_invalid_json(monkeypatch):
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="not valid json")]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message

    with patch("roman_empire_ig.fact_generator.anthropic.Anthropic", return_value=mock_client):
        with pytest.raises(ValueError, match="invalid JSON"):
            generate_fact_and_prompt()


# ── composer ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def dummy_image() -> Image.Image:
    return Image.new("RGB", (1080, 1080), color=(100, 60, 30))


def test_compose_returns_image(dummy_image):
    result = compose(dummy_image, MOCK_FACT)
    assert isinstance(result, Image.Image)
    assert result.size == (1080, 1080)


def test_compose_no_overlay(dummy_image, monkeypatch):
    from roman_empire_ig import composer
    monkeypatch.setattr(composer.settings, "text_overlay", False)
    result = compose(dummy_image, MOCK_FACT)
    # Without overlay the image is returned unchanged
    assert result.size == dummy_image.size
