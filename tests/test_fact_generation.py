"""
Tests for the fact_generator module — replaces the out-of-date Anthropic stubs.
Uses real Ollama mocking + a manual live-run test that prints to the console.

Run with: pytest tests/ -v
Run just the live-LLM test (requires local Ollama):
    pytest tests/test_fact_generation.py::TestIntegration::test_live_generate_and_print -v
"""

import json
import sys
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from roman_empire_ig.fact_generator import (
    TOPICS,
    _LLMSceneResponse,
    _pick_topic,
    client,
    generate_fact_and_prompt,
)


# ── Mock Helpers ────────────────────────────────────────────────────────

def _mock_ollama_chat(fact="Roman soldiers received part of their pay in salt.",
                      scene="A dramatic arena filled with cheering crowds..."):
    """Return a fully configured MagicMock that pretends to be an Ollama .chat() call."""
    mock_msg = MagicMock()
    response_body = {
        "topic": "gladiators",
        "fact": fact,
        "scene_description": scene,
    }
    mock_msg.content = json.dumps(response_body)

    mock_response = MagicMock()
    mock_response.message = mock_msg  # Ollama wraps result in .message

    return mock_response


# ── Unit Tests (no real LLM required) ───────────────────────────────────

class TestGenerateFactAndPrompt:
    """Core generation tests — patching the client so no network calls happen."""

    def test_returns_valid_roman_content(self, monkeypatch):
        monkeypatch.setattr(client, "chat", lambda **_: _mock_ollama_chat())
        result = generate_fact_and_prompt()
        assert result is not None
        assert len(result.fact) > 0
        assert len(result.image_prompt) > 0

    def test_topic_hint_is_forwarded(self, monkeypatch):
        forced_topic = "aqueducts"

        def capture_kwargs(**kwargs):
            assert kwargs["messages"][0]["content"].find(forced_topic) >= 0
            return _mock_ollama_chat()

        monkeypatch.setattr(client, "chat", capture_kwargs)
        generate_fact_and_prompt(topic_hint=forced_topic)

    def test_invalid_json_raises(self, monkeypatch):
        broken = MagicMock()
        broken.message.content = "{ garbage text }"  # triggers Pydantic parse error
        monkeypatch.setattr(client, "chat", lambda **_: broken)

        with pytest.raises(Exception):  # broad catch — real schema errors vary
            generate_fact_and_prompt()


# ── Topic Rotation Tests ────────────────────────────────────────────────

class TestPickTopic:
    """Verify topic selection and recent-history avoidance."""

    @pytest.fixture(autouse=True)
    def isolate_state(self, tmp_path):
        """Point _STATE_PATH into a temp dir so tests don't touch real data."""
        import roman_empire_ig.fact_generator as mod
        old = mod._STATE_PATH
        mod._STATE_PATH = tmp_path / "recent_topics.json"
        yield
        mod._STATE_PATH = old

    def test_hint_in_pool_is_forced(self):
        assert _pick_topic("gladiators") == "gladiators"

    def test_hint_not_in_pool_is_ignored(self, monkeypatch):
        import random as rnd_mod
        monkeypatch.setattr(rnd_mod, "choice", lambda lst: "aqueducts")
        # "nonexistent" is picked as a hint, but since it's not in TOPICS
        # the function falls through to random selection. We force random.choice
        # above so we can deterministically assert aqueducts was returned instead.
        result = _pick_topic("nonexistent-thing")
        assert result == "aqueducts"


# ── Pydantic Schema Tests ───────────────────────────────────────────────

class TestSchemaValidation:
    def test_schema_parses_valid_json(self):
        raw = {
            "topic": "public baths",
            "fact": "Rich Romans used scented oils in the thermae.",
            "scene_description": "Steam rises from marble pools while togas hang nearby...",
        }
        model = _LLMSceneResponse(**raw)
        assert model.topic == raw["topic"]

    def test_schema_rejects_missing_fields(self):
        with pytest.raises(Exception):
            _LLMSceneResponse(topic="x", fact="y")  # missing scene_description


# ── Live Run (OPTIONAL — requires local Ollama) ────────────────────────

class TestIntegration:
    """Talks to the real Ollama model. Skipped automatically if unreachable."""

    def test_live_generate_and_print(self):
        """
        Actually generates a fact via Ollama and prints the structured reply
        straight to the console so developers can eyeball quality. Wrap with -k skip_live
        or use --ignore tests/test_fact_generation.py in CI.
        """
        try:
            resp = urllib.request.urlopen("http://localhost:11434", timeout=2)
            if resp.status != 200:
                pytest.skip(f"Ollama did not report healthy (HTTP {resp.status})")
        except Exception:
            pytest.skip("Ollama does not appear to be running on localhost:11434")

        print("\n" + "=" * 60, file=sys.stderr)
        print("LIVE OLLAMA CALL — generating a Roman fact...", file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)

        content = generate_fact_and_prompt()

        print(f"\nFact:      {content.fact}", file=sys.stderr)
        print(f"Image Prompt:\n{content.image_prompt}\n", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        # Basic sanity — Ollama might still return garbage if model is overloaded
        assert len(content.fact) > 10, "LLM returned an unexpectedly short fact."
        assert len(content.image_prompt) > 50


# ── Composer Tests (kept since tests still live in pipeline area) ────────

class TestComposer:
    def test_compose_returns_image(self):
        from PIL import Image as PILImage
        from roman_empire_ig.composer import compose

        dummy = PILImage.new("RGB", (1080, 1080), color=(100, 60, 30))
        result = compose(dummy, "A short fact about Rome.")
        assert isinstance(result, PILImage.Image)
        assert result.size == (1080, 1080)
