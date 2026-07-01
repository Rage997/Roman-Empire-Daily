"""
fact_generator.py
Generates a Roman daily-life fact + image-generation prompt using a local
Ollama model (qwen3.6:27b). Replaces the hardcoded test stubs.

Design notes:
- The LLM only generates the *creative content* (topic, fact, scene
  description). The photographic style boilerplate (camera, lighting,
  "no CGI", etc.) is a fixed template appended in code — this keeps every
  generated image stylistically consistent and gives the model a shorter,
  more reliable generation target.
- Output is constrained via Ollama's structured-outputs `format` parameter
  (a JSON schema derived from a Pydantic model), not free-text parsing.
  This also suppresses "thinking" mode automatically: JSON-grammar
  constrained sampling and <think> token emission are mutually exclusive
  in Ollama, so there's no need to separately disable reasoning traces.
- Topics are drawn from a curated list (factually safer than free
  generation) with simple recent-history avoidance to reduce repeats.
- Previously generated facts are persisted per topic and injected into
  the prompt as explicit exclusions, preventing near-duplicate facts
  across runs even for the same topic.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from ollama import Client
from pydantic import BaseModel, Field

client = Client(host="http://localhost:11434")

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "qwen3.6:27b"

# Unified state file: tracks recent topics and all past facts per topic.
_STATE_PATH = Path(__file__).parent / "data" / "generator_state.json"
_RECENT_HISTORY_SIZE = 8
_MAX_FACTS_PER_TOPIC = 10  # how many past facts to inject as exclusions per prompt

# Curated topic pool. Keeping this list-driven (rather than letting the LLM
# invent topics freely) constrains it to themes we've vetted for being
# real, well-attested aspects of Roman daily life — reduces hallucination
# risk on niche/invented "facts".
TOPICS = [
    "gladiators",
    "aqueducts",
    "public baths",
    "the Roman forum marketplace",
    "a legionary camp at dusk",
    "a wealthy domus household",
    "an insula tenement street",
    "a harbor at Ostia",
    "a provincial bread bakery",
    "a Roman school (ludus)",
    "a wine tavern (thermopolium)",
    "a textile/dye workshop",
    "a triumphal procession",
    "a rural villa during harvest",
    "a Roman funeral procession",
    "chariot racing at the Circus Maximus",
]

# Fixed photographic style appended to every generated scene description.
# Keeping this in code (not LLM-generated) guarantees consistent output
# style across every image, regardless of topic.
STYLE_SUFFIX = (
    " Photographed on a Hasselblad H6D-400C, 50mm lens, f/2.8, ISO 200, "
    "golden hour natural light, RAW photo, photorealistic, 8K, ultra detailed, "
    "no CGI, no illustration, no painting, hyperrealistic skin and material textures, "
    "avoid illustration style, avoid CGI, avoid painterly look, avoid anachronisms."
)


@dataclass
class RomanContent:
    fact: str
    image_prompt: str


class _LLMSceneResponse(BaseModel):
    """Schema the model's JSON output is constrained to."""

    topic: str = Field(description="The Roman daily-life topic this content covers")
    fact: str = Field(
        description="One factual, historically attested sentence about the topic"
    )
    scene_description: str = Field(
        description=(
            "A vivid, detailed visual description of the scene: setting, people, "
            "actions, objects, lighting. No camera/photography jargon — that is "
            "added separately."
        )
    )


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if not _STATE_PATH.exists():
        return {"recent_topics": [], "facts_by_topic": {}}
    try:
        return json.loads(_STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read state, starting fresh: %s", e)
        return {"recent_topics": [], "facts_by_topic": {}}


def _save_state(state: dict) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(state, indent=2))
    except OSError as e:
        logger.warning("Could not persist state: %s", e)


# ---------------------------------------------------------------------------
# Topic selection
# ---------------------------------------------------------------------------

def _pick_topic(topic_hint: str = "", recent_topics: list[str] = []) -> str:
    if topic_hint:
        return topic_hint
    available = [t for t in TOPICS if t not in recent_topics] or TOPICS
    return random.choice(available)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(topic: str, previous_facts: list[str]) -> str:
    exclusion_block = ""
    if previous_facts:
        formatted = "\n".join(f"- {f}" for f in previous_facts)
        exclusion_block = (
            f"\n\nIMPORTANT: The following facts about this topic have already been "
            f"used. Do NOT generate anything similar to these — pick a different "
            f"angle entirely:\n{formatted}\n"
        )

    return (
        f"Generate factual, historically grounded content about Roman daily life, "
        f"specifically: {topic}.\n\n"
        "Provide:\n"
        "1. One concise, verifiably true fact about this topic from Roman history.\n"
        "2. A vivid visual scene description suitable for an image generation model "
        "— describe the setting, the people present, what they are doing, objects, "
        "architecture, and lighting. Be specific and concrete. Do not include camera "
        "or photography terminology; that will be added separately. Avoid anachronisms.\n"
        f"{exclusion_block}"
        "Respond with JSON only, matching the required schema."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_fact_and_prompt(topic_hint: str = "") -> RomanContent:
    """
    Generate a Roman daily-life fact + image prompt via a local Ollama model.

    Args:
        topic_hint: If provided, forces that topic (free-form string accepted).
                    Otherwise a topic is chosen automatically from TOPICS,
                    avoiding recently used ones.

    Returns:
        RomanContent with .fact and .image_prompt populated.

    Raises:
        RuntimeError: if the model response fails schema validation.
    """
    state = _load_state()
    topic = _pick_topic(topic_hint, state["recent_topics"])

    previous_facts = state["facts_by_topic"].get(topic, [])[-_MAX_FACTS_PER_TOPIC:]
    logger.info(
        "Generating content for topic '%s' (avoiding %d known facts)",
        topic,
        len(previous_facts),
    )

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": _build_prompt(topic, previous_facts)}],
        format=_LLMSceneResponse.model_json_schema(),
        options={"temperature": 0.7},
        keep_alive=0,  # unload from VRAM immediately after this response
    )

    try:
        parsed = _LLMSceneResponse.model_validate_json(response.message.content)
    except Exception as e:
        raise RuntimeError(
            f"Ollama returned content that failed schema validation: {e}\n"
            f"Raw response: {response.message.content!r}"
        ) from e

    # Persist updated state
    recent = state["recent_topics"]
    recent.append(topic)
    state["recent_topics"] = recent[-_RECENT_HISTORY_SIZE:]

    facts = state["facts_by_topic"].get(topic, [])
    facts.append(parsed.fact)
    state["facts_by_topic"][topic] = facts  # full history kept; no cap on disk

    _save_state(state)

    image_prompt = parsed.scene_description.strip() + STYLE_SUFFIX
    logger.info("Generated fact for '%s': %s", topic, parsed.fact)

    return RomanContent(fact=parsed.fact, image_prompt=image_prompt)