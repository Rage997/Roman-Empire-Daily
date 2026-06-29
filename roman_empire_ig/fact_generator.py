"""
fact_generator.py
Returns a hardcoded Roman fact + image prompt for testing image generation.
Replace this with a real LLM call (local or API) when ready.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RomanContent:
    fact: str
    image_prompt: str


# Test stubs — add more here to rotate through different scenes
_STUBS: dict[str, RomanContent] = {
    "gladiators": RomanContent(
        fact="Gladiators rarely fought to the death — they were expensive to train and crowds preferred skill over slaughter.",
        image_prompt=(
            "A cinematic scene inside the Colosseum in ancient Rome at golden hour. "
            "Two gladiators face each other in the sand arena, one a heavily armoured Secutor "
            "with a rectangular shield and short gladius, the other a nimble Retiarius holding "
            "a trident and net. Thousands of Roman spectators fill the tiered marble stands, "
            "cheering and waving coloured cloths. Shafts of late afternoon sunlight cut through "
            "the arched openings and illuminate swirling dust in the air. The sand is ochre and "
            "slightly worn. The gladiators armour gleams with bronze highlights. "
            "Photorealistic, dramatic lighting, National Geographic style, shot on 35mm."
        ),
    ),
    "aqueducts": RomanContent(
        fact="Rome's aqueducts delivered over one million cubic metres of fresh water to the city every day.",
        image_prompt=(
            "A sweeping aerial view of a massive ancient Roman aqueduct stretching across a "
            "sunlit Italian valley. The long arcade of perfectly proportioned stone arches recedes "
            "into the distance toward a hilltop city. The arches cast sharp shadows on the dry "
            "grass below. Cypress trees dot the landscape. The sky is deep blue with a few "
            "scattered white clouds. The stone is warm travertine, weathered and ancient. "
            "Photorealistic, golden hour light, cinematic wide angle, hyper-detailed."
            "Photographed on a Hasselblad H6D-400C, 50mm lens, f/2.8, ISO 200,"
            "golden hour natural light, RAW photo, photorealistic, 8K, ultra detailed, "
            "no CGI, no illustration, no painting, hyperrealistic skin and material textures."
            "avoid illustration style, avoid CGI, avoid painterly look, avoid anachronisms."
        ),
    ),
    "default": RomanContent(
        fact="At its peak the Roman Empire was home to 70 million people — roughly 20% of the world's population.",
        image_prompt=(
            "A breathtaking panoramic view of ancient Rome at its imperial peak, seen from the "
            "Palatine Hill at sunrise. The Forum Romanum stretches below with its white marble "
            "temples, triumphal arches and colonnaded basilicas glowing in warm golden light. "
            "The Colosseum dominates the middle distance. Smoke rises from a hundred altars. "
            "Citizens in white togas move along the Sacred Way. The Tiber river gleams silver "
            "in the far background. The sky is vivid blue with dramatic clouds. "
            "Photorealistic, epic scale, cinematic, National Geographic cover quality."
        ),
    ),
}


def generate_fact_and_prompt(topic_hint: str = "") -> RomanContent:
    """
    Return a test Roman fact + image prompt.

    Args:
        topic_hint: Matches against known stubs; falls back to default.

    Returns:
        RomanContent with .fact and .image_prompt populated.
    """
    key = topic_hint.lower().strip()
    content = _STUBS.get(key, _STUBS["default"])
    logger.info("Using stub content for topic '%s': %s", key or "default", content.fact)
    return content
