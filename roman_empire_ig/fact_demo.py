"""
Quick demo script — generates a single Roman fact via Ollama and prints
the raw LLM reply straight to the console.

Usage:
    python -m roman_empire_ig.fact_demo                  # auto-pick topic
    python -m roman_empire_ig.fact_demo --topic gladiators  # force topic
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate a Roman Empire fact via Ollama.")
    parser.add_argument("--topic", type=str, default="", help="Force a specific topic (see TOPICS list in code).")
    args = parser.parse_args()

    from .fact_generator import generate_fact_and_prompt

    print("\n" + "=" * 60)
    print("🏛️  Roman Empire IG — Live Fact Generator")
    print("=" * 60)

    try:
        content = generate_fact_and_prompt(topic_hint=args.topic or "")
        print(f"\n📜 Fact:\n{content.fact}\n")
        print(f"📸 Image Prompt:\n{content.image_prompt}")
    except RuntimeError as exc:
        logger.error("Generation failed: %s", exc)
        sys.exit(1)

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
