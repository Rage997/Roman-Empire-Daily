"""
cli.py
Command-line interface for the Roman Empire Instagram bot.

Usage examples:
  python -m roman_empire_ig                        # random topic
  python -m roman_empire_ig --topic "gladiators"  # specific topic
  python -m roman_empire_ig --dry-run             # skip Instagram post
  python -m roman_empire_ig --count 5             # generate 5 posts
"""

import logging
import sys
import time

import click

from .config import settings
from .pipeline import run


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@click.command()
@click.option(
    "--topic",
    default="",
    show_default=True,
    help="Optional Roman topic hint (e.g. 'aqueducts', 'Julius Caesar').",
)
@click.option(
    "--count",
    default=1,
    show_default=True,
    help="Number of posts to generate in one run.",
)
@click.option(
    "--delay",
    default=5,
    show_default=True,
    help="Seconds to wait between posts when --count > 1.",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="Generate images locally without posting to Instagram.",
)
@click.option(
    "--log-level",
    default=settings.log_level,
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
def main(topic: str, count: int, delay: int, dry_run: bool, log_level: str) -> None:
    """Roman Empire Instagram Bot — generates AI images + facts and posts them."""
    _setup_logging(log_level)

    if dry_run:
        # Override at runtime without touching .env
        settings.auto_post = False
        click.echo("Dry-run mode — images will be saved locally, not posted.")

    for i in range(count):
        if count > 1:
            click.echo(f"\n[{i + 1}/{count}] Generating post…")
        run(topic_hint=topic)
        if i < count - 1:
            time.sleep(delay)

    click.echo("\nAll done! Check the output/ directory.")


if __name__ == "__main__":
    main()
