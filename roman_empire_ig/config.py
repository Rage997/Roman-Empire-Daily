"""
Central configuration — all tuneable values and secrets live here.
Values are loaded from environment variables (or a .env file via python-dotenv).
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Anthropic ─────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # ── Diffusion model ───────────────────────────────────────────────────────
    # HuggingFace repo id or absolute path to local weights
    flux_model_id: str = "black-forest-labs/FLUX.1-dev"
    # bfloat16 fits a 3090 comfortably; switch to float16 if you hit OOM
    torch_dtype: str = "bfloat16"
    # Number of diffusion steps (28 = average quality; 4 = schnell turbo; 50 = good quality)
    num_inference_steps: int = 50
    # Guidance scale (3.5–7 works well for FLUX)
    guidance_scale: float = 3.5
    # Output resolution — 1080x1080 for square IG posts
    image_width: int = 1024
    image_height: int = 1024

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: Path = Path("output")
    # Overlay semi-transparent black bar so white text is always readable
    text_overlay: bool = True
    overlay_opacity: float = 0.55      # 0.0 = invisible, 1.0 = solid black
    font_size: int = 38
    font_color: str = "#FFFFFF"

    # ── Instagram (Graph API) ─────────────────────────────────────────────────
    instagram_access_token: str = ""
    instagram_account_id: str = ""
    # Set to False to generate locally without posting
    auto_post: bool = False

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = -1   # -1 = random seed each run
    log_level: str = "INFO"


# Module-level singleton — import this everywhere
settings = Settings()
