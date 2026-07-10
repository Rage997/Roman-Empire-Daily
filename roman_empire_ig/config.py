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

    # ── Instagram (instagrapi) ──────────────────────────────────────────────
    instagram_username: str = ""
    instagram_password: str = ""
    # Session file for persistent login
    instagram_session_file: str = "instagram_session.json"
    # Set to False to generate locally without posting
    auto_post: bool = False

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = -1   # -1 = random seed each run
    log_level: str = "INFO"

    # --- Video-specific settings ────────────────────────────────────────────────
    video_width: int = 704         # vertical for reels; native Wan resolution
    video_height: int = 1280
    video_seconds: int = 15        # total reel length; will be chained as 3x5s segments
    fps: int = 24
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    negative_prompt: str = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, "
        "works, paintings, images, static, overall gray, worst quality, low "
        "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
        "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "malformed limbs, fused fingers, still picture, cluttered background, "
        "three legs, many people in the background, walking backwards, "
        "modern clothing, anachronistic objects, cars, plastic, text, watermark"
        ", morphing, warping, sudden changes, inconsistent lighting, "
        "flickering, color shifting, temporal artifacts, identity drift"
    )
    seed: int = -1  # -1 = random
    
    


# Module-level singleton — import this everywhere
settings = Settings()
