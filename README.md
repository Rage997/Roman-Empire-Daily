# 🏛️ Roman Empire Instagram Bot

Automated pipeline that generates Roman Empire facts with AI and turns them into cinematic images using a local FLUX model — then (optionally) posts them to Instagram.

```
Ollama (local GPU)  →  fact + image prompt
         ↓
FLUX (local GPU)  →  1080×1080 image
         ↓
Pillow composer   →  text overlay
         ↓
Instagram Graph API  →  published post
```

---

## Requirements

- Python 3.11+
- NVIDIA GPU with 24 GB VRAM (RTX 3090 / 4090) — FLUX.1-dev runs at full BF16
- CUDA 12.1 drivers
- ollama running on port 11434
- (Optional) Instagram Business account + Facebook App for posting

---

## Setup

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd roman-empire-ig
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install PyTorch with CUDA **first**

```bash
pip install -r requirements-torch.txt
```

> If you're on CUDA 11.8, edit `requirements-torch.txt` and swap the index URL.

### 4. Install the rest of the project

```bash
pip install -e ".[dev]"
```

The `-e` flag installs in editable mode — changes to source files take effect immediately.

### 5. Configure secrets

```bash
cp .env.example .env
# Edit .env and fill in at minimum ANTHROPIC_API_KEY
```

---

## Usage

```bash
# Generate one post (random Roman topic), save locally
python -m roman_empire_ig

# Specific topic
python -m roman_empire_ig --topic "aqueducts"

# Generate 5 posts with a 10-second delay between each
python -m roman_empire_ig --count 5 --delay 10

# Enable posting to Instagram (requires token in .env)
python -m roman_empire_ig --topic "gladiators"
# (set AUTO_POST=true in .env first)

# Verbose debug output
python -m roman_empire_ig --log-level DEBUG
```

Or use the installed script:

```bash
roman-ig --topic "Julius Caesar"
```

---

## Project layout

```
roman_empire_ig/
├── roman_empire_ig/
│   ├── __init__.py
│   ├── __main__.py        # python -m roman_empire_ig entry point
│   ├── cli.py             # Click CLI
│   ├── config.py          # Pydantic settings (reads .env)
│   ├── fact_generator.py  # Claude API → fact + image prompt
│   ├── image_generator.py # diffusers FLUX pipeline
│   ├── composer.py        # Pillow text overlay
│   ├── pipeline.py        # Orchestrator
│   └── poster.py          # Instagram Graph API
├── tests/
│   └── test_pipeline.py
├── output/                # Generated images land here (git-ignored)
├── .env.example
├── .gitignore
├── pyproject.toml         # Single source of truth for packaging + tooling
├── requirements-torch.txt # PyTorch + CUDA (install before everything else)
└── README.md
```

---

## Development

```bash
# Run tests
pytest

# Lint + format
ruff check .
ruff format .

# Type check
mypy roman_empire_ig/
```

---

## Instagram posting

The Graph API requires a publicly reachable image URL.
The `poster.py` module intentionally raises `NotImplementedError` until you add an
image hosting step (S3, GCS, Cloudflare R2, etc.).

Steps to enable:
1. Upload the generated JPEG to object storage and get a public URL.
2. Pass that URL to the Graph API container endpoint.
3. Set `AUTO_POST=true` in `.env`.

Full guide: https://developers.facebook.com/docs/instagram-api/guides/content-publishing

---

## Using a local model checkpoint instead of HuggingFace

If you've already downloaded FLUX weights locally, point the config to your path:

```env
FLUX_MODEL_ID=/mnt/models/FLUX.1-dev
```

---

## License

MIT