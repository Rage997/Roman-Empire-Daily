# Roman Empire Instagram Bot

This project generates AI-powered Roman history content for Instagram posts.

## Features

- Generates Roman facts and image prompts
- Creates photorealistic images using FLUX.2 
- Adds text overlays and saves to output directory
- Posts to Instagram (if configured)

## Usage

```bash
# Random topic
python -m roman_empire_ig

# Specific topic  
python -m roman_empire_ig --topic "gladiators"

# Generate multiple posts (optimization for count > 1)
python -m roman_empire_ig --topic "aqueducts" --count 5

# Dry-run mode (generate locally without posting)
python -m roman_empire_ig --dry-run
```

## Optimization Note

For `--count > 1`, the pipeline now uses an optimized approach that:
- Pre-generates all facts first
- Then generates all images using a single model load 
- Dramatically improves performance and reduces resource usage

## Fixed Issues

### Argument Parsing
- ✅ The `--count` parameter works correctly
- ✅ Multiple runs execute as expected (`--count 3` shows `[1/3]`, `[2/3]`, `[3/3]`)  
- ✅ Previously there was a logging bug that affected display but not functionality

### Performance
- ⚡ Optimized pipeline to avoid repeated model loading for multiple posts
- 💾 Reduced memory consumption through better resource management