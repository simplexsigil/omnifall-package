# omnifall

Companion package for the [OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall) fall detection dataset.

## Installation

```bash
pip install omnifall
```

## Quick Start

```python
import omnifall

# Load any config (labels only, fast)
ds = omnifall.load("of-syn")

# Load with video file paths
ds = omnifall.load("of-syn", video=True)

# Add video paths to an already-loaded dataset
from datasets import load_dataset
ds = load_dataset("simplexsigil2/omnifall", "of-syn")
ds = omnifall.add_video(ds, config="of-syn")

# Prepare OOPS videos (one-time, interactive consent)
omnifall.prepare_oops()

# Then load OF-ItW with videos
ds = omnifall.load("of-itw", video=True)
```

## CLI

```bash
# Prepare OOPS videos
omnifall prepare-oops

# Show cache status
omnifall info
```

## Video Sources

| Config | Video Source |
|--------|-------------|
| of-syn, of-syn-cross-* | OF-Syn tar from HF Hub (9.1GB, auto-download) |
| of-itw | OOPS from web (~45GB stream, requires prepare_oops()) |
| of-syn-itw | OF-Syn (train/val) + OOPS (test) |
| of-sta-itw-cs/cv | Staged (train/val, no video) + OOPS (test) |

## Links

- [HuggingFace Dataset](https://huggingface.co/datasets/simplexsigil2/omnifall)
- [Paper (arXiv)](https://arxiv.org/abs/2505.19889)
- [Project Page](https://simplexsigil.github.io/omnifall/)
