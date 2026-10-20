# Diffusive

A `uv`-managed, headless ComfyUI Python workspace for running and compiling standalone AI video editing and generation pipelines.

---

## Project Structure

```text
diffusive/
├── pyproject.toml                     # uv project configuration & dependencies
├── uv.lock                            # Locked dependency versions
├── .python-version                    # Pinned Python version (3.12)
├── implementation_plan.md             # Living architecture & implementation plan
├── comfyui_to_python_walkthrough.md   # Detailed guide on Frontend JSON -> Python compilation
├── uv_pip_comfy.md                    # Reference for uv pip ComfyUI installation
│
├── models/                            # Downloaded model weights (diffusion_models, vae, loras, etc.)
├── custom_nodes/                      # Git-cloned ComfyUI custom nodes
├── workflows/                         # ComfyUI JSON workflows (Frontend UI & API formats)
├── input/                             # Input videos & audio files
├── output/                            # Generated videos, diff masks, and stacked comparison videos
├── scratch/                           # Intermediate serialized API & frontend JSONs
│
└── scripts/
    ├── inference_ifTalk.py            # Standalone Wan2.1 + InfiniteTalk V2V / lip-sync script
    ├── convert.py                     # Headless Playwright converter (Workflow JSON -> Python script)
    ├── diff_masks.py                  # Frame-by-frame RGB difference mask & lossless MP4 generator
    ├── stack_video.py                 # Side-by-side batch video comparator (Original | Edited | Diff)
    ├── download_models.sh             # Downloads model weights defined in model_links.txt
    ├── model_links.txt                # Model URLs & target subfolders
    ├── download_custom_nodes.sh       # Clones custom nodes & installs their requirements into .venv
    └── custom_nodes_links.txt         # Custom node GitHub repository URLs
```

---

## Quick Start

### 1. Environment Setup

Install `uv` if not already installed, then sync the project environment:

```bash
uv sync --inexact
```

If you plan to compile frontend ComfyUI workflows into standalone Python scripts with `scripts/convert.py`, install the headless Chromium binary once:

```bash
uv run playwright install chromium
```

### 2. Download Custom Nodes & Models

Clone all configured custom nodes into `./custom_nodes/` and install their Python dependencies into `.venv`:

```bash
./scripts/download_custom_nodes.sh
```

Download all required model weights into `./models/`:

```bash
./scripts/download_models.sh
```

---

## Usage

### Run a Standalone Python Workflow Headlessly

Execute compiled ComfyUI Python scripts directly with `uv run`:

```bash
uv run python scripts/inference_ifTalk.py \
  --video input/sample.mp4 \
  --audio input/sample.wav \
  --positive_prompt "A person speaking clearly to the camera." \
  --output_prefix ./output/generated \
  --device cuda:0 \
  --steps 4
```

View all CLI options for the workflow:

```bash
uv run python scripts/inference_ifTalk.py --help
```

---

### Convert a ComfyUI JSON Workflow to a Standalone Python Script

1. Start the ComfyUI server (which loads `custom_nodes/ComfyUI-to-Python-Extension`):
   ```bash
   uv run comfyui serve --guess-settings
   ```
2. In another terminal, run the headless Playwright converter:
   ```bash
   uv run python scripts/convert.py \
     --input workflows/kiwiEdit.json \
     --output scripts/inference_kiwiEdit.py
   ```
3. Stop the ComfyUI server and run your newly generated standalone script directly:
   ```bash
   uv run python scripts/inference_kiwiEdit.py
   ```

---

### Post-Processing & Video Comparison

#### Batch Side-by-Side Video Stacking (`Original | Edited | Difference`)

```bash
uv run python scripts/stack_video.py \
  --json-dir output/ \
  --video-dir input/ \
  --output-dir output/stacks
```

Useful flags:
* `--latest N`: Process only the `N` most recently generated videos.
* `--random N`: Process `N` randomly sampled videos.
* `--limit N`: Process up to `N` videos.
* `--dry-run`: Inspect matched video pairs and resolutions without rendering.
