# Simplified Restructuring Plan: Flat `uv` Project + ComfyUI-to-Python Standalone Compiler

This plan keeps the repository **flat and simple** so you run everything via `uv run python scripts/<script>.py`, and ensures the **ComfyUI JSON → Standalone Python Script compiler** ([scripts/convert.py](file:///home/tmdt-admin/comfy/scripts/convert.py) + `ComfyUI-to-Python-Extension`) works end-to-end with the `uv`-installed `comfyui` package.

---

## 1. Current Directory Structure

All workspace folders live at `/home/tmdt-admin/comfy`, and all runnable scripts live in `scripts/`:

```text
/home/tmdt-admin/comfy/
├── pyproject.toml                     # uv project file (dependencies + AppMana & PyTorch cu132 indexes)
├── uv.lock                            # Locked environment for `uv run`
├── .python-version                    # Pinned to 3.12
├── .venv/                             # Managed by uv (Python 3.12 + torch 2.14.1+cu132 + comfyui 0.41.0.1)
├── implementation_plan.md             # Living project plan & status tracker
│
├── models/                            # Downloaded model weights (diffusion_models, vae, loras, etc.)
├── custom_nodes/                      # Git-cloned custom nodes (including ComfyUI-to-Python-Extension)
├── workflows/                         # Input ComfyUI JSON workflows (e.g. kiwiEdit.json, void.json)
├── scratch/                           # Intermediate serialized API & frontend JSONs from convert.py
├── input/                             # Input videos / audio files
├── output/                            # Generated outputs, diff masks, stacked videos
│
└── scripts/                           # Standalone scripts (run via `uv run python scripts/...`)
    ├── convert.py                     # Headless Playwright converter (JSON -> standalone .py script)
    ├── inference_ifTalk.py            # Standalone InfiniteTalk workflow (compiled via ComfyUI-to-Python)
    ├── diff_masks.py                  # Frame difference mask & lossless MP4 generator
    ├── stack_video.py                 # Side-by-side batch video stacker (Original | Edited | Diff)
    ├── download_models.sh             # Downloads models from model_links.txt -> ../models/
    ├── model_links.txt                # Model URLs & target folders
    ├── download_custom_nodes.sh       # Clones nodes from custom_nodes_links.txt -> ../custom_nodes/
    └── custom_nodes_links.txt         # Cleaned list of custom node GitHub URLs
```

---

## 2. How the ComfyUI-to-Python Standalone Conversion Works in `uv`

As documented in [comfyui_to_python_walkthrough.md](file:///home/tmdt-admin/comfy/comfyui_to_python_walkthrough.md), converting a ComfyUI Frontend JSON workflow into a standalone Python script (like [scripts/inference_ifTalk.py](file:///home/tmdt-admin/comfy/scripts/inference_ifTalk.py)) follows this flow:

```mermaid
flowchart LR
    A["workflows/my_workflow.json\n(Frontend UI or API JSON)"] -->|1. Load in headless Chromium| B["scripts/convert.py\n(Playwright)"]
    B -->|2. app.graphToPrompt()| C["ComfyUI Server\n(uv run comfyui serve)"]
    C -->|3. POST /saveasscript| D["ComfyUI-to-Python-Extension\n(custom_nodes/)"]
    D -->|4. Patch uv runtime header| E["scripts/inference_my_workflow.py\n(Standalone Python Script)"]
```

### How It Works with the `uv`-Packaged `comfyui`
1. **Custom Node Setup (`ComfyUI-to-Python-Extension`)**:
   * [scripts/custom_nodes_links.txt](file:///home/tmdt-admin/comfy/scripts/custom_nodes_links.txt) uses `https://github.com/lucasb-eyer/ComfyUI-to-Python-Extension.git` and removes the conflicting `ComfyUI-SaveAsScript` repo.
   * [scripts/download_custom_nodes.sh](file:///home/tmdt-admin/comfy/scripts/download_custom_nodes.sh) clones custom nodes into `/home/tmdt-admin/comfy/custom_nodes`, installs their `requirements.txt` files into `.venv` via `uv pip`, and installs `ComfyUI-to-Python-Extension` in editable mode (`uv pip install -e`).
2. **Automatic `uv` Runtime Bootstrap**:
   * Both [scripts/inference_ifTalk.py](file:///home/tmdt-admin/comfy/scripts/inference_ifTalk.py) and any new script compiled by [scripts/convert.py](file:///home/tmdt-admin/comfy/scripts/convert.py) (`patch_generated_script_for_uv`) use a runtime header (`bootstrap_comfyui_runtime()`, `add_extra_model_paths()`, `import_custom_nodes()`) that initializes the `uv`-installed `comfy` package directly (configuring `models/`, `custom_nodes/`, `input/`, `output/` at `/home/tmdt-admin/comfy`, enabling `comfy-aimdo` DynamicVRAM, and activating the `nodes` shim).

---

## 3. Implementation Checklist & Status

- [x] **Step 1: Create `pyproject.toml`, `.python-version`, `uv.lock`, and workspace directories** (`models/`, `custom_nodes/`, `workflows/`, `input/`, `output/`, `scratch/`) at `/home/tmdt-admin/comfy/`.
  - Locked `comfyui==0.41.0.1`, `torch==2.14.1+cu132`, `torchvision==0.29.1+cu132`, `playwright`, `black`, `opencv-contrib-python-headless`, `numpy`, `pillow`, and `tqdm`.
- [x] **Step 2: Update `scripts/convert.py`** to remove hardcoded `/home/tmdt-user/ComfyUI` paths, add `--input` / `--output` / `--url` CLI arguments, and automatically patch generated scripts with the `uv`-compatible ComfyUI runtime bootstrap.
- [x] **Step 3: Update `scripts/inference_ifTalk.py`** to use the `uv`-compatible ComfyUI runtime bootstrap so `uv run python scripts/inference_ifTalk.py` works out of the box with `.venv`, `./models`, and `./custom_nodes`.
- [x] **Step 4: Fix `scripts/download_models.sh`, `scripts/download_custom_nodes.sh`, and `scripts/custom_nodes_links.txt`** so paths resolve to `/home/tmdt-admin/comfy/`, conflicting/duplicate nodes are cleaned up, and `ComfyUI-to-Python-Extension` is installed in editable mode.

---

## 4. Quick Usage Reference

```bash
# 1. Download models into ./models/
./scripts/download_models.sh

# 2. Clone custom nodes into ./custom_nodes/ and install their requirements into .venv
./scripts/download_custom_nodes.sh

# 3. Convert a ComfyUI JSON workflow into a standalone Python script
#    (requires `uv run comfyui serve --guess-settings` running on port 8188)
uv run python scripts/convert.py -i workflows/kiwiEdit.json -o scripts/inference_kiwiEdit.py

# 4. Run a standalone Python workflow headlessly
uv run python scripts/inference_ifTalk.py --video input/video.mp4 --audio input/audio.wav

# 5. Stack original, edited, and diff videos for evaluation
uv run python scripts/stack_video.py --json-dir output/ --video-dir input/ --output-dir output/stacks
```
