# Installing ComfyUI with `uv`

Install ComfyUI directly from the published index without cloning any repository.

## 1. Create and Activate Virtual Environment

```bash
mkdir -p ~/comfy && cd ~/comfy
uv venv --python 3.12
source .venv/bin/activate
```

## 2. Install ComfyUI

`--torch-backend=auto` automatically detects your GPU and installs the matching PyTorch version:

```bash
uv pip install --torch-backend=auto --extra-index-url https://nodes.appmana.com/simple/ comfyui
```

## 3. Run ComfyUI

```bash
comfyui serve --guess-settings
```

Access the UI at `http://127.0.0.1:8188`.
