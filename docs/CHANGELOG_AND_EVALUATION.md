# Diffusive Evolution Log & Workflow Evaluation Traces

A chronological record of architectural milestones, upstream compatibility fixes, and end-to-end workflow evaluations.

---

## 1. Architectural Evolution Log

### Phase 1: Initial `uv` Workspace & `inference_ifTalk.py` (2026-10-20 – 2026-10-21)
- Set up Python 3.12 `uv` environment (`pyproject.toml`, `uv.lock`) with PyTorch `2.14.1+cu132` and `comfyui==0.41.0.1`.
- Created headless Playwright converter ([scripts/convert.py](file:///home/tmdt-admin/comfy/scripts/convert.py)), model/custom-node auto-downloaders (`scripts/download_models.sh`, `scripts/download_custom_nodes.sh`), and video comparison utilities (`scripts/diff_masks.py`, `scripts/stack_video.py`).
- Compiled and parameterized `scripts/inference_ifTalk.py` (Wan2.1 + InfiniteTalk lip-sync V2V pipeline) with `--dry-run` support and initial `pytest` suite.

### Phase 2: `qedit.json` Compilation & Upstream Bug Fixes (2026-10-22)
While compiling and testing [workflows/qedit.json](file:///home/tmdt-admin/comfy/workflows/qedit.json) (Qwen-Image GGUF text-to-image), three upstream incompatibilities between `pydn/ComfyUI-to-Python-Extension` and `comfyui==0.41.0.1` were identified and permanently fixed:
1. **Setuptools Flat-Layout Build Failure**:
   - *Issue*: `uv pip install -e` failed with `Multiple top-level packages discovered in a flat-layout: ['js', 'images', 'comfyui_to_python']`.
   - *Fix*: Added explicit `[build-system]` and `packages = ["comfyui_to_python", "comfyui_to_python.generator"]` in `packages/comfyui-to-python/pyproject.toml`.
2. **`TypeError: cannot pickle '_thread.RLock' object` on `/saveasscript`**:
   - *Issue*: `copy.deepcopy(self.node_class_mappings)` in `comfyui_to_python/app.py` crashed because `NODE_CLASS_MAPPINGS` in ComfyUI `0.41.0.1` is backed by an execution context proxy holding a `_thread.RLock`.
   - *Fix*: Replaced `copy.deepcopy(...)` with a shallow dictionary snapshot `dict(self.node_class_mappings)`.
3. **`ImportError: cannot import name 'init_extra_nodes' from 'nodes'`**:
   - *Issue*: Legacy `init_extra_nodes` does not exist in packaged ComfyUI when importing custom nodes like `UnetLoaderGGUF`.
   - *Fix*: Updated `import_custom_nodes()` and `get_node_class_mappings()` in `comfyui_to_python/node_runtime.py` to use `import_all_nodes_in_workspace()` and refresh `self.node_class_mappings` in `app.py`.

### Phase 3: Static In-Repo Core (`packages/`) & Clean Layout (2026-10-22 – 2026-10-23)
- Vendored `comfyui==0.41.0.1` into `packages/comfyui/` and `comfyui-to-python-extension==2.1.0` into `packages/comfyui-to-python/` as static, Git-tracked packages installed via `[tool.uv.sources]`.
- Added `@ <commit_sha>` commit pinning to `scripts/download_custom_nodes.sh` and `scripts/custom_nodes_links.txt`.
- Tracked all 33 standard `models/*/` subdirectories via `.gitkeep` while ignoring heavy weight files.
- Removed non-standard `scratch/` directory and consolidated documentation into `docs/ARCHITECTURE.md`, `docs/CHANGELOG_AND_EVALUATION.md`, and `docs/COMMAND.txt`.

---

## 2. Workflow Evaluation Traces

### Trace 1: `workflows/qedit.json` → `scripts/inference_qedit.py` (Qwen-Image GGUF T2I)
- **Frontend JSON**: [workflows/qedit.json](file:///home/tmdt-admin/comfy/workflows/qedit.json)
- **Serialized Backend API JSON**: [workflows/qedit_api.json](file:///home/tmdt-admin/comfy/workflows/qedit_api.json)
- **Compiled Standalone Script**: [scripts/inference_qedit.py](file:///home/tmdt-admin/comfy/scripts/inference_qedit.py)
- **Nodes (9 total)**:
  - Custom (`ComfyUI-GGUF @ b3ec875a68d94b758914fd48d30571d953bb7a54`): `UnetLoaderGGUF` (Node 71)
  - Core (`comfy-core`): `CLIPLoader` (Node 38), `VAELoader` (Node 39), `CLIPTextEncode` (Nodes 6 & 7), `EmptySD3LatentImage` (Node 58), `ModelSamplingAuraFlow` (Node 66), `KSampler` (Node 3), `VAEDecode` (Node 8), `SaveImage` (Node 60)
- **Model Weights**:
  - `models/unet/qwen-image-Q8_0.gguf` (20.4 GB)
  - `models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` (8.74 GB)
  - `models/vae/qwen_image_vae.safetensors` (242 MB)
- **Execution & Benchmark**:
  - Hardware: NVIDIA RTX A4000 (16 GB VRAM) + 31 GB System RAM (`DynamicVRAM` enabled)
  - Dry-Run (`--dry-run`): Passed (`All 9 required nodes = True`)
  - Full Inference (`--batch_size 1 --steps 10 --cfg 2.0 --width 1024 --height 768`): Completed in **103.92s**, output saved to `output/qedit_test_00001_.png` (715 KB).

### Trace 2: `scripts/inference_ifTalk.py` (Wan2.1 + InfiniteTalk Lip-Sync V2V)
- **Compiled Standalone Script**: [scripts/inference_ifTalk.py](file:///home/tmdt-admin/comfy/scripts/inference_ifTalk.py)
- **Key Nodes**: `WanVideoModelLoader`, `WanVideoSampler`, `MultiTalkModelLoader`, `WanVideoImageToVideoMultiTalk`, `VHS_LoadVideo`, `VHS_VideoCombine`
- **Verification**: Validated via `tests/test_inference_iftalk.py` (`--help`, CLI `--dry-run`, and programmatic `main(dry_run=True)`).
