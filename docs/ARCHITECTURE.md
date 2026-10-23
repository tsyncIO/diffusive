# Diffusive Architecture & Compilation Guide

**Diffusive** is a self-contained, `uv`-managed headless Python framework for compiling and executing ComfyUI image and video manipulation pipelines without a browser UI.

---

## 1. Core Design Philosophy: Static In-Repo Engine (`packages/`)

Upstream `ComfyUI` and `pydn/ComfyUI-to-Python-Extension` evolve rapidly and frequently introduce breaking changes. To guarantee 100% reproducibility:

1. **`packages/comfyui/` (`v0.41.0.1`)**:
   - Full Python source code of the ComfyUI core runtime (`comfy`, `comfy_api`, `comfy_api_nodes`, `comfy_compatibility`, `comfy_config`, `comfy_execution`, `comfy_extras`) tracked directly in this repository.
2. **`packages/comfyui-to-python/` (`v2.1.0`)**:
   - Full source code of the ComfyUI-to-Python AST compiler with `uv` runtime bootstrap helpers and ComfyUI `0.41+` compatibility fixes permanently baked into `node_runtime.py` and `app.py`.
3. **Editable Local Wiring ([pyproject.toml](file:///home/tmdt-admin/comfy/pyproject.toml))**:
   ```toml
   [tool.uv.sources]
   comfyui = { path = "packages/comfyui", editable = true }
   comfyui-to-python-extension = { path = "packages/comfyui-to-python", editable = true }
   ```
4. **Commit-Pinned Custom Nodes ([scripts/custom_nodes_links.txt](file:///home/tmdt-admin/comfy/scripts/custom_nodes_links.txt))**:
   - External custom nodes support `<GITHUB_URL> @ <COMMIT_SHA>` pinning via [scripts/download_custom_nodes.sh](file:///home/tmdt-admin/comfy/scripts/download_custom_nodes.sh), matching the exact `properties.ver` commit recorded inside each workflow JSON.
   - `custom_nodes/ComfyUI-to-Python-Extension` is automatically symlinked to `../packages/comfyui-to-python`.

---

## 2. Frontend JSON → Backend API JSON → Standalone Python Pipeline

ComfyUI stores workflows in two distinct JSON formats:
- **Frontend Layout JSON (`workflows/<name>.json`)**: Contains UI coordinates, groups, links, and positional `widgets_values` arrays. Cannot be executed directly by Python without serialization.
- **Backend API JSON (`workflows/<name>_api.json`)**: Contains pure execution node IDs, `class_type`, and key-value `inputs`.

```mermaid
flowchart LR
    A["Frontend Layout JSON\n(workflows/qedit.json)"] -->|1. Load in Headless Chromium| B["Playwright\n(scripts/convert.py)"]
    B -->|2. app.graphToPrompt()| C["Backend API JSON\n(workflows/qedit_api.json)"]
    C -->|3. POST /saveasscript| D["Static Compiler\n(packages/comfyui-to-python)"]
    D -->|4. AST + Black Format| E["Standalone Python Script\n(scripts/inference_qedit.py)"]
```

### How `scripts/convert.py` Works
1. Launches headless Chromium via Playwright and connects to a local ComfyUI server (`uv run comfyui serve --guess-settings --listen 127.0.0.1 --port 8189`).
2. Loads the Frontend JSON into the ComfyUI graph (`app.loadGraphData()`) and executes `app.graphToPrompt()` to serialize all widget arrays and links into the **Backend API JSON** (`workflows/<name>_api.json`).
3. Posts the API JSON to `/saveasscript` (`packages/comfyui-to-python`), which resolves topological node execution order (initializers/loaders outside the queue loop, samplers/decoders inside) and formats the standalone Python script with `black`.

---

## 3. Standalone Runtime Bootstrap

Every compiled standalone script (`scripts/inference_qedit.py`, `scripts/inference_ifTalk.py`) embeds the `uv` runtime bootstrap from `packages/comfyui-to-python/comfyui_to_python/node_runtime.py`:
- **`bootstrap_comfyui_runtime()`**: Configures `FolderNames` and `Configuration` rooted at the repository directory (`/home/tmdt-admin/comfy`), initializes `comfy-aimdo` DynamicVRAM, and binds all 33 static `models/*/` subdirectories.
- **`import_custom_nodes()`**: Loads all custom nodes in `./custom_nodes/` via `import_all_nodes_in_workspace()` and activates the `nodes.NODE_CLASS_MAPPINGS` shim.
- **`cleanup_comfyui_runtime()`**: Runs garbage collection and soft cache cleanup in a `finally:` block after inference completes.

---

## 4. Adding a New Workflow (Standard Protocol)

1. **Place the Frontend JSON** in `workflows/<name>.json`.
2. **Inspect Dependencies**: Check node `type`, `properties.cnr_id`, `properties.ver` (commit hash), and model filenames in `widgets_values`.
3. **Update Links**:
   - Add required custom node repo + `@ <commit_sha>` to `scripts/custom_nodes_links.txt` and run `./scripts/download_custom_nodes.sh`.
   - Add required model URLs + `| models/<subfolder>` to `scripts/model_links.txt` and run `./scripts/download_models.sh`.
4. **Compile to Standalone Python**:
   - Start server: `uv run comfyui serve --guess-settings --listen 127.0.0.1 --port 8189`
   - Run converter: `uv run python scripts/convert.py -i workflows/<name>.json -o scripts/inference_<name>.py --url http://127.0.0.1:8189`
5. **Verify & Log**:
   - Test with `--dry-run` and full inference, then record the trace in [CHANGELOG_AND_EVALUATION.md](file:///home/tmdt-admin/comfy/docs/CHANGELOG_AND_EVALUATION.md).
