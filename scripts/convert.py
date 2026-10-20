import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from playwright.async_api import async_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKFLOW_PATH = PROJECT_ROOT / "workflows" / "kiwiEdit.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "scripts" / "void_api.py"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
DEFAULT_COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")

UV_RUNTIME_HEADER = '''def get_comfyui_path() -> str:
    """Return the configured ComfyUI workspace path, preferring COMFYUI_PATH when set."""
    comfyui_path = os.environ.get("COMFYUI_PATH")
    if comfyui_path:
        return comfyui_path
    found = find_path("ComfyUI")
    if found:
        return found
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if os.path.isdir(os.path.join(project_root, "models")) or os.path.isdir(os.path.join(project_root, "custom_nodes")):
        return project_root
    return os.getcwd()


def find_path(name: str, path: str = None) -> str:
    """Recursively search parent folders until the named entry is found."""
    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))

    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None

    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """Add the ComfyUI workspace or checkout to sys.path."""
    comfyui_path = get_comfyui_path()
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        if comfyui_path in sys.path:
            sys.path.remove(comfyui_path)
        sys.path.insert(0, comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """Load ComfyUI extra model paths configuration when available."""
    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is None:
        return
    try:
        from comfy.cmd.extra_model_paths import load_extra_path_config
    except ImportError:
        try:
            from main import load_extra_path_config
        except ImportError:
            from utils.extra_config import load_extra_path_config
    load_extra_path_config(extra_model_paths)


def bootstrap_comfyui_runtime() -> None:
    """Initialize ComfyUI runtime (supports both uv-packaged comfyui and git checkouts)."""
    add_comfyui_directory_to_sys_path()
    comfyui_path = get_comfyui_path()

    try:
        from dataclasses import replace
        from comfy.cli_args_types import Configuration
        from comfy.component_model.folder_path_types import FolderNames
        from comfy.component_model.setup import setup_pre_torch, setup_post_torch
        from comfy.cmd.folder_paths import init_default_paths
        from comfy.execution_context import comfyui_execution_context, current_execution_context

        config = Configuration(
            cwd=comfyui_path,
            base_directory=comfyui_path,
            guess_settings=True,
        )
        setup_pre_torch(config)
        folder_names = FolderNames(is_root=True)
        init_default_paths(folder_names, config, replace_existing=True)
        comfyui_execution_context.set(
            replace(current_execution_context(), configuration=config, folder_names_and_paths=folder_names)
        )
        setup_post_torch(config)
        return
    except ImportError:
        pass

    import comfy.options

    comfy.options.enable_args_parsing()

    from comfy.cli_args import args

    if os.name == "nt":
        os.environ["MIMALLOC_PURGE_DELAY"] = "0"

    if args.default_device is not None:
        default_dev = args.default_device
        devices = list(range(32))
        devices.remove(default_dev)
        devices.insert(0, default_dev)
        devices = ",".join(map(str, devices))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
        os.environ["HIP_VISIBLE_DEVICES"] = str(devices)

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.cuda_device)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.cuda_device)

    if args.oneapi_device_selector is not None:
        os.environ["ONEAPI_DEVICE_SELECTOR"] = args.oneapi_device_selector

    if args.deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import cuda_malloc

    if "rocm" in cuda_malloc.get_torch_version_noimport():
        os.environ["OCL_SET_SVM_SIZE"] = "262144"


def cleanup_comfyui_runtime(unload_models: bool | None = None) -> None:
    """Best-effort cleanup for embedded or repeated generated-script execution."""
    import gc
    import warnings

    def run_cleanup_hook(name: str, should_run: bool = True) -> None:
        if not should_run or not hasattr(model_management, name):
            return
        cleanup_fn = getattr(model_management, name)
        try:
            cleanup_fn()
        except Exception as exc:
            warnings.warn(
                f"ComfyUI cleanup hook {name} failed during teardown: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    should_unload = unload_models
    if should_unload is None:
        should_unload = os.environ.get("COMFYUI_TOPYTHON_UNLOAD_MODELS", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    try:
        import comfy.model_management as model_management
    except ModuleNotFoundError:
        gc.collect()
        return

    run_cleanup_hook("cleanup_models_gc")
    run_cleanup_hook("unload_all_models", should_run=should_unload)
    run_cleanup_hook("soft_empty_cache")
    gc.collect()


def import_custom_nodes() -> None:
    """Initialize ComfyUI custom nodes in the exporter runtime."""
    comfyui_path = get_comfyui_path()
    if comfyui_path and comfyui_path not in sys.path:
        sys.path.insert(0, comfyui_path)

    try:
        from dataclasses import replace
        from comfy_compatibility.vanilla import prepare_vanilla_environment
        from comfy.execution_context import comfyui_execution_context, current_execution_context
        from comfy.nodes.package import import_all_nodes_in_workspace
        from comfy.nodes_context import get_nodes

        prepare_vanilla_environment()
        exported_nodes = import_all_nodes_in_workspace(raise_on_failure=False)
        comfyui_execution_context.set(
            replace(current_execution_context(), custom_nodes=exported_nodes)
        )
        get_nodes()
        nodes_shim = sys.modules.get("nodes")
        if nodes_shim is not None and hasattr(nodes_shim, "activate"):
            nodes_shim.activate()
        return
    except ImportError:
        pass

    import asyncio
    import execution
    from nodes import init_extra_nodes

    if comfyui_path in sys.path:
        sys.path.remove(comfyui_path)
    sys.path.insert(0, comfyui_path)

    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    asyncio.run(init_extra_nodes())'''


def patch_generated_script_for_uv(script_text: str) -> str:
    """Replace standard ComfyUI-to-Python runtime helpers with uv-compatible helpers."""
    pattern = re.compile(
        r"def (?:get_comfyui_path|find_path)\(.*?# Workflow data",
        flags=re.DOTALL,
    )
    if pattern.search(script_text):
        return pattern.sub(UV_RUNTIME_HEADER + "\n\n\n# Workflow data", script_text, count=1)
    return script_text


async def convert_workflow(
    workflow_path: Path,
    output_path: Path,
    comfyui_url: str = DEFAULT_COMFYUI_URL,
) -> None:
    if not workflow_path.exists():
        raise FileNotFoundError(f"Workflow file not found: {workflow_path}")

    print(f"Reading {workflow_path}...")
    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow_data = json.load(f)

    async with async_playwright() as p:
        print("Launching headless Chromium...")
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        page.on("console", lambda msg: print(f"CONSOLE: {msg.text}"))
        page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))

        print(f"Navigating to {comfyui_url}...")
        await page.goto(comfyui_url)

        print("Waiting for page load state to complete...")
        await page.wait_for_load_state("networkidle")

        print("Waiting for ComfyUI app instance to initialize...")
        await page.wait_for_function(
            "typeof app !== 'undefined' && app.graph !== undefined && app.graphToPrompt !== undefined"
        )

        print("Serializing graph to API prompt...")
        api_data = await page.evaluate(
            """
            async (workflowData) => {
                app.graph.clear();
                await app.loadGraphData(workflowData);
                await new Promise(resolve => setTimeout(resolve, 1000));
                const p = await app.graphToPrompt();
                const frontendWorkflow = p.workflow ?? app.graph.serialize();
                return {
                    api_output: p.output,
                    frontend_workflow: frontendWorkflow
                };
            }
            """,
            workflow_data,
        )

        SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        api_out_path = SCRATCH_DIR / "api_prompt.json"
        frontend_out_path = SCRATCH_DIR / "frontend_workflow.json"
        print(f"Saving API prompt to: {api_out_path}")
        with open(api_out_path, "w", encoding="utf-8") as f:
            json.dump(api_data["api_output"], f, indent=2, ensure_ascii=False)

        print(f"Saving frontend workflow copy to: {frontend_out_path}")
        with open(frontend_out_path, "w", encoding="utf-8") as f:
            json.dump(api_data["frontend_workflow"], f, indent=2, ensure_ascii=False)

        print("Posting to /saveasscript to generate python script...")
        payload = json.dumps(
            {
                "name": f"{workflow_path.stem}_api.json",
                "workflow": json.dumps(api_data["api_output"], ensure_ascii=False),
                "frontend_workflow": json.dumps(api_data["frontend_workflow"], ensure_ascii=False),
            }
        )

        response = await page.request.post(
            f"{comfyui_url.rstrip('/')}/saveasscript",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        if response.status != 200:
            text = await response.text()
            print(f"Conversion failed with status {response.status}: {text}")
            print(f"API prompt saved to {api_out_path} — inspect which node is missing 'class_type'.")
            await browser.close()
            raise SystemExit(1)

        python_script = await response.text()
        python_script = patch_generated_script_for_uv(python_script)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(python_script)

        print(f"Stand-alone script successfully written to: {output_path}")
        await browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a ComfyUI frontend/API workflow JSON into a standalone uv-runnable Python script."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_WORKFLOW_PATH,
        help=f"Input workflow JSON path (default: {DEFAULT_WORKFLOW_PATH})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output Python script path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_COMFYUI_URL,
        help=f"Running ComfyUI server URL (default: {DEFAULT_COMFYUI_URL})",
    )
    args = parser.parse_args()
    asyncio.run(convert_workflow(args.input.resolve(), args.output.resolve(), args.url))


if __name__ == "__main__":
    main()
