# Imports
import json
import os
import random
import sys
from typing import Sequence, Mapping, Any, Union


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Return a sequence or mapping result item by index."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def get_comfyui_path() -> str:
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
    asyncio.run(init_extra_nodes())


# Workflow data
def build_workflow() -> dict[str, Any]:
    return {
        "3": {
            "inputs": {
                "seed": 620337578931029,
                "steps": 10,
                "cfg": 2,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1,
                "model": ["66", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["58", 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
        },
        "6": {
            "inputs": {
                "text": '一个穿着"Qwen绘图"标志的T恤的中国美女正拿着黑色的马克笔面相镜头微笑。她身后的玻璃板上手写体写着 '
                "“大桥未久，小富即安。三上悠亚，六出祁山。”",
                "clip": ["38", 0],
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
        },
        "7": {
            "inputs": {
                "text": " "
                "色调艳丽，过曝，细节模糊不清，字幕，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多",
                "clip": ["38", 0],
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
        },
        "8": {
            "inputs": {"samples": ["3", 0], "vae": ["39", 0]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        "38": {
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP"},
        },
        "39": {
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"},
        },
        "58": {
            "inputs": {"width": 1024, "height": 768, "batch_size": 2},
            "class_type": "EmptySD3LatentImage",
            "_meta": {"title": "EmptySD3LatentImage"},
        },
        "60": {
            "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
        },
        "66": {
            "inputs": {
                "shift": 3.500000000000001,
                "sampling": "flow",
                "model": ["71", 0],
            },
            "class_type": "ModelSamplingAuraFlow",
            "_meta": {"title": "ModelSamplingAuraFlow"},
        },
        "71": {
            "inputs": {"unet_name": "qwen-image-Q8_0.gguf"},
            "class_type": "UnetLoaderGGUF",
            "_meta": {"title": "Unet Loader (GGUF)"},
        },
    }


def build_extra_pnginfo() -> dict[str, Any] | None:
    return {
        "workflow": {
            "id": "91f6bbe2-ed41-4fd6-bac7-71d5b5864ecb",
            "revision": 0,
            "last_node_id": 71,
            "last_link_id": 133,
            "nodes": [
                {
                    "id": 8,
                    "type": "VAEDecode",
                    "pos": [1210, 190],
                    "size": [210, 46],
                    "flags": {},
                    "order": 8,
                    "mode": 0,
                    "showAdvanced": None,
                    "inputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "samples",
                            "locked": None,
                            "name": "samples",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "LATENT",
                            "pos": None,
                            "link": 128,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "vae",
                            "locked": None,
                            "name": "vae",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "VAE",
                            "pos": None,
                            "link": 76,
                        },
                    ],
                    "outputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "IMAGE",
                            "locked": None,
                            "name": "IMAGE",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "IMAGE",
                            "pos": None,
                            "slot_index": 0,
                            "links": [110],
                        }
                    ],
                    "properties": {
                        "Node name for S&R": "VAEDecode",
                        "cnr_id": "comfy-core",
                        "ver": "0.3.48",
                        "widget_ue_connectable": {},
                    },
                },
                {
                    "id": 60,
                    "type": "SaveImage",
                    "pos": [1478.3486328125, 190.30274963378906],
                    "size": [821.0496215820312, 871.7067260742188],
                    "flags": {},
                    "order": 9,
                    "mode": 0,
                    "showAdvanced": None,
                    "inputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "images",
                            "locked": None,
                            "name": "images",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "IMAGE",
                            "pos": None,
                            "link": 110,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "filename_prefix",
                            "locked": None,
                            "name": "filename_prefix",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "STRING",
                            "widget": {"name": "filename_prefix"},
                            "link": None,
                        },
                    ],
                    "outputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": None,
                            "locked": None,
                            "name": "images",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "IMAGE",
                            "pos": None,
                            "slot_index": None,
                            "links": None,
                        }
                    ],
                    "properties": {
                        "cnr_id": "comfy-core",
                        "ver": "0.3.48",
                        "widget_ue_connectable": {},
                        "Node name for S&R": "SaveImage",
                    },
                    "widgets_values": ["ComfyUI"],
                    "widgets_values_named": {"filename_prefix": "ComfyUI"},
                },
                {
                    "id": 66,
                    "type": "ModelSamplingAuraFlow",
                    "pos": [418.2298278808594, 68.82577514648438],
                    "size": [270, 82],
                    "flags": {},
                    "order": 6,
                    "mode": 0,
                    "showAdvanced": None,
                    "inputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "model",
                            "locked": None,
                            "name": "model",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "MODEL",
                            "pos": None,
                            "link": 133,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "shift",
                            "locked": None,
                            "name": "shift",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "FLOAT",
                            "widget": {"name": "shift"},
                            "link": None,
                        },
                    ],
                    "outputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "MODEL",
                            "locked": None,
                            "name": "MODEL",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "MODEL",
                            "pos": None,
                            "slot_index": None,
                            "links": [125],
                        }
                    ],
                    "properties": {
                        "Node name for S&R": "ModelSamplingAuraFlow",
                        "cnr_id": "comfy-core",
                        "ver": "0.3.48",
                        "widget_ue_connectable": {},
                    },
                    "widgets_values": [3.500000000000001, "flow"],
                    "widgets_values_named": {
                        "shift": 3.500000000000001,
                        "sampling": "flow",
                    },
                },
                {
                    "id": 7,
                    "type": "CLIPTextEncode",
                    "pos": [413, 389],
                    "size": [425.27801513671875, 180.6060791015625],
                    "flags": {},
                    "order": 5,
                    "mode": 0,
                    "showAdvanced": None,
                    "inputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "clip",
                            "locked": None,
                            "name": "clip",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "CLIP",
                            "pos": None,
                            "link": 75,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "text",
                            "locked": None,
                            "name": "text",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "STRING",
                            "widget": {"name": "text"},
                            "link": None,
                        },
                    ],
                    "outputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "CONDITIONING",
                            "locked": None,
                            "name": "CONDITIONING",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "CONDITIONING",
                            "pos": None,
                            "slot_index": 0,
                            "links": [52],
                        }
                    ],
                    "title": "CLIP Text Encode (Negative Prompt)",
                    "properties": {
                        "Node name for S&R": "CLIPTextEncode",
                        "cnr_id": "comfy-core",
                        "ver": "0.3.48",
                        "widget_ue_connectable": {},
                    },
                    "widgets_values": [
                        " "
                        "色调艳丽，过曝，细节模糊不清，字幕，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多"
                    ],
                    "widgets_values_named": {
                        "text": " "
                        "色调艳丽，过曝，细节模糊不清，字幕，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多"
                    },
                    "color": "#322",
                    "bgcolor": "#533",
                },
                {
                    "id": 3,
                    "type": "KSampler",
                    "pos": [863, 187],
                    "size": [315, 440.51922607421875],
                    "flags": {},
                    "order": 7,
                    "mode": 0,
                    "showAdvanced": None,
                    "inputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "model",
                            "locked": None,
                            "name": "model",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "MODEL",
                            "pos": None,
                            "link": 125,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "positive",
                            "locked": None,
                            "name": "positive",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "CONDITIONING",
                            "pos": None,
                            "link": 46,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "negative",
                            "locked": None,
                            "name": "negative",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "CONDITIONING",
                            "pos": None,
                            "link": 52,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "latent_image",
                            "locked": None,
                            "name": "latent_image",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "LATENT",
                            "pos": None,
                            "link": 107,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "seed",
                            "locked": None,
                            "name": "seed",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "INT",
                            "widget": {"name": "seed"},
                            "link": None,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "steps",
                            "locked": None,
                            "name": "steps",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "INT",
                            "widget": {"name": "steps"},
                            "link": None,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "cfg",
                            "locked": None,
                            "name": "cfg",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "FLOAT",
                            "widget": {"name": "cfg"},
                            "link": None,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "sampler_name",
                            "locked": None,
                            "name": "sampler_name",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "COMBO",
                            "widget": {"name": "sampler_name"},
                            "link": None,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "scheduler",
                            "locked": None,
                            "name": "scheduler",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "COMBO",
                            "widget": {"name": "scheduler"},
                            "link": None,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "denoise",
                            "locked": None,
                            "name": "denoise",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "FLOAT",
                            "widget": {"name": "denoise"},
                            "link": None,
                        },
                    ],
                    "outputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "LATENT",
                            "locked": None,
                            "name": "LATENT",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "LATENT",
                            "pos": None,
                            "slot_index": 0,
                            "links": [128],
                        }
                    ],
                    "properties": {
                        "Node name for S&R": "KSampler",
                        "cnr_id": "comfy-core",
                        "ver": "0.3.48",
                        "widget_ue_connectable": {},
                    },
                    "widgets_values": [
                        620337578931029,
                        "randomize",
                        10,
                        2,
                        "euler",
                        "simple",
                        1,
                    ],
                    "widgets_values_named": {
                        "seed": 620337578931029,
                        "control_after_generate": "randomize",
                        "steps": 10,
                        "cfg": 2,
                        "sampler_name": "euler",
                        "scheduler": "simple",
                        "denoise": 1,
                    },
                },
                {
                    "id": 39,
                    "type": "VAELoader",
                    "pos": [20, 340],
                    "size": [330, 60],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "showAdvanced": None,
                    "inputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "vae_name",
                            "locked": None,
                            "name": "vae_name",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "COMBO",
                            "widget": {"name": "vae_name"},
                            "link": None,
                        }
                    ],
                    "outputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "VAE",
                            "locked": None,
                            "name": "VAE",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "VAE",
                            "pos": None,
                            "slot_index": 0,
                            "links": [76],
                        }
                    ],
                    "properties": {
                        "Node name for S&R": "VAELoader",
                        "cnr_id": "comfy-core",
                        "ver": "0.3.48",
                        "widget_ue_connectable": {},
                    },
                    "widgets_values": ["qwen_image_vae.safetensors"],
                    "widgets_values_named": {"vae_name": "qwen_image_vae.safetensors"},
                    "color": "#223",
                    "bgcolor": "#335",
                },
                {
                    "id": 38,
                    "type": "CLIPLoader",
                    "pos": [20, 190],
                    "size": [380, 106],
                    "flags": {},
                    "order": 1,
                    "mode": 0,
                    "showAdvanced": None,
                    "inputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "clip_name",
                            "locked": None,
                            "name": "clip_name",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "COMBO",
                            "widget": {"name": "clip_name"},
                            "link": None,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "type",
                            "locked": None,
                            "name": "type",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "COMBO",
                            "widget": {"name": "type"},
                            "link": None,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "device",
                            "locked": None,
                            "name": "device",
                            "nameLocked": None,
                            "removable": None,
                            "shape": 7,
                            "type": "COMBO",
                            "widget": {"name": "device"},
                            "link": None,
                        },
                    ],
                    "outputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "CLIP",
                            "locked": None,
                            "name": "CLIP",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "CLIP",
                            "pos": None,
                            "slot_index": 0,
                            "links": [74, 75],
                        }
                    ],
                    "properties": {
                        "Node name for S&R": "CLIPLoader",
                        "cnr_id": "comfy-core",
                        "ver": "0.3.48",
                        "widget_ue_connectable": {},
                    },
                    "widgets_values": [
                        "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                        "qwen_image",
                        "default",
                    ],
                    "widgets_values_named": {
                        "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                        "type": "qwen_image",
                        "device": "default",
                    },
                    "color": "#223",
                    "bgcolor": "#335",
                },
                {
                    "id": 71,
                    "type": "UnetLoaderGGUF",
                    "pos": [-75.64920806884766, 14.61060619354248],
                    "size": [480, 60],
                    "flags": {},
                    "order": 2,
                    "mode": 0,
                    "showAdvanced": None,
                    "inputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "unet_name",
                            "locked": None,
                            "name": "unet_name",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "COMBO",
                            "widget": {"name": "unet_name"},
                            "link": None,
                        }
                    ],
                    "outputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "MODEL",
                            "locked": None,
                            "name": "MODEL",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "MODEL",
                            "pos": None,
                            "slot_index": None,
                            "links": [133],
                        }
                    ],
                    "properties": {
                        "Node name for S&R": "UnetLoaderGGUF",
                        "cnr_id": "ComfyUI-GGUF",
                        "ver": "b3ec875a68d94b758914fd48d30571d953bb7a54",
                        "widget_ue_connectable": {},
                    },
                    "widgets_values": ["qwen-image-Q8_0.gguf"],
                    "widgets_values_named": {"unet_name": "qwen-image-Q8_0.gguf"},
                },
                {
                    "id": 58,
                    "type": "EmptySD3LatentImage",
                    "pos": [410.6574401855469, 620.5468139648438],
                    "size": [270, 106],
                    "flags": {},
                    "order": 3,
                    "mode": 0,
                    "showAdvanced": None,
                    "inputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "width",
                            "locked": None,
                            "name": "width",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "INT",
                            "widget": {"name": "width"},
                            "link": None,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "height",
                            "locked": None,
                            "name": "height",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "INT",
                            "widget": {"name": "height"},
                            "link": None,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "batch_size",
                            "locked": None,
                            "name": "batch_size",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "INT",
                            "widget": {"name": "batch_size"},
                            "link": None,
                        },
                    ],
                    "outputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "LATENT",
                            "locked": None,
                            "name": "LATENT",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "LATENT",
                            "pos": None,
                            "slot_index": None,
                            "links": [107],
                        }
                    ],
                    "properties": {
                        "Node name for S&R": "EmptySD3LatentImage",
                        "cnr_id": "comfy-core",
                        "ver": "0.3.48",
                        "widget_ue_connectable": {},
                    },
                    "widgets_values": [1024, 768, 2],
                    "widgets_values_named": {
                        "width": 1024,
                        "height": 768,
                        "batch_size": 2,
                    },
                },
                {
                    "id": 6,
                    "type": "CLIPTextEncode",
                    "pos": [415, 186],
                    "size": [422.84503173828125, 164.31304931640625],
                    "flags": {},
                    "order": 4,
                    "mode": 0,
                    "showAdvanced": None,
                    "inputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "clip",
                            "locked": None,
                            "name": "clip",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "CLIP",
                            "pos": None,
                            "link": 74,
                        },
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "text",
                            "locked": None,
                            "name": "text",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "STRING",
                            "widget": {"name": "text"},
                            "link": None,
                        },
                    ],
                    "outputs": [
                        {
                            "color_off": None,
                            "color_on": None,
                            "dir": None,
                            "label": "CONDITIONING",
                            "locked": None,
                            "name": "CONDITIONING",
                            "nameLocked": None,
                            "removable": None,
                            "shape": None,
                            "type": "CONDITIONING",
                            "pos": None,
                            "slot_index": 0,
                            "links": [46],
                        }
                    ],
                    "title": "CLIP Text Encode (Positive Prompt)",
                    "properties": {
                        "Node name for S&R": "CLIPTextEncode",
                        "cnr_id": "comfy-core",
                        "ver": "0.3.48",
                        "widget_ue_connectable": {},
                    },
                    "widgets_values": [
                        '一个穿着"Qwen绘图"标志的T恤的中国美女正拿着黑色的马克笔面相镜头微笑。她身后的玻璃板上手写体写着 '
                        "“大桥未久，小富即安。三上悠亚，六出祁山。”"
                    ],
                    "widgets_values_named": {
                        "text": '一个穿着"Qwen绘图"标志的T恤的中国美女正拿着黑色的马克笔面相镜头微笑。她身后的玻璃板上手写体写着 '
                        "“大桥未久，小富即安。三上悠亚，六出祁山。”"
                    },
                    "color": "#232",
                    "bgcolor": "#353",
                },
            ],
            "links": [
                [46, 6, 0, 3, 1, "CONDITIONING"],
                [52, 7, 0, 3, 2, "CONDITIONING"],
                [74, 38, 0, 6, 0, "CLIP"],
                [75, 38, 0, 7, 0, "CLIP"],
                [76, 39, 0, 8, 1, "VAE"],
                [107, 58, 0, 3, 3, "LATENT"],
                [110, 8, 0, 60, 0, "IMAGE"],
                [125, 66, 0, 3, 0, "MODEL"],
                [128, 3, 0, 8, 0, "LATENT"],
                [133, 71, 0, 66, 0, "MODEL"],
            ],
            "floatingLinks": None,
            "groups": [],
            "definitions": None,
            "config": {},
            "extra": {
                "links_added_by_ue": [],
                "VHS_KeepIntermediate": True,
                "ue_links": [],
                "VHS_MetadataImage": False,
                "0246.VERSION": [0, 0, 4],
                "VHS_latentpreviewrate": 0,
                "frontendVersion": "1.55.17",
                "VHS_latentpreview": True,
                "ds": {
                    "scale": 0.8769226950000008,
                    "offset": [177.3323960486771, 180.60489023560007],
                },
                "reroutes": None,
            },
            "version": 0.4,
        }
    }


workflow = build_workflow()
prompt = json.loads(json.dumps(workflow))
extra_pnginfo = build_extra_pnginfo()


# Workflow execution
def main(
    positive_prompt: str = '一个穿着"Qwen绘图"标志的T恤的中国美女正拿着黑色的马克笔面相镜头微笑。她身后的玻璃板上手写体写着 “大桥未久，小富即安。三上悠亚，六出祁山。”',
    negative_prompt: str = " 色调艳丽，过曝，细节模糊不清，字幕，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多",
    output_prefix: str = "ComfyUI",
    width: int = 1024,
    height: int = 768,
    batch_size: int = 2,
    steps: int = 10,
    cfg: float = 2.0,
    shift: float = 3.5,
    seed: int | None = None,
    device: str = "cuda:0",
    unload_models: bool | None = None,
    dry_run: bool = False,
):
    import time

    start_time = time.time()

    if device.startswith("cuda:"):
        cuda_idx = device.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx
        os.environ["HIP_VISIBLE_DEVICES"] = cuda_idx

    bootstrap_comfyui_runtime()
    add_extra_model_paths()
    import_custom_nodes()

    # Node imports
    from comfy.nodes.base_nodes import (
        CLIPLoader,
        CLIPTextEncode,
        KSampler,
        SaveImage,
        VAEDecode,
        VAELoader,
    )
    from comfy_extras.nodes.nodes_model_advanced import ModelSamplingAuraFlow
    from comfy_extras.nodes.nodes_sd3 import EmptySD3LatentImage
    from nodes import NODE_CLASS_MAPPINGS

    if dry_run:
        print("[DRY-RUN] ComfyUI runtime bootstrap and custom nodes imported successfully.")
        required_nodes = [
            "UnetLoaderGGUF",
            "CLIPLoader",
            "VAELoader",
            "CLIPTextEncode",
            "EmptySD3LatentImage",
            "ModelSamplingAuraFlow",
            "KSampler",
            "VAEDecode",
            "SaveImage",
        ]
        status = {node_name: node_name in NODE_CLASS_MAPPINGS for node_name in required_nodes}
        print(f"[DRY-RUN] Required node availability: {status}")
        return status

    import torch

    try:
        with torch.inference_mode():
            cliploader = CLIPLoader()
            cliploader_38 = cliploader.load_clip(
                clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors",
                type="qwen_image",
                device="default",
            )
            cliptextencode = CLIPTextEncode()
            cliptextencode_6 = cliptextencode.encode(
                text=positive_prompt,
                clip=get_value_at_index(cliploader_38, 0),
            )
            cliptextencode_7 = cliptextencode.encode(
                text=negative_prompt,
                clip=get_value_at_index(cliploader_38, 0),
            )
            vaeloader = VAELoader()
            vaeloader_39 = vaeloader.load_vae(vae_name="qwen_image_vae.safetensors")
            emptysd3latentimage = EmptySD3LatentImage()
            emptysd3latentimage_58 = emptysd3latentimage.EXECUTE_NORMALIZED(
                width=width, height=height, batch_size=batch_size
            )
            unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
            unetloadergguf_71 = unetloadergguf.load_unet(
                unet_name="qwen-image-Q8_0.gguf"
            )
            modelsamplingauraflow = ModelSamplingAuraFlow()
            ksampler = KSampler()
            vaedecode = VAEDecode()
            saveimage = SaveImage()
            for q in range(1):
                modelsamplingauraflow_66 = modelsamplingauraflow.patch_aura(
                    shift=shift,
                    sampling="flow",
                    model=get_value_at_index(unetloadergguf_71, 0),
                )
                node_3_seed = prompt["3"]["inputs"]["seed"] = (
                    seed if seed is not None else random.randint(1, 2**64)
                )
                ksampler_3 = ksampler.sample(
                    seed=node_3_seed,
                    steps=steps,
                    cfg=cfg,
                    sampler_name="euler",
                    scheduler="simple",
                    denoise=1,
                    model=get_value_at_index(modelsamplingauraflow_66, 0),
                    positive=get_value_at_index(cliptextencode_6, 0),
                    negative=get_value_at_index(cliptextencode_7, 0),
                    latent_image=get_value_at_index(emptysd3latentimage_58, 0),
                )
                vaedecode_8 = vaedecode.decode(
                    samples=get_value_at_index(ksampler_3, 0),
                    vae=get_value_at_index(vaeloader_39, 0),
                )
                saveimage_60 = saveimage.save_images(
                    filename_prefix=output_prefix,
                    images=get_value_at_index(vaedecode_8, 0),
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                )
    finally:
        cleanup_comfyui_runtime(unload_models=unload_models)
        end_time = time.time()
        print(f"Pipeline execution took {end_time - start_time:.2f} seconds")


# Entrypoint
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Qwen-Image (qedit) standalone workflow")
    parser.add_argument(
        "--positive_prompt",
        type=str,
        default='一个穿着"Qwen绘图"标志的T恤的中国美女正拿着黑色的马克笔面相镜头微笑。她身后的玻璃板上手写体写着 “大桥未久，小富即安。三上悠亚，六出祁山。”',
        help="Positive prompt for image generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=" 色调艳丽，过曝，细节模糊不清，字幕，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多",
        help="Negative prompt",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="ComfyUI",
        help="Output image filename prefix (saved inside ./output/)",
    )
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=768, help="Image height")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--steps", type=int, default=10, help="Sampling steps")
    parser.add_argument("--cfg", type=float, default=2.0, help="CFG scale")
    parser.add_argument("--shift", type=float, default=3.5, help="AuraFlow shift")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target CUDA device")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate runtime setup, paths, and node availability without loading models or executing inference.",
    )

    args = parser.parse_args()
    sys.argv = [sys.argv[0]]

    main(
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
        output_prefix=args.output_prefix,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
        steps=args.steps,
        cfg=args.cfg,
        shift=args.shift,
        seed=args.seed,
        device=args.device,
        dry_run=args.dry_run,
    )
