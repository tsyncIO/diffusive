# Imports
import json
import os
import random
import sys
import time
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
    # Default to the project root (parent of scripts/ directory)
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
    return {'120': {'inputs': {'model': 'InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors'},
         'class_type': 'MultiTalkModelLoader',
         '_meta': {'title': 'Multi/InfiniteTalk Model Loader'}},
 '122': {'inputs': {'model': 'Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors',
                    'base_precision': 'fp16_fast',
                    'quantization': 'fp8_e4m3fn_scaled',
                    'load_device': 'offload_device',
                    'attention_mode': 'sageattn',
                    'rms_norm_function': 'default',
                    'block_swap_args': ['134', 0],
                    'lora': ['138', 0],
                    'multitalk_model': ['120', 0]},
         'class_type': 'WanVideoModelLoader',
         '_meta': {'title': 'WanVideo Model Loader'}},
 '125': {'inputs': {'audio': 'dc215636605ce9dca12951049e7cf1f520fbe592d3303843a09fbdc251b43125.wav'},
         'class_type': 'LoadAudio',
         '_meta': {'title': 'Load Audio'}},
 '128': {'inputs': {'steps': 4,
                    'cfg': 1.0000000000000002,
                    'shift': 11.000000000000002,
                    'seed': 417502862519959,
                    'force_offload': True,
                    'scheduler': 'dpm++_sde',
                    'riflex_freq_index': 0,
                    'denoise_strength': 1,
                    'batched_cfg': False,
                    'rope_function': 'comfy',
                    'start_step': 1,
                    'end_step': -1,
                    'add_noise_to_samples': True,
                    'model': ['122', 0],
                    'image_embeds': ['192', 0],
                    'text_embeds': ['241', 0],
                    'samples': ['229', 0],
                    'multitalk_embeds': ['194', 0]},
         'class_type': 'WanVideoSampler',
         '_meta': {'title': 'WanVideo Sampler'}},
 '129': {'inputs': {'model_name': 'Wan2_1_VAE_bf16.safetensors',
                    'precision': 'bf16',
                    'use_cpu_cache': False,
                    'verbose': False},
         'class_type': 'WanVideoVAELoader',
         '_meta': {'title': 'WanVideo VAE Loader'}},
 '130': {'inputs': {'enable_vae_tiling': False,
                    'tile_x': 272,
                    'tile_y': 272,
                    'tile_stride_x': 144,
                    'tile_stride_y': 128,
                    'normalization': 'default',
                    'vae': ['129', 0],
                    'samples': ['128', 0]},
         'class_type': 'WanVideoDecode',
         '_meta': {'title': 'WanVideo Decode'}},
 '131': {'inputs': {'frame_rate': 25,
                    'loop_count': 0,
                    'filename_prefix': 'WanVideo2_1_InfiniteTalk',
                    'format': 'video/h264-mp4',
                    'pix_fmt': 'yuv420p',
                    'crf': 19,
                    'save_metadata': True,
                    'trim_to_audio': False,
                    'pingpong': False,
                    'save_output': True,
                    'no_preview': False,
                    'images': ['299', 0],
                    'audio': ['302', 0]},
         'class_type': 'VHS_VideoCombine',
         '_meta': {'title': 'Video Combine 🎥🅥🅗🅢'}},
 '134': {'inputs': {'blocks_to_swap': 20,
                    'offload_img_emb': False,
                    'offload_txt_emb': False,
                    'use_non_blocking': True,
                    'vace_blocks_to_swap': 0,
                    'prefetch_blocks': 1,
                    'block_swap_debug': False},
         'class_type': 'WanVideoBlockSwap',
         '_meta': {'title': 'WanVideo Block Swap'}},
 '137': {'inputs': {'model': 'TencentGameMate/chinese-wav2vec2-base',
                    'base_precision': 'fp16',
                    'load_device': 'main_device'},
         'class_type': 'DownloadAndLoadWav2VecModel',
         '_meta': {'title': '(Down)load Wav2Vec Model'}},
 '138': {'inputs': {'lora': 'lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors',
                    'strength': 1,
                    'low_mem_load': False,
                    'merge_loras': False},
         'class_type': 'WanVideoLoraSelect',
         '_meta': {'title': 'WanVideo Lora Select'}},
 '170': {'inputs': {'chunk_fade_shape': 'linear',
                    'chunk_length': 10,
                    'chunk_overlap': 0.1,
                    'audio': ['302', 0]},
         'class_type': 'AudioSeparation',
         '_meta': {'title': 'AudioSeparation'}},
 '177': {'inputs': {'backend': 'inductor',
                    'fullgraph': False,
                    'mode': 'default',
                    'dynamic': False,
                    'dynamo_cache_size_limit': 64,
                    'compile_transformer_blocks_only': True,
                    'dynamo_recompile_limit': 128,
                    'force_parameter_static_shapes': False,
                    'allow_unmerged_lora_compile': False},
         'class_type': 'WanVideoTorchCompileSettings',
         '_meta': {'title': 'WanVideo Torch Compile Settings'}},
 '192': {'inputs': {'width': ['291', 1],
                    'height': ['291', 2],
                    'frame_window_size': 81,
                    'motion_frame': 9,
                    'force_offload': False,
                    'colormatch': 'disabled',
                    'tiled_vae': False,
                    'mode': 'infinitetalk',
                    'output_path': '',
                    'vae': ['129', 0],
                    'start_image': ['291', 0],
                    'clip_embeds': ['237', 0]},
         'class_type': 'WanVideoImageToVideoMultiTalk',
         '_meta': {'title': 'WanVideo Long I2V Multi/InfiniteTalk'}},
 '194': {'inputs': {'normalize_loudness': True,
                    'num_frames': ['270', 0],
                    'fps': 25,
                    'audio_scale': 1.5,
                    'audio_cfg_scale': 1,
                    'multi_audio_type': 'para',
                    'add_noise_floor': False,
                    'smooth_transients': False,
                    'wav2vec_model': ['137', 0],
                    'audio_1': ['170', 3]},
         'class_type': 'MultiTalkWav2VecEmbeds',
         '_meta': {'title': 'Multi/InfiniteTalk Wav2vec2 Embeds'}},
 '228': {'inputs': {'video': '5461d1067b5ec0f30fcf27771d0758fd70d025703efcd6877664e92448414deb.mp4',
                    'force_rate': 0,
                    'force_size': 'Disabled',
                    'custom_width': 0,
                    'custom_height': 0,
                    'frame_load_cap': 0,
                    'skip_first_frames': 0,
                    'select_every_nth': 1,
                    'format': 'Wan'},
         'class_type': 'VHS_LoadVideo',
         '_meta': {'title': 'Load Video (Upload) 🎥🅥🅗🅢'}},
 '229': {'inputs': {'enable_vae_tiling': False,
                    'tile_x': 272,
                    'tile_y': 272,
                    'tile_stride_x': 144,
                    'tile_stride_y': 128,
                    'noise_aug_strength': 0,
                    'latent_strength': 1,
                    'vae': ['129', 0],
                    'image': ['230', 0]},
         'class_type': 'WanVideoEncode',
         '_meta': {'title': 'WanVideo Encode'}},
 '230': {'inputs': {'width': ['304', 3],
                    'height': ['304', 4],
                    'upscale_method': 'lanczos',
                    'keep_proportion': 'crop',
                    'pad_color': '0, 0, 0',
                    'crop_position': 'center',
                    'divisible_by': 16,
                    'device': 'cpu',
                    'image': ['304', 0]},
         'class_type': 'ImageResizeKJv2',
         '_meta': {'title': 'Resize Image v2'}},
 '231': {'inputs': {'start_index': 0, 'num_frames': 1, 'images': ['230', 0]},
         'class_type': 'GetImageRangeFromBatch',
         '_meta': {'title': 'Get Image or Mask Range From Batch'}},
 '237': {'inputs': {'strength_1': 1,
                    'strength_2': 1,
                    'crop': 'center',
                    'combine_embeds': 'average',
                    'force_offload': True,
                    'tiles': 0,
                    'ratio': 0.5,
                    'clip_vision': ['238', 0],
                    'image_1': ['291', 0]},
         'class_type': 'WanVideoClipVisionEncode',
         '_meta': {'title': 'WanVideo ClipVision Encode'}},
 '238': {'inputs': {'clip_name': 'clip_vision_h.safetensors'},
         'class_type': 'CLIPVisionLoader',
         '_meta': {'title': 'Load CLIP Vision'}},
 '241': {'inputs': {'model_name': 'umt5-xxl-enc-bf16.safetensors',
                    'precision': 'bf16',
                    'positive_prompt': ['306', 0],
                    'negative_prompt': 'bright tones, overexposed, static, '
                                       'blurred details, subtitles, style, '
                                       'works, paintings, images, static, '
                                       'overall gray, worst quality, low '
                                       'quality, JPEG compression residue, '
                                       'ugly, incomplete, extra fingers, '
                                       'poorly drawn hands, poorly drawn '
                                       'faces, deformed, disfigured, misshapen '
                                       'limbs, fused fingers, still picture, '
                                       'messy background, three legs, many '
                                       'people in the background, walking '
                                       'backwards',
                    'quantization': 'disabled',
                    'use_disk_cache': False,
                    'device': 'gpu'},
         'class_type': 'WanVideoTextEncodeCached',
         '_meta': {'title': 'WanVideo TextEncode Cached'}},
 '270': {'inputs': {'value': 1000},
         'class_type': 'INTConstant',
         '_meta': {'title': 'INT Constant'}},
 '291': {'inputs': {'image': ['231', 0]},
         'class_type': 'GetImageSizeAndCount',
         '_meta': {'title': 'Get Image Size & Count'}},
 '293': {'inputs': {'preview': '', 'source': ['194', 2]},
         'class_type': 'PreviewAny',
         '_meta': {'title': 'Preview as Text'}},
 '299': {'inputs': {'inputcount': 2,
                    'direction': 'left',
                    'match_image_size': False,
                    'Update inputs': None,
                    'image_1': ['301', 0],
                    'image_2': ['230', 0]},
         'class_type': 'ImageConcatMulti',
         '_meta': {'title': 'Image Concatenate Multi'}},
 '300': {'inputs': {'image': ['130', 0]},
         'class_type': 'GetImageSizeAndCount',
         '_meta': {'title': 'Get Image Size & Count'}},
 '301': {'inputs': {'start_index': 0,
                    'num_frames': ['300', 3],
                    'images': ['300', 0]},
         'class_type': 'GetImageRangeFromBatch',
         '_meta': {'title': 'Get Image or Mask Range From Batch'}},
 '302': {'inputs': {'gain_db': 0,
                    'offset_seconds': 0,
                    'duration_seconds': 60,
                    'resample_to_hz': 0,
                    'make_stereo': True,
                    'audio': ['125', 0]},
         'class_type': 'AudioCropProcessUTK',
         '_meta': {'title': 'Audio Crop Process (UTK)'}},
 '303': {'inputs': {'audio': ['170', 3]},
         'class_type': 'PreviewAudio',
         '_meta': {'title': 'Preview Audio'}},
 '304': {'inputs': {'aspect_ratio': 'original',
                    'proportional_width': 1,
                    'proportional_height': 1,
                    'fit': 'letterbox',
                    'method': 'lanczos',
                    'round_to_multiple': '16',
                    'scale_to_side': 'total_pixel(kilo pixel)',
                    'scale_to_length': 500,
                    'background_color': '#000000',
                    'image': ['228', 0]},
         'class_type': 'ImageScaleByAspectRatio_UTK',
         '_meta': {'title': 'Image Scale By Aspect Ratio (UTK)'}},
 '306': {'inputs': {'text': 'I hereby announce that the United States Armed '
                            'Forces Command formally surrenders and agrees to '
                            'all terms and conditions set forth by the Iranian '
                            'Supreme Leader, effective immediately. The Strait '
                            'of Hormuz will remain in complete Iranian '
                            'supervision as well.'},
         'class_type': 'LayerUtility: TextBox',
         '_meta': {'title': 'Prompt'}}}

def build_extra_pnginfo() -> dict[str, Any] | None:
    return None

workflow = build_workflow()
prompt = json.loads(json.dumps(workflow))
extra_pnginfo = build_extra_pnginfo()

# Workflow execution
def main(
    video: str,
    audio: str,
    positive_prompt: str,
    negative_prompt: str,
    output_prefix: str,
    device: str = "cuda:0",
    seed: int | None = None,
    audio_scale: float = 1.5,
    motion_frame: int = 9,
    steps: int = 4,
    unload_models: bool | None = None,
    dry_run: bool = False
):
    start_time = time.time()
    
    if device.startswith("cuda:"):
        cuda_idx = device.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx
        os.environ["HIP_VISIBLE_DEVICES"] = cuda_idx

    bootstrap_comfyui_runtime()
    add_extra_model_paths()
    import_custom_nodes()

    # Node imports
    from nodes import CLIPVisionLoader, NODE_CLASS_MAPPINGS

    if dry_run:
        print("[DRY-RUN] ComfyUI runtime bootstrap and custom nodes imported successfully.")
        required_nodes = [
            "WanVideoModelLoader",
            "WanVideoSampler",
            "MultiTalkModelLoader",
            "WanVideoImageToVideoMultiTalk",
            "VHS_VideoCombine",
            "VHS_LoadVideo"
        ]
        status = {}
        for node_name in required_nodes:
            status[node_name] = node_name in NODE_CLASS_MAPPINGS
        print(f"[DRY-RUN] Required node availability: {status}")
        return status

    import torch

    try:
        with torch.inference_mode():
            multitalkmodelloader = NODE_CLASS_MAPPINGS["MultiTalkModelLoader"]()
            multitalkmodelloader_120 = multitalkmodelloader.loadmodel(model="InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors")
            loadaudio = NODE_CLASS_MAPPINGS["LoadAudio"]()
            loadaudio_125 = loadaudio.EXECUTE_NORMALIZED(audio=audio)
            wanvideovaeloader = NODE_CLASS_MAPPINGS["WanVideoVAELoader"]()
            wanvideovaeloader_129 = wanvideovaeloader.loadmodel(model_name="Wan2_1_VAE_bf16.safetensors", precision="bf16", use_cpu_cache=False, verbose=False)
            wanvideoblockswap = NODE_CLASS_MAPPINGS["WanVideoBlockSwap"]()
            wanvideoblockswap_134 = wanvideoblockswap.setargs(blocks_to_swap=20, offload_img_emb=False, offload_txt_emb=False, use_non_blocking=True, vace_blocks_to_swap=0, prefetch_blocks=1, block_swap_debug=False)
            downloadandloadwav2vecmodel = NODE_CLASS_MAPPINGS["DownloadAndLoadWav2VecModel"]()
            downloadandloadwav2vecmodel_137 = downloadandloadwav2vecmodel.loadmodel(model="TencentGameMate/chinese-wav2vec2-base", base_precision="fp16", load_device="main_device")
            wanvideoloraselect = NODE_CLASS_MAPPINGS["WanVideoLoraSelect"]()
            wanvideoloraselect_138 = wanvideoloraselect.getlorapath(lora="lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", strength=1, low_mem_load=False, merge_loras=False, unique_id="138")
            wanvideotorchcompilesettings = NODE_CLASS_MAPPINGS["WanVideoTorchCompileSettings"]()
            wanvideotorchcompilesettings_177 = wanvideotorchcompilesettings.set_args(backend="inductor", fullgraph=False, mode="default", dynamic=False, dynamo_cache_size_limit=64, compile_transformer_blocks_only=True, dynamo_recompile_limit=128, force_parameter_static_shapes=False, allow_unmerged_lora_compile=False)
            vhs_loadvideo = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
            vhs_loadvideo_228 = vhs_loadvideo.load_video(video=video, force_rate=0, force_size="Disabled", custom_width=0, custom_height=0, frame_load_cap=0, skip_first_frames=0, select_every_nth=1, format="Wan", unique_id="228")
            imagescalebyaspectratio_utk = NODE_CLASS_MAPPINGS["ImageScaleByAspectRatio_UTK"]()
            imagescalebyaspectratio_utk_304 = imagescalebyaspectratio_utk.image_scale_by_aspect_ratio(aspect_ratio="original", proportional_width=1, proportional_height=1, fit="letterbox", method="lanczos", round_to_multiple="16", scale_to_side="total_pixel(kilo pixel)", scale_to_length=500, background_color="black", crop_position="center", image=get_value_at_index(vhs_loadvideo_228, 0))
            imageresizekjv2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
            imageresizekjv2_230 = imageresizekjv2.resize(width=get_value_at_index(imagescalebyaspectratio_utk_304, 3), height=get_value_at_index(imagescalebyaspectratio_utk_304, 4), upscale_method="lanczos", keep_proportion="crop", pad_color="0, 0, 0", crop_position="center", divisible_by=16, device="cpu", image=get_value_at_index(imagescalebyaspectratio_utk_304, 0), unique_id="230")
            wanvideoencode = NODE_CLASS_MAPPINGS["WanVideoEncode"]()
            wanvideoencode_229 = wanvideoencode.encode(enable_vae_tiling=False, tile_x=272, tile_y=272, tile_stride_x=144, tile_stride_y=128, noise_aug_strength=0, latent_strength=1, vae=get_value_at_index(wanvideovaeloader_129, 0), image=get_value_at_index(imageresizekjv2_230, 0))
            clipvisionloader = CLIPVisionLoader()
            clipvisionloader_238 = clipvisionloader.load_clip(clip_name="clip_vision_h.safetensors")
            intconstant = NODE_CLASS_MAPPINGS["INTConstant"]()
            intconstant_270 = intconstant.get_value(value=1000)
            layerutility_textbox_306 = [positive_prompt]
            wanvideomodelloader = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
            getimagerangefrombatch = NODE_CLASS_MAPPINGS["GetImageRangeFromBatch"]()
            getimagesizeandcount = NODE_CLASS_MAPPINGS["GetImageSizeAndCount"]()
            wanvideoclipvisionencode = NODE_CLASS_MAPPINGS["WanVideoClipVisionEncode"]()
            wanvideoimagetovideomultitalk = NODE_CLASS_MAPPINGS["WanVideoImageToVideoMultiTalk"]()
            wanvideotextencodecached = NODE_CLASS_MAPPINGS["WanVideoTextEncodeCached"]()
            audiocropprocessutk = NODE_CLASS_MAPPINGS["AudioCropProcessUTK"]()
            audioseparation = NODE_CLASS_MAPPINGS["AudioSeparation"]()
            multitalkwav2vecembeds = NODE_CLASS_MAPPINGS["MultiTalkWav2VecEmbeds"]()
            wanvideosampler = NODE_CLASS_MAPPINGS["WanVideoSampler"]()
            wanvideodecode = NODE_CLASS_MAPPINGS["WanVideoDecode"]()
            imageconcatmulti = NODE_CLASS_MAPPINGS["ImageConcatMulti"]()
            vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
            previewany = NODE_CLASS_MAPPINGS["PreviewAny"]()
            previewaudio = NODE_CLASS_MAPPINGS["PreviewAudio"]()
            for q in range(1):
                wanvideomodelloader_122 = wanvideomodelloader.loadmodel(model="Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors", base_precision="fp16_fast", quantization="fp8_e4m3fn_scaled", load_device="offload_device", attention_mode="sdpa", rms_norm_function="default", block_swap_args=get_value_at_index(wanvideoblockswap_134, 0), lora=get_value_at_index(wanvideoloraselect_138, 0), multitalk_model=get_value_at_index(multitalkmodelloader_120, 0))
                getimagerangefrombatch_231 = getimagerangefrombatch.imagesfrombatch(start_index=0, num_frames=1, images=get_value_at_index(imageresizekjv2_230, 0))
                getimagesizeandcount_291 = getimagesizeandcount.getsize(image=get_value_at_index(getimagerangefrombatch_231, 0))
                wanvideoclipvisionencode_237 = wanvideoclipvisionencode.process(strength_1=1, strength_2=1, crop="center", combine_embeds="average", force_offload=True, tiles=0, ratio=0.5, clip_vision=get_value_at_index(clipvisionloader_238, 0), image_1=get_value_at_index(getimagesizeandcount_291, 0))
                wanvideoimagetovideomultitalk_192 = wanvideoimagetovideomultitalk.process(width=get_value_at_index(getimagesizeandcount_291, 1), height=get_value_at_index(getimagesizeandcount_291, 2), frame_window_size=81, motion_frame=motion_frame, force_offload=False, colormatch="disabled", tiled_vae=False, mode="infinitetalk", output_path="", vae=get_value_at_index(wanvideovaeloader_129, 0), start_image=get_value_at_index(getimagesizeandcount_291, 0), clip_embeds=get_value_at_index(wanvideoclipvisionencode_237, 0))
                wanvideotextencodecached_241 = wanvideotextencodecached.process(model_name="umt5-xxl-enc-bf16.safetensors", precision="bf16", positive_prompt=get_value_at_index(layerutility_textbox_306, 0), negative_prompt=negative_prompt, quantization="disabled", use_disk_cache=False, device="gpu")
                audiocropprocessutk_302 = audiocropprocessutk.execute(gain_db=0, offset_seconds=0, duration_seconds=60, resample_to_hz=0, make_stereo=True, audio=get_value_at_index(loadaudio_125, 0))
                audioseparation_170 = audioseparation.main(chunk_fade_shape="linear", chunk_length=10, chunk_overlap=0.1, audio=get_value_at_index(audiocropprocessutk_302, 0))
                multitalkwav2vecembeds_194 = multitalkwav2vecembeds.process(normalize_loudness=True, num_frames=get_value_at_index(intconstant_270, 0), fps=25, audio_scale=audio_scale, audio_cfg_scale=1, multi_audio_type="para", add_noise_floor=False, smooth_transients=False, wav2vec_model=get_value_at_index(downloadandloadwav2vecmodel_137, 0), audio_1=get_value_at_index(audioseparation_170, 3))
                node_128_seed = prompt["128"]["inputs"]["seed"] = seed if seed is not None else random.randint(1, 2**64)
                wanvideosampler_128 = wanvideosampler.process(steps=steps, cfg=1.0000000000000002, shift=11.000000000000002, seed=node_128_seed, force_offload=True, scheduler="dpm++_sde", riflex_freq_index=0, denoise_strength=1, batched_cfg=False, rope_function="comfy", start_step=1, end_step=-1, add_noise_to_samples=True, model=get_value_at_index(wanvideomodelloader_122, 0), image_embeds=get_value_at_index(wanvideoimagetovideomultitalk_192, 0), text_embeds=get_value_at_index(wanvideotextencodecached_241, 0), samples=get_value_at_index(wanvideoencode_229, 0), multitalk_embeds=get_value_at_index(multitalkwav2vecembeds_194, 0))
                wanvideodecode_130 = wanvideodecode.decode(enable_vae_tiling=False, tile_x=272, tile_y=272, tile_stride_x=144, tile_stride_y=128, normalization="default", vae=get_value_at_index(wanvideovaeloader_129, 0), samples=get_value_at_index(wanvideosampler_128, 0))
                getimagesizeandcount_300 = getimagesizeandcount.getsize(image=get_value_at_index(wanvideodecode_130, 0))
                getimagerangefrombatch_301 = getimagerangefrombatch.imagesfrombatch(start_index=0, num_frames=get_value_at_index(getimagesizeandcount_300, 3), images=get_value_at_index(getimagesizeandcount_300, 0))
                # imageconcatmulti_299 = imageconcatmulti.EXECUTE_NORMALIZED(inputcount=2, direction="left", match_image_size=False, **{"Update inputs": None}, image_1=get_value_at_index(getimagerangefrombatch_301, 0), image_2=get_value_at_index(imageresizekjv2_230, 0))
                vhs_videocombine_131 = vhs_videocombine.combine_video(frame_rate=25, loop_count=0, filename_prefix=output_prefix, format="video/h264-mp4", pix_fmt="yuv420p", crf=19, save_metadata=True, trim_to_audio=True, pingpong=False, save_output=True, no_preview=False, images=get_value_at_index(getimagerangefrombatch_301, 0), audio=get_value_at_index(audiocropprocessutk_302, 0), unique_id="131", prompt=prompt, extra_pnginfo=extra_pnginfo)
                previewany_293 = previewany.main(source=get_value_at_index(multitalkwav2vecembeds_194, 2))
                previewaudio_cls = previewaudio.PREPARE_CLASS_CLONE({"hidden_inputs": {"PROMPT": prompt, "EXTRA_PNGINFO": extra_pnginfo}})
                previewaudio_303 = previewaudio_cls.EXECUTE_NORMALIZED(audio=get_value_at_index(audioseparation_170, 3))
    finally:
        cleanup_comfyui_runtime(unload_models=unload_models)
        
        end_time = time.time()
        print(f"Pipeline execution took {end_time - start_time:.2f} seconds")

# Entrypoint
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the InfiniteTalk V2V workflow")
    parser.add_argument("--video", type=str, default="ZohranMamdani_10sec.mp4", help="Path or name of the input video file")
    parser.add_argument("--audio", type=str, default="ZohranMamdani_5sec_fake.wav", help="Path or name of the input audio file")
    parser.add_argument("--positive_prompt", type=str, default="", help="Positive prompt / transcript")
    parser.add_argument("--negative_prompt", type=str, default="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", help="Negative prompt")
    parser.add_argument("--output_prefix", type=str, default="./output/generated", help="Output video filename prefix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target device (e.g., cuda:1)")
    parser.add_argument("--audio_scale", type=float, default=1.5, help="Strength of the audio guidance for expressions. Increase for more pronounced lip-sync and movements.")
    parser.add_argument("--motion_frame", type=int, default=9, help="Motion frame parameter. Can tweak for adjusting movement scale.")
    parser.add_argument("--steps", type=int, default=4, help="Number of sampling steps. Higher = better quality but slower.")
    parser.add_argument("--dry-run", action="store_true", help="Validate runtime setup, paths, and node availability without loading models or executing inference.")
    
    args = parser.parse_args()
    
    # Clear sys.argv so ComfyUI's internal parser doesn't crash on our custom arguments
    import sys
    sys.argv = [sys.argv[0]]
    
    main(
        video=args.video,
        audio=args.audio,
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
        output_prefix=args.output_prefix,
        device=args.device,
        seed=args.seed,
        audio_scale=args.audio_scale,
        motion_frame=args.motion_frame,
        steps=args.steps,
        dry_run=args.dry_run
    )