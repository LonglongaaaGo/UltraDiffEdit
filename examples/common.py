from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch
from diffusers.utils import load_image
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Util.img_pad_crop import crop_image_to_original, get_start_size, pad_image_to_multiple_num


DEFAULT_NEGATIVE_PROMPT = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"


def torch_dtype_for_device(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


def sdxl_from_pretrained_kwargs(dtype: torch.dtype, device: str) -> dict:
    kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    if device == "cuda":
        kwargs["variant"] = "fp16"
    return kwargs


def load_ultradiffedit_pipeline(ckpt: str, dtype: torch.dtype, device: str):
    from pipeline_ultradiffedit_sdxl import StableAnysizeInpaintPipeline

    pipe = StableAnysizeInpaintPipeline.from_pretrained(
        ckpt,
        **sdxl_from_pretrained_kwargs(dtype, device),
    )
    return pipe.to(device)


def load_sdxl_controlnet_pipeline(ckpt: str, controlnet, dtype: torch.dtype, device: str):
    try:
        from diffusers import StableDiffusionXLControlNetPipeline
    except ImportError as exc:
        raise ImportError(
            "ControlNet examples require diffusers with StableDiffusionXLControlNetPipeline. "
            "Install the optional example dependencies or upgrade diffusers."
        ) from exc

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        ckpt,
        controlnet=controlnet,
        **sdxl_from_pretrained_kwargs(dtype, device),
    )
    return pipe.to(device)


def load_rgb(path_or_url: str) -> Image.Image:
    return load_image(path_or_url).convert("RGB")


def resolve_target_size(image: Image.Image, target_width: Optional[int], target_height: Optional[int]) -> tuple[int, int]:
    width, height = image.size
    return target_width or width, target_height or height


def make_generator(device: str, seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(seed)


def make_canny_condition(image: Image.Image, low_threshold: int, high_threshold: int) -> Image.Image:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Canny examples require opencv-python. Install it with `pip install opencv-python`.") from exc
    import numpy as np

    image_array = np.array(image.convert("RGB"))
    edges = cv2.Canny(image_array, low_threshold, high_threshold)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)


def resize_for_first_stage(
    image: Image.Image,
    mask: Image.Image,
    control_source: Image.Image,
    resolution: int,
) -> tuple[Image.Image, Image.Image, Image.Image, tuple[int, int]]:
    width, height = image.size
    first_width, first_height = get_start_size(width, height, fix_size=resolution)
    first_width = max(8, (first_width // 8) * 8)
    first_height = max(8, (first_height // 8) * 8)
    size = (first_width, first_height)
    return image.resize(size), mask.resize(size), control_source.resize(size), size


def composite_edit_region(generated: Image.Image, original: Image.Image, mask: Image.Image) -> Image.Image:
    return Image.composite(generated.convert("RGB"), original.convert("RGB"), mask.convert("L"))


def prepare_refinement_inputs(
    image: Image.Image,
    mask: Image.Image,
    content: Image.Image,
    target_width: int,
    target_height: int,
) -> tuple[Image.Image, Image.Image, Image.Image, tuple[int, int]]:
    image = image.resize((target_width, target_height)).convert("RGB")
    mask = mask.resize((target_width, target_height)).convert("RGB")
    content = content.resize((target_width, target_height)).convert("RGB")

    max_scale = math.ceil(max(target_width, target_height) / 1024)
    pad_multiple = math.lcm(max_scale * 8, 1024)

    image, original_size = pad_image_to_multiple_num(image, num=pad_multiple, color=(255, 255, 255))
    mask, _ = pad_image_to_multiple_num(mask, num=pad_multiple, color=(0, 0, 0))
    content, _ = pad_image_to_multiple_num(content, num=pad_multiple, color=(255, 255, 255))
    return image, mask, content, original_size


def save_last_image(images: Iterable[Image.Image], output_path: str, original_size: tuple[int, int]) -> None:
    output = list(images)[-1]
    if output.size != original_size:
        output = crop_image_to_original(output, original_size)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path)
