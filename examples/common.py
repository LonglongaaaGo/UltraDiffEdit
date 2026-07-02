from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

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


def load_rgb(path_or_url: str) -> Image.Image:
    return load_image(path_or_url).convert("RGB")


def resolve_target_size(image: Image.Image, target_width: int | None, target_height: int | None) -> tuple[int, int]:
    width, height = image.size
    return target_width or width, target_height or height


def make_generator(device: str, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(seed)


def make_canny_condition(image: Image.Image, low_threshold: int, high_threshold: int) -> Image.Image:
    import cv2
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
    size = (first_width, first_height)
    return image.resize(size), mask.resize(size), control_source.resize(size), size


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

