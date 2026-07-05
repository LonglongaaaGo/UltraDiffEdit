from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_cache import configure_model_cache, ensure_model_cache_dir

configure_model_cache()

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from examples.common import (
    DEFAULT_NEGATIVE_PROMPT,
    composite_edit_region,
    load_rgb,
    load_sdxl_inpaint_pipeline,
    load_ultradiffedit_pipeline,
    first_stage_size_for_target,
    prepare_refinement_inputs,
    resolve_target_size,
    save_last_image,
    torch_dtype_for_device,
)
from ip_adapter import IPAdapterXL, IPAdapterXL_ultra_inpaint


DEFAULT_IMAGE_ENCODER_PATH = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
DEFAULT_IP_ADAPTER_REPO = "h94/IP-Adapter"
DEFAULT_IP_ADAPTER_FILENAME = "sdxl_models/ip-adapter_sdxl.bin"


def resolve_ip_ckpt(ip_ckpt: Optional[str]) -> str:
    if ip_ckpt:
        return ip_ckpt

    return hf_hub_download(
        repo_id=DEFAULT_IP_ADAPTER_REPO,
        filename=DEFAULT_IP_ADAPTER_FILENAME,
        cache_dir=ensure_model_cache_dir(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UltraDiffEdit with IP-Adapter guided inpainting.")
    parser.add_argument("--image", required=True, help="Target image path or URL.")
    parser.add_argument("--mask", required=True, help="Inpainting mask path or URL. White pixels are edited.")
    parser.add_argument("--reference_image", required=True, help="Visual prompt image consumed by IP-Adapter.")
    parser.add_argument("--prompt", default="best quality, high quality")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", default="results/ip_adapter.png")
    parser.add_argument("--ckpt", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument(
        "--image_encoder_path",
        default=DEFAULT_IMAGE_ENCODER_PATH,
        help=f"CLIP image encoder repo id or local path. Default: {DEFAULT_IMAGE_ENCODER_PATH}.",
    )
    parser.add_argument(
        "--ip_ckpt",
        default=None,
        help=(
            "Local IP-Adapter SDXL checkpoint path. If omitted, the script downloads "
            f"{DEFAULT_IP_ADAPTER_REPO}/{DEFAULT_IP_ADAPTER_FILENAME}."
        ),
    )
    parser.add_argument("--target_width", type=int, default=None)
    parser.add_argument("--target_height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--scale", type=float, default=0.8)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--condition_resolution", type=int, default=1024)
    parser.add_argument("--view_batch_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--run_stage", default="two", choices=["two", "three", "S"])
    parser.add_argument("--ug_weight", type=float, default=0.2)
    return parser.parse_args()


def ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch_dtype_for_device(device)
    ip_ckpt = resolve_ip_ckpt(args.ip_ckpt)

    image = load_rgb(args.image)
    mask = load_rgb(args.mask)
    reference_image = load_rgb(args.reference_image)
    target_width, target_height = resolve_target_size(image, args.target_width, args.target_height)
    if max(target_width, target_height) <= args.condition_resolution:
        first_width = ceil_to_multiple(target_width, 8)
        first_height = ceil_to_multiple(target_height, 8)
    else:
        first_width, first_height = first_stage_size_for_target(
            target_width, target_height, args.condition_resolution
        )
    first_size = (first_width, first_height)
    first_image = image.resize(first_size, Image.Resampling.BICUBIC)
    first_mask = mask.resize(first_size, Image.Resampling.NEAREST)

    first_pipe = load_sdxl_inpaint_pipeline(args.ckpt, dtype, device)
    first_adapter = IPAdapterXL(first_pipe, args.image_encoder_path, ip_ckpt, device)
    first_images = first_adapter.generate(
        pil_image=reference_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        scale=args.scale,
        seed=args.seed,
        num_samples=1,
        num_inference_steps=args.num_inference_steps,
        image=first_image,
        mask_image=first_mask,
        height=first_height,
        width=first_width,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
    )
    first_output = composite_edit_region(first_images[-1], first_image, first_mask)

    if max(target_width, target_height) <= args.condition_resolution:
        save_last_image([first_output], args.output, (target_width, target_height))
        return

    del first_adapter, first_pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    target_image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
    target_mask = mask.resize((target_width, target_height), Image.Resampling.NEAREST)
    target_first_output = first_output.resize((target_width, target_height), Image.Resampling.BICUBIC)
    refine_seed = composite_edit_region(target_first_output, target_image, target_mask)

    refine_image, refine_mask, _content_image, original_size = prepare_refinement_inputs(
        refine_seed, mask, refine_seed, target_width, target_height
    )

    pipe = load_ultradiffedit_pipeline(args.ckpt, dtype, device)
    adapter = IPAdapterXL_ultra_inpaint(pipe, args.image_encoder_path, ip_ckpt, device)

    images = adapter.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=reference_image,
        edit_image=refine_image,
        mask_image=refine_mask,
        scale=args.scale,
        seed=args.seed,
        num_samples=1,
        num_inference_steps=args.num_inference_steps,
        strength=args.strength,
        tar_height=refine_image.size[1],
        tar_width=refine_image.size[0],
        view_batch_size=args.view_batch_size,
        stride=args.stride,
        beta_scale_1=3,
        beta_scale_2=1,
        cosine_scale_3=1,
        sigma=0.8,
        multi_decoder=True,
        run_stage=args.run_stage,
        ug_weight=args.ug_weight,
        orig_size=[original_size[1], original_size[0]],
    )
    save_last_image(images, args.output, original_size)


if __name__ == "__main__":
    main()
