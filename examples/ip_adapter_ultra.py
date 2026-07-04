from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_cache import configure_model_cache

configure_model_cache()

import torch

from examples.common import (
    DEFAULT_NEGATIVE_PROMPT,
    load_rgb,
    load_sdxl_pipeline,
    load_ultradiffedit_pipeline,
    prepare_generation_inputs,
    resize_for_first_stage,
    resolve_target_size,
    save_last_image,
    torch_dtype_for_device,
)
from ip_adapter import IPAdapterXL, IPAdapterXL_ultra_inpaint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UltraDiffEdit with an IP-Adapter visual prompt.")
    parser.add_argument("--image", required=True, help="Visual prompt image path or URL consumed by IP-Adapter.")
    parser.add_argument("--mask", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reference_image", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--prompt", default="best quality, high quality")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", default="results/ip_adapter.png")
    parser.add_argument("--ckpt", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--image_encoder_path", required=True, help="CLIP image encoder path, e.g. laion/CLIP-ViT-bigG-14-laion2B-39B-b160k.")
    parser.add_argument("--ip_ckpt", required=True, help="IP-Adapter SDXL checkpoint path.")
    parser.add_argument("--target_width", type=int, default=None)
    parser.add_argument("--target_height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--strength", type=float, default=0.9999)
    parser.add_argument("--scale", type=float, default=0.8)
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

    visual_prompt = load_rgb(args.reference_image or args.image)
    target_width, target_height = resolve_target_size(visual_prompt, args.target_width, args.target_height)
    if max(target_width, target_height) <= 1024:
        first_width = ceil_to_multiple(target_width, 8)
        first_height = ceil_to_multiple(target_height, 8)
    else:
        _first_image, _first_mask, _first_control, first_size = resize_for_first_stage(
            visual_prompt, visual_prompt, visual_prompt, 1024
        )
        first_width, first_height = first_size

    first_pipe = load_sdxl_pipeline(args.ckpt, dtype, device)
    first_adapter = IPAdapterXL(first_pipe, args.image_encoder_path, args.ip_ckpt, device)
    first_images = first_adapter.generate(
        pil_image=visual_prompt,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        scale=args.scale,
        seed=args.seed,
        num_samples=1,
        num_inference_steps=args.num_inference_steps,
        height=first_height,
        width=first_width,
    )
    first_output = first_images[-1]

    if max(target_width, target_height) <= 1024:
        save_last_image([first_output], args.output, (target_width, target_height))
        return

    del first_adapter, first_pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    canvas, full_mask, original_size = prepare_generation_inputs(
        target_width,
        target_height,
        init_image=first_output,
    )

    pipe = load_ultradiffedit_pipeline(args.ckpt, dtype, device)
    adapter = IPAdapterXL_ultra_inpaint(pipe, args.image_encoder_path, args.ip_ckpt, device)

    images = adapter.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=visual_prompt,
        edit_image=canvas,
        mask_image=full_mask,
        scale=args.scale,
        seed=args.seed,
        num_samples=1,
        num_inference_steps=args.num_inference_steps,
        strength=args.strength,
        tar_height=canvas.size[1],
        tar_width=canvas.size[0],
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
