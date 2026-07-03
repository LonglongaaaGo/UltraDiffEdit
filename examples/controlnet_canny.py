from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from diffusers import ControlNetModel


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.common import (
    DEFAULT_NEGATIVE_PROMPT,
    composite_edit_region,
    load_rgb,
    load_sdxl_controlnet_pipeline,
    load_ultradiffedit_pipeline,
    make_canny_condition,
    make_generator,
    prepare_refinement_inputs,
    resize_for_first_stage,
    resolve_target_size,
    save_last_image,
    torch_dtype_for_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UltraDiffEdit with a Canny ControlNet first stage.")
    parser.add_argument("--image", required=True, help="Input image path or URL.")
    parser.add_argument("--mask", required=True, help="Inpainting mask path or URL. White pixels are edited.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", default="results/controlnet_canny.png")
    parser.add_argument("--ckpt", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--controlnet_ckpt", default="diffusers/controlnet-canny-sdxl-1.0")
    parser.add_argument("--control_source", default=None, help="Optional image used only to build the Canny map.")
    parser.add_argument("--target_width", type=int, default=None)
    parser.add_argument("--target_height", type=int, default=None)
    parser.add_argument("--condition_resolution", type=int, default=1024)
    parser.add_argument("--low_threshold", type=int, default=100)
    parser.add_argument("--high_threshold", type=int, default=200)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--first_stage_steps", type=int, default=20)
    parser.add_argument("--refine_steps", type=int, default=30)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--view_batch_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--run_stage", default="two", choices=["two", "three", "S"])
    parser.add_argument("--ug_weight", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch_dtype_for_device(device)
    generator = make_generator(device, args.seed)

    image = load_rgb(args.image)
    mask = load_rgb(args.mask)
    control_source = load_rgb(args.control_source) if args.control_source else image
    target_width, target_height = resolve_target_size(image, args.target_width, args.target_height)

    first_image, first_mask, first_control_source, first_size = resize_for_first_stage(
        image, mask, control_source, args.condition_resolution
    )
    first_control = make_canny_condition(first_control_source, args.low_threshold, args.high_threshold)

    controlnet = ControlNetModel.from_pretrained(args.controlnet_ckpt, torch_dtype=dtype)
    first_pipe = load_sdxl_controlnet_pipeline(args.ckpt, controlnet, dtype, device)

    first_generated = first_pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=first_control,
        height=first_size[1],
        width=first_size[0],
        num_inference_steps=args.first_stage_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        generator=generator,
    ).images[0]
    first_output = composite_edit_region(first_generated, first_image, first_mask)
    del first_pipe, controlnet
    if device == "cuda":
        torch.cuda.empty_cache()

    refine_image, refine_mask, content_image, original_size = prepare_refinement_inputs(
        image, mask, first_output, target_width, target_height
    )

    pipe = load_ultradiffedit_pipeline(args.ckpt, dtype, device)

    images = pipe.refine_editing(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=refine_image,
        content_img=content_image,
        mask_image=refine_mask,
        num_inference_steps=args.refine_steps,
        strength=args.strength,
        generator=generator,
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
