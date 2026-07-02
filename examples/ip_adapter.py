from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.common import (
    DEFAULT_NEGATIVE_PROMPT,
    load_rgb,
    prepare_refinement_inputs,
    resolve_target_size,
    save_last_image,
    torch_dtype_for_device,
)
from ip_adapter import IPAdapterXL_ultra_inpaint
from pipeline_ultradiffedit_sdxl import StableAnysizeInpaintPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UltraDiffEdit with an IP-Adapter image prompt.")
    parser.add_argument("--image", required=True, help="Input image path or URL.")
    parser.add_argument("--mask", required=True, help="Inpainting mask path or URL. White pixels are edited.")
    parser.add_argument("--reference_image", required=True, help="Image prompt consumed by IP-Adapter.")
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
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--scale", type=float, default=0.8)
    parser.add_argument("--view_batch_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--run_stage", default="two", choices=["two", "three", "S"])
    parser.add_argument("--ug_weight", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch_dtype_for_device(device)

    image = load_rgb(args.image)
    mask = load_rgb(args.mask)
    reference_image = load_rgb(args.reference_image)
    target_width, target_height = resolve_target_size(image, args.target_width, args.target_height)

    refine_image, refine_mask, content_image, original_size = prepare_refinement_inputs(
        image, mask, image, target_width, target_height
    )

    pipe = StableAnysizeInpaintPipeline.from_pretrained(
        args.ckpt,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
    )
    adapter = IPAdapterXL_ultra_inpaint(pipe, args.image_encoder_path, args.ip_ckpt, device)

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

