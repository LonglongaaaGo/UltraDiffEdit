# UltraDiffEdit: Tuning-Free Latent Diffusion Models for Ultrahigh-Resolution Image Editing

[![IEEE TNNLS](https://img.shields.io/badge/IEEE%20TNNLS-2026-00629B?logo=ieee&logoColor=white)](https://doi.org/10.1109/TNNLS.2026.3707463)
[![GitHub Stars](https://img.shields.io/github/stars/LonglongaaaGo/UltraDiffEdit?style=social)](https://github.com/LonglongaaaGo/UltraDiffEdit/stargazers)
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=LonglongaaaGo.UltraDiffEdit)](https://github.com/LonglongaaaGo/UltraDiffEdit)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Official code release for **Tuning-Free Latent Diffusion Models for Ultrahigh-Resolution Image Editing**.

UltraDiffEdit extends off-the-shelf latent diffusion models to ultrahigh-resolution real-image editing without additional training. The method performs multiscale progressive editing with multipatch encoding, global-local consistency denoising, and patch-based hybrid sampling.

<img src="imgs/framework.png" width="800"/>
<img src="imgs/res1.png" width="800"/>

## News

- **2026-06-21**: Accepted by *IEEE Transactions on Neural Networks and Learning Systems*.
- **2026-07-01**: Initial public code release.

## Environment

Use a clean environment for both the main UltraDiffEdit pipeline and the optional ControlNet/IP-Adapter examples. Avoid installing unrelated editable packages such as local `peft` checkouts into this environment, because Diffusers may import them automatically.

```bash
conda create -n ultradiffedit python=3.9
conda activate ultradiffedit
pip install -r requirements.txt
```

By default, Hugging Face and Diffusers use their standard cache location, usually `~/.cache/huggingface/hub`. To place model downloads on a different disk, set `ULTRADIFFEDIT_MODEL_CACHE` before running any script:

```bash
export ULTRADIFFEDIT_MODEL_CACHE=/path/to/your/model_cache
```

The default examples use SDXL and require a CUDA GPU. The paper reports editing up to 8K resolution on a single NVIDIA RTX 3090. For larger images, reduce `view_batch_size` if you run out of memory.
ControlNet and IP-Adapter examples additionally require external model checkpoints, but their Python dependencies are included in `requirements.txt`.

## Quick Start

The recommended entry point is `pipeline_ultradiffedit_sdxl.py`.

```python
import os
import time

import torch
from diffusers.utils import load_image

from model_cache import ensure_model_cache_dir
from pipeline_ultradiffedit_sdxl import StableAnysizeInpaintPipeline


os.makedirs("results", exist_ok=True)
name = str(time.time())

pipe = StableAnysizeInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    cache_dir=ensure_model_cache_dir(),
)
pipe.to("cuda")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

height = 2048
width = 2048

image = load_image(img_url).convert("RGB").resize((width, height))
mask = load_image(mask_url).convert("RGB").resize((width, height))

generator = torch.Generator(device="cuda").manual_seed(5)

images = pipe(
    prompt="a cute cat sitting on a bench",
    negative_prompt="blurry, ugly, duplicate, poorly drawn, deformed, mosaic",
    image=image,
    mask_image=mask,
    num_inference_steps=50,
    strength=0.80,
    generator=generator,
    tar_height=height,
    tar_width=width,
    view_batch_size=16,
    stride=64,
    beta_scale_1=3,
    beta_scale_2=1,
    cosine_scale_3=1,
    sigma=0.8,
    multi_decoder=True,
    show_image=False,
    save_image_tag=False,
    file_name=name,
    save_root="results",
    run_stage="two",
    ug_weight=0.2,
)

images[-1].save(f"results/{name}_finalout.png")
```

## Main Parameters

- `tar_height`, `tar_width`: target output height and width. They must be divisible by 8. For SDXL, multiples of 1024 are recommended for stable multiscale behavior.
- `view_batch_size`: batch size for patch denoising paths. Larger values are faster but require more GPU memory.
- `stride`: stride for local latent patches. Smaller values reduce boundary artifacts but increase runtime.
- `multi_decoder`: use tiled decoding. This is recommended for resolutions above 3072 x 3072.
- `beta_scale_1`: weight schedule for global-local consistency denoising.
- `beta_scale_2`: weight schedule for local, upsample-guided, and dilated sampling.
- `cosine_scale_3`, `sigma`: Gaussian filtering controls used in dilated global sampling.
- `run_stage`: multiscale schedule. `two` runs the base stage and target stage; `three` adds an intermediate stage; `S` runs multiple even scale stages.
- `ug_weight`: strength of patch-based upsample guidance sampling.
- `save_image_tag`: save intermediate masks and results for debugging.

## Optional Multimodal Examples

The official pipeline remains `pipeline_ultradiffedit_sdxl.py`. Clean public examples for the supplemental multimodal settings are provided under `examples/`:

- `examples/controlnet_canny.py`: Canny-conditioned editing. The default Canny thresholds are 100 and 200, matching the supplemental setup.
- `examples/controlnet_depth.py`: DPT depth-conditioned editing.
- `examples/controlnet_pose.py`: OpenPose-conditioned editing.
- `examples/ip_adapter_ultra.py`: IP-Adapter image-prompt editing. This requires the CLIP image encoder path and IP-Adapter SDXL checkpoint path.

Example:

```bash
python examples/controlnet_canny.py \
  --image path/to/image.png \
  --mask path/to/mask.png \
  --prompt "a restored stone archway" \
  --target_width 4096 \
  --target_height 4096 \
  --output results/canny_4k.png
```

For ControlNet examples, the script first creates a 1K ControlNet proposal, composites the generated region with the input mask, and then refines it with UltraDiffEdit at the target resolution. The raw historical `DemoFusion-main/` experiment folder is intentionally ignored and is not required for the public examples.

## Repository Notes

- `pipeline_ultradiffedit_sdxl.py` is the cleaned SDXL UltraDiffEdit pipeline and should be used for normal inference.
- `demo.py` is a minimal smoke-test script for the official SDXL pipeline.
- `examples/` contains the cleaned runnable examples for ControlNet and IP-Adapter use cases.
- `tools/mask_gui.py` is a small optional utility for drawing binary inpainting masks.
- Historical backups and ablation scripts are preserved on the `legacy` branch.
- `pietorch_local/` provides the Poisson blending utility used by the final image blending step.

## Dataset

We construct three high-resolution image editing benchmarks covering 2K to 8K resolution. The benchmarks provide paired images, text prompts, editing masks, dilated masks, and conditional maps for the multimodal editing settings evaluated in the paper.

| Benchmark | Source and Scale | Released Contents | Download |
| --- | --- | --- | --- |
| DIV2KEdit | 100 real 2K images from the DIV2K validation set. | Images, BLIP-2 prompts, manually adjusted object masks, dilated masks, and edge/sketch maps. | [Baidu Netdisk](https://pan.baidu.com/s/1L8_gIIzCn2gEpei7kuuy3A?pwd=xcfy), password: `xcfy`; [Google Drive](https://drive.google.com/file/d/1KDWC_Ysb3C33_xkeyG--oLgI2qhqC7OO/view?usp=sharing). |
| Syn2KEdit | 100 synthetic 2048 x 2048 images generated from ChatGPT prompts and DemoFusion, covering diverse styles, scenes, and objects. | Images, text prompts, editing masks, dilated masks, and generated condition maps. | [Baidu Netdisk](https://pan.baidu.com/s/1o46DmbLBfUgIfl_feZtlKQ?pwd=ign3), password: `ign3`; [Google Drive](https://drive.google.com/file/d/1e6M7FHtN53j8PR3OQk9eXIPp9G-lKBer/view?usp=sharing). |
| UHRSDEdit | 988 real images from the UHRSD test set, with resolutions ranging from 4K to 8K. | Images, prompts, saliency-derived editing masks, dilated masks, and generated condition maps. | [Baidu Netdisk](https://pan.baidu.com/s/1NQecPyqp22uKdJKfpoFlGA?pwd=b69w), password: `b69w`; Google Drive pending. |

After extraction, each dataset sample uses matched file names across subfolders. The local `DIV2K_edit/` example follows this organization:

```text
DIV2K_edit/
  img/
  mask/
  mask_dilate/
  sketch/
  caption/
```

`caption/` stores text prompts, `mask/` stores editing masks, `mask_dilate/` stores the dilated masks used for sampling, and `sketch/` stores edge/sketch control maps. Other conditional inputs used by the supplemental ControlNet settings, such as depth maps and pose keypoints, are provided or generated only where applicable. The public benchmark names are `DIV2KEdit`, `Syn2KEdit`, and `UHRSDEdit`; local archive or folder names may use underscores for convenience.

## Citation

If you find this project useful, please cite:

```bibtex
@article{lu2026tuning,
  title={Tuning-Free Latent Diffusion Models for Ultrahigh-Resolution Image Editing},
  author={Lu, Wanglong and Su, Lingming and Shi, Kaijie and Gong, Minglun and Jin, Xiaogang and Zhao, Hanli and Jiang, Xianta},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2026},
  doi={10.1109/TNNLS.2026.3707463}
}
```

## Acknowledgements

This project builds on [Diffusers](https://github.com/huggingface/diffusers) and is inspired by high-resolution diffusion generation work such as [DemoFusion](https://github.com/PRIS-CV/DemoFusion). We thank the authors and maintainers for their open-source contributions.

## License

This code is released under the [Apache License 2.0](LICENSE). Model weights and third-party checkpoints are subject to their own licenses.
