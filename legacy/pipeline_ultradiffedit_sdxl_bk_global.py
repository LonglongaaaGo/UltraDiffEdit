# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch


from pietorch_local import blend

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    is_invisible_watermark_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from Util.GaussianBlurLayers import ConfidenceDrivenMaskLayer
if is_invisible_watermark_available():
    from .watermark import StableDiffusionXLWatermarker



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        from diffusers.utils import load_image
        import pipeline_ultradiffedit_sdxl
        import time
        import torch
        import os
    
        name_ = str(time.time())
        os.makedirs("./results", exist_ok=True)
        pipe = pipeline_ultradiffedit_sdxl.StableAnysizeInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        
        pipe.to("cuda")
    
        img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    
        height = 2048
        width = 2048
    
        init_image = load_image(img_url).resize((width, height))
        mask_image = load_image(mask_url).resize((width, height))
    
        # prompt = "A majestic tiger sitting on a bench"
        prompt = "a cute cat sitting on a bench"
        negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
    
        generator = torch.Generator(device='cuda')  # random seed generator
        generator = generator.manual_seed(5)
    
        import time
    
        start = time.time()
        images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image, 
                mask_image=mask_image, 
                num_inference_steps=50, strength=0.80,
                generator = generator,
                tar_height=height,
                tar_width=width,
                view_batch_size=16,
                stride=64,
                beta_scale_1=3, beta_scale_2=1, 
                cosine_scale_3=1, sigma=0.8,
                multi_decoder=True, show_image=False,
                save_image_tag = False,
                file_name=name_,
                save_root='./results',
                ug_weight = 0.2,
            )
    
        images[-1].save(f"results/{name_}_finalout.png")
    
        end = time.time()
        print('time for running is : %s Seconds' % (end - start))
"""


def gaussian_kernel(kernel_size=3, sigma=1.0, channels=3):
    x_coord = torch.arange(kernel_size)
    gaussian_1d = torch.exp(-(x_coord - (kernel_size - 1) / 2) ** 2 / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
    kernel = gaussian_2d[None, None, :, :].repeat(channels, 1, 1, 1)

    return kernel


def gaussian_filter(latents, kernel_size=3, sigma=1.0):
    channels = latents.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, channels).to(latents.device, latents.dtype)
    blurred_latents = F.conv2d(latents, kernel, padding=kernel_size // 2, groups=channels)

    return blurred_latents


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def mask_pil_to_torch(mask, height, width):
    # preprocess mask
    if isinstance(mask, (PIL.Image.Image, np.ndarray)):
        mask = [mask]

    if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
        mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
        mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
        mask = mask.astype(np.float32) / 255.0
    elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
        mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

    mask = torch.from_numpy(mask)
    return mask


def prepare_mask_and_masked_image(image, mask, height, width, return_image: bool = False):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """

    # checkpoint. TOD(Yiyi) - need to clean this up later
    deprecation_message = "The prepare_mask_and_masked_image method is deprecated and will be removed in a future version. Please use VaeImageProcessor.preprocess instead"
    deprecate(
        "prepare_mask_and_masked_image",
        "0.30.0",
        deprecation_message,
    )
    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            mask = mask_pil_to_torch(mask, height, width)

        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        # assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        # if image.min() < -1 or image.max() > 1:
        #    raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        mask = mask_pil_to_torch(mask, height, width)
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

    if image.shape[1] == 4:
        # images are in latent space and thus can't
        # be masked set masked_image to None
        # we assume that the checkpoint is not an inpainting
        # checkpoint. TOD(Yiyi) - need to clean this up later
        masked_image = None
    else:
        masked_image = image * (mask < 0.5)

    # n.b. ensure backwards compatibility as old function does not return image
    if return_image:
        return mask, masked_image, image

    return mask, masked_image


class StableAnysizeInpaintPipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *LoRA*: [`loaders.LoraLoaderMixin.load_lora_weights`]
        - *Ckpt*: [`loaders.FromSingleFileMixin.from_single_file`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.LoraLoaderMixin.save_lora_weights`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        requires_aesthetics_score (`bool`, *optional*, defaults to `"False"`):
            Whether the `unet` requires a aesthetic_score condition to be passed during inference. Also see the config
            of `stabilityai/stable-diffusion-xl-refiner-1-0`.
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
    """
    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"

    _optional_components = ["tokenizer", "text_encoder"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        requires_aesthetics_score: bool = False,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.default_sample_size = self.unet.config.sample_size

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

        # self.confidence_mask_layer = ConfidenceDrivenMaskLayer(size=65, sigma=1.0 / 40, iters=7, pad=32)
        # self.confidence_mask_layer = ConfidenceDrivenMaskLayer(size=65, sigma=1.0 / 40, iters=7, pad=32)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(
                    text_input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt, negative_prompt_2]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        strength,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        add_noise=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if image.shape[1] == 4:
            image_latents = image.to(device=device, dtype=dtype)
        elif return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(image=image, generator=generator)

        if latents is None and add_noise:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        elif add_noise:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma
        else:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = image_latents.to(device)

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        dtype = image.dtype
        if self.vae.config.force_upcast:
            image = image.float()
            self.vae.to(dtype=torch.float32)

        if isinstance(generator, list):
            image_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self.vae.encode(image).latent_dist.sample(generator=generator)

        if self.vae.config.force_upcast:
            self.vae.to(dtype)

        image_latents = image_latents.to(dtype)
        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask

        if masked_image is not None and masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = None

        if masked_image is not None:
            if masked_image_latents is None:
                masked_image = masked_image.to(device=device, dtype=dtype)
                masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

            if masked_image_latents.shape[0] < batch_size:
                if not batch_size % masked_image_latents.shape[0] == 0:
                    raise ValueError(
                        "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                        f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                        " Make sure the number of images that you pass is divisible by the total requested batch size."
                    )
                masked_image_latents = masked_image_latents.repeat(
                    batch_size // masked_image_latents.shape[0], 1, 1, 1
                )

            masked_image_latents = (
                torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
            )

            # aligning device to prevent device errors when concating it with the latent model input
            masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        return mask, masked_image_latents

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            t_start = 0

        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_start * self.scheduler.config.num_train_timesteps)
                )
            )
            timesteps = list(filter(lambda ts: ts < discrete_timestep_cutoff, timesteps))
            return torch.tensor(timesteps), len(timesteps)

        return timesteps, num_inference_steps - t_start

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline._get_add_time_ids
    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype,
    ):
        if self.config.requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        elif expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    def get_views(self, height, width, window_size=128, stride=64, random_jitter=False):
        # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
        # if panorama's height/width < window_size, num_blocks of height/width should return 1
        height //= self.vae_scale_factor
        width //= self.vae_scale_factor
        num_blocks_height = int((height - window_size) / stride - 1e-6) + 2 if height > window_size else 1
        num_blocks_width = int((width - window_size) / stride - 1e-6) + 2 if width > window_size else 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size
            # if the boundry is exceeded, move back
            if h_end > height:
                h_start = int(h_start + height - h_end)
                h_end = int(height)
            if w_end > width:
                w_start = int(w_start + width - w_end)
                w_end = int(width)
            if h_start < 0:
                h_end = int(h_end - h_start)
                h_start = 0
            if w_start < 0:
                w_end = int(w_end - w_start)
                w_start = 0

            if random_jitter: #why use jitter?
                jitter_range = (window_size - stride) // 4
                w_jitter = 0
                h_jitter = 0
                if (w_start != 0) and (w_end != width):
                    w_jitter = random.randint(-jitter_range, jitter_range)
                elif (w_start == 0) and (w_end != width):
                    w_jitter = random.randint(-jitter_range, 0)
                elif (w_start != 0) and (w_end == width):
                    w_jitter = random.randint(0, jitter_range)
                if (h_start != 0) and (h_end != height):
                    h_jitter = random.randint(-jitter_range, jitter_range)
                elif (h_start == 0) and (h_end != height):
                    h_jitter = random.randint(-jitter_range, 0)
                elif (h_start != 0) and (h_end == height):
                    h_jitter = random.randint(0, jitter_range)
                h_start += (h_jitter + jitter_range)
                h_end += (h_jitter + jitter_range)
                w_start += (w_jitter + jitter_range)
                w_end += (w_jitter + jitter_range)

            views.append((h_start, h_end, w_start, w_end))
        return views

    def views_filter(self,views,mask):

        views_filtered = []
        for view in views:
            h_start, h_end, w_start, w_end = view
            if torch.sum(mask[:, :, h_start:h_end, w_start:w_end])==0: continue
            views_filtered.append(view)
            # for h_start, h_end, w_start, w_end in batch_view
            # torch.sum()
        return views_filtered




    def get_img_views(self, height, width, window_size=1024, stride=512, random_jitter=False):
        # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
        # if panorama's height/width < window_size, num_blocks of height/width should return 1
        # height //= self.vae_scale_factor
        # width //= self.vae_scale_factor
        num_blocks_height = int((height - window_size) / stride - 1e-6) + 2 if height > window_size else 1
        num_blocks_width = int((width - window_size) / stride - 1e-6) + 2 if width > window_size else 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size
            # if the boundry is exceeded, move back
            if h_end > height:
                h_start = int(h_start + height - h_end)
                h_end = int(height)
            if w_end > width:
                w_start = int(w_start + width - w_end)
                w_end = int(width)
            if h_start < 0:
                h_end = int(h_end - h_start)
                h_start = 0
            if w_start < 0:
                w_end = int(w_end - w_start)
                w_start = 0

            if random_jitter:  # why use jitter?
                jitter_range = (window_size - stride) // 4
                w_jitter = 0
                h_jitter = 0
                if (w_start != 0) and (w_end != width):
                    w_jitter = random.randint(-jitter_range, jitter_range)
                elif (w_start == 0) and (w_end != width):
                    w_jitter = random.randint(-jitter_range, 0)
                elif (w_start != 0) and (w_end == width):
                    w_jitter = random.randint(0, jitter_range)
                if (h_start != 0) and (h_end != height):
                    h_jitter = random.randint(-jitter_range, jitter_range)
                elif (h_start == 0) and (h_end != height):
                    h_jitter = random.randint(-jitter_range, 0)
                elif (h_start != 0) and (h_end == height):
                    h_jitter = random.randint(0, jitter_range)
                h_start += (h_jitter + jitter_range)
                h_end += (h_jitter + jitter_range)
                w_start += (w_jitter + jitter_range)
                w_end += (w_jitter + jitter_range)

            views.append((h_start, h_end, w_start, w_end))
        return views



    def multi_patch_encoding(self, latents, pre_img, device, current_height, current_width,
                         scale_masked_imgs, scale_masks, current_scale_num, output_type, show_image,
                         dtype, generator, sigma,
                         window_size=1024, stride=512,
                         random_jitter=True,
                         file_name = "multi_encoding_img"
                         ):

        print("### multipatch encoding ####")
        count_editing = torch.zeros_like(latents)
        value_editing = torch.zeros_like(latents)

        ## hard replacing
        # up sample the output from low-resolution
        cur_Up_img = F.interpolate(pre_img.to(device),size=(int(current_height), int(current_width)), mode='bicubic')
        # get the mask
        cur_mask = scale_masks[current_scale_num - 1]
        # get the softmask
        confidence_mask_layer = ConfidenceDrivenMaskLayer(size=(2 * current_scale_num * self.vae_scale_factor - 1),
                                                          sigma=sigma, iters=7,
                                                          pad=(current_scale_num) * self.vae_scale_factor - 1)
        soft_cur_mask = confidence_mask_layer(1 - cur_mask)
        # get the mask for this scale
        cur_mask = cur_mask + soft_cur_mask
        hard_inpaint_img = cur_Up_img
        ##
        latent_stride = stride // self.vae_scale_factor
        img_views = self.get_views(current_height, current_width, stride=latent_stride,window_size=self.unet.config.sample_size, random_jitter=True)

        if random_jitter == True:
            img_jitter_range = (window_size - stride) // 4
            hard_inpaint_img = F.pad(hard_inpaint_img, (img_jitter_range, img_jitter_range, img_jitter_range, img_jitter_range),
                                  'constant', 0)
            jitter_range = (window_size // self.vae_scale_factor - stride // self.vae_scale_factor) // 4
            count_editing = F.pad(count_editing, (jitter_range, jitter_range, jitter_range, jitter_range),
                                  'constant', 0)
            value_editing = F.pad(value_editing, (jitter_range, jitter_range, jitter_range, jitter_range),
                                  'constant', 0)
        # used for debug
        # if show_image:
        #     hard_img_show = self.image_processor.postprocess(hard_inpaint_img, output_type=output_type)
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(hard_img_show[0])
        #     plt.axis('off')  # Turn off axis numbers and ticks
        #     plt.show()
        # if self.save_image_tag:
        #     hard_img_show = self.image_processor.postprocess(hard_inpaint_img, output_type=output_type)
        #     hard_img_show[0].save(f"{self.save_root}/{file_name}_{current_scale_num}_multi_encoding_img.png")

        #
        with self.progress_bar(total=len(img_views)) as progress_bar:

            for ii, view in enumerate(img_views):
                (h_start, h_end, w_start, w_end) = view

                (l_h_start, l_h_end, l_w_start, l_w_end) = (h_start * self.vae_scale_factor, h_end * self.vae_scale_factor, w_start * self.vae_scale_factor,
                 w_end * self.vae_scale_factor)
    
                if True:
                    sub_img = hard_inpaint_img[:, :, l_h_start:l_h_end, l_w_start:l_w_end]
                    # used for debug
                    # show the sub images
                    # sub_img_show = self.image_processor.postprocess(sub_img, output_type=output_type)
                    # if show_image:
                    #     plt.figure(figsize=(10, 10))
                    #     plt.imshow(sub_img_show[0])
                    #     plt.axis('off')  # Turn off axis numbers and ticks
                    #     plt.show()
                    # encoding sub img
                    sub_latents = self._encode_vae_image(sub_img.to(dtype=dtype), generator=generator)
                    value_editing[:, :, h_start:h_end, w_start:w_end] += sub_latents
                    count_editing[:, :, h_start:h_end, w_start:w_end] += 1

                    progress_bar.update()

            if random_jitter == True:
                jitter_range = (window_size // self.vae_scale_factor - stride // self.vae_scale_factor) // 4
                # after all views, you may get all the local patches, so the jetter is trying to reduce the boundary issues
                value_editing = value_editing[:, :,
                                jitter_range: jitter_range + current_height // self.vae_scale_factor,
                                jitter_range: jitter_range + current_width // self.vae_scale_factor]
                count_editing = count_editing[:, :,
                                jitter_range: jitter_range + current_height // self.vae_scale_factor,
                                jitter_range: jitter_range + current_width // self.vae_scale_factor]

        print("### multipatch encoding  done!####")
        # get average
        value_editing = value_editing / (count_editing)

        return value_editing

    def tiled_decode(self, latents, current_height, current_width):
        sample_size = self.unet.config.sample_size
        core_size = self.unet.config.sample_size // 4  # 32
        core_stride = core_size    # 32
        pad_size = self.unet.config.sample_size // 8 * 3   #16*3
        decoder_view_batch_size = 1

        if self.lowvram:
            core_stride = core_size // 2
            pad_size = core_size

        # views for 32 *32 patches
        views = self.get_views(current_height, current_width, stride=core_stride, window_size=core_size)
        views_batch = [views[i: i + decoder_view_batch_size] for i in range(0, len(views), decoder_view_batch_size)]
        latents_ = F.pad(latents, (pad_size, pad_size, pad_size, pad_size), 'constant', 0) # 256 + 48*2
        image = torch.zeros(latents.size(0), 3, current_height, current_width).to(latents.device) #init images
        count = torch.zeros_like(image).to(latents.device)  # init count
        # get the latents corresponding to the current view coordinates
        with self.progress_bar(total=len(views_batch)) as progress_bar:
            for j, batch_view in enumerate(views_batch):
                vb_size = len(batch_view)
                latents_for_view = torch.cat(
                    [
                        latents_[:, :, h_start:h_end + pad_size * 2, w_start:w_end + pad_size * 2] # 32+48*2 =128
                        for h_start, h_end, w_start, w_end in batch_view
                    ]
                ).to(self.vae.device)
                image_patch = self.vae.decode(latents_for_view / self.vae.config.scaling_factor, return_dict=False)[0]
                h_start, h_end, w_start, w_end = views[j]
                h_start, h_end, w_start, w_end = h_start * self.vae_scale_factor, h_end * self.vae_scale_factor, w_start * self.vae_scale_factor, w_end * self.vae_scale_factor
                p_h_start, p_h_end, p_w_start, p_w_end = pad_size * self.vae_scale_factor, image_patch.size(
                    2) - pad_size * self.vae_scale_factor, pad_size * self.vae_scale_factor, image_patch.size(
                    3) - pad_size * self.vae_scale_factor
                image[:, :, h_start:h_end, w_start:w_end] += image_patch[:, :, p_h_start:p_h_end, p_w_start:p_w_end].to(
                    latents.device)
                count[:, :, h_start:h_end, w_start:w_end] += 1
                progress_bar.update()
        image = image / count

        return image

    def get_start_size(self,width,height,fix_size=1024):
        w, h = width,height
        aspect_ratio = w / h
        size = (min(fix_size, int(fix_size * aspect_ratio)),
                min(fix_size, int(fix_size / aspect_ratio)))

        return size

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.FloatTensor = None,
        strength: float = 0.9999,
        num_inference_steps: int = 50,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        # return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        ####
        tar_height: Optional[int] = None,  ## image heihgt for editing
        tar_width: Optional[int] = None,
        view_batch_size: int = 16,
        multi_decoder: bool = True,
        stride: Optional[int] = 64,
        beta_scale_1: Optional[float] = 3.,
        beta_scale_2: Optional[float] = 1.,
        cosine_scale_3: Optional[float] = 1.,
        sigma: Optional[float] = 1.0,
        show_image: bool = False,
        multi_patch_encoding_window_size: int = 1024,
        multi_patch_encoding_stride: int = 512,
        # for saving imges,
        file_name = "out_img",
        save_root = "results",
        save_image_tag = False,
        vis_analysis = False,
        run_stage = "two",
        ug_weight=0.2,
        orig_size = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            mask_image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted
                to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
                instead of 3, so the expected shape would be `(B, H, W, 1)`.
            strength (`float`, *optional*, defaults to 0.9999):
                Conceptually, indicates how much to transform the masked portion of the reference `image`. Must be
                between 0 and 1. `image` will be used as a starting point, adding more noise to it the larger the
                `strength`. The number of denoising steps depends on the amount of noise initially added. When
                `strength` is 1, added noise will be maximum and the denoising process will run for the full number of
                iterations specified in `num_inference_steps`. A value of 1, therefore, essentially ignores the masked
                portion of the reference `image`. Note that in the case of `denoising_start` being declared as an
                integer, the value of `strength` will be ignored.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            denoising_start (`float`, *optional*):
                When specified, indicates the fraction (between 0.0 and 1.0) of the total denoising process to be
                bypassed before it is initiated. Consequently, the initial part of the denoising process is skipped and
                it is assumed that the passed `image` is a partly denoised image. Note that when this is specified,
                strength will be ignored. The `denoising_start` parameter is particularly beneficial when this pipeline
                is integrated into a "Mixture of Denoisers" multi-pipeline setup, as detailed in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output).
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise (ca. final 20% of timesteps still needed) and should be
                denoised by a successor pipeline that has `denoising_start` set to 0.8 so that it only denoises the
                final 20% of the scheduler. The denoising_end parameter should ideally be utilized when this pipeline
                forms a part of a "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output).
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            aesthetic_score (`float`, *optional*, defaults to 6.0):
                Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
                Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_aesthetic_score (`float`, *optional*, defaults to 2.5):
                Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). Can be used to
                simulate an aesthetic score of the generated image by influencing the negative text condition.
            ################### UltraDiffEdit specific parameters ####################
            tar_height (`int`):
                The height in pixels of the edited image. This is set to Integer multiples of 1024.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            tar_width (`int`):
                The width in pixels of the generated image.This is set to Integer multiples of 1024.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            view_batch_size (`int`, defaults to 16):
                The batch size for multiple denoising paths. Typically, a larger batch size can result in higher 
                efficiency but comes with increased GPU memory requirements.
            multi_decoder (`bool`, defaults to True):
                Determine whether to use a tiled decoder. Generally, when the resolution exceeds 3072x3072, 
                a tiled decoder becomes necessary.
            stride (`int`, defaults to 64):
                The stride of moving local patches. A smaller stride is better for alleviating seam issues,
                but it also introduces additional computational overhead and inference time.
            beta_scale_1 (`float`, defaults to 3):
                Control the weights of diffused and denoised latent maps in global-local consistent denoising. 
            beta_scale_2 (`float`, defaults to 1):
                Control the weights of patch-based sampling, patch-based upsample guidance sampling, and dilated sampling in patch-based hybrid sampling. 
            cosine_scale_3 (`float`, defaults to 1):
                Control the strength of the gaussion filter. For specific impacts, please refer to Appendix C
                in the DemoFusion paper.
            sigma (`float`, defaults to 1):
                The standard value of the gaussian filter.
            show_image (`bool`, defaults to False):
                Determine whether to show intermediate results during generation.
            file_name (`str`, defaults to ``out_img``):
                The file prefix of the saved output image.
            save_root (`str`, defaults to ``results``):
                The root path of the saved output image.
            save_image_tag (`bool`, defaults to False):
                Determine whether to save intemediate results, such as masked images, and masks.
            vis_analysis (`bool`, defaults to False):
                Determine whether to save intemediate denoised images.
            run_stage (`str`, defaults to ``two``):
                define the phase set, using ``two``  to set the two stages, ``three``  to set the three stages, and ``S``  to set the S stages.
            ug_weight (`float`, defaults to 0.2):
                the weight used for the patch-based upsample guidnace sampling.

        Examples:

        Returns:
            a `list` with the generated images at each phase.
        """
        # 0. Default height and width to unet
        # height = height or self.unet.config.sample_size * self.vae_scale_factor
        # width = width or self.unet.config.sample_size * self.vae_scale_factor

        print(f"we use beta1",  beta_scale_1)
        print(f"we use beta2",  beta_scale_2)
        print(f"we use ug_weight", ug_weight)
        print("the size of the window is :", self.unet.config.sample_size)
        print("the stride of the window is :", stride)
        
        # multi_patch_encoding_window_size= self.unet.config.sample_size*self.vae_scale_factor
        # multi_patch_encoding_stride= stride*self.vae_scale_factor
        print("the multi_patch_encoding_window_size is :", multi_patch_encoding_window_size)
        print("the multi_patch_encoding_stride is :", multi_patch_encoding_stride)

        import time
        start_time = time.time()
        ####set the size for the first stage (initial size for the first stage editing)
        x1_size = self.default_sample_size * self.vae_scale_factor
        x1_width, x1_hight = self.get_start_size( tar_width,tar_height, fix_size=x1_size)
        # the initial size
        width = x1_width
        height = x1_hight

        if orig_size!= None:
            self.orig_size  =orig_size
        else:
            self.orig_size  =[tar_height,tar_width]

        #
        height_scale = tar_height / x1_hight
        width_scale = tar_width / x1_width
        scale_num = int(max(height_scale, width_scale))
        aspect_ratio = min(height_scale, width_scale) / max(height_scale, width_scale)
        #
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        self.lowvram = False
        ####
        self.save_root = save_root
        self.save_image_tag = save_image_tag
        self.vis_analysis = vis_analysis
        ##
        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            strength,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. set timesteps
        def denoising_value_valid(dnv):
            return isinstance(denoising_end, float) and 0 < dnv < 1

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device, denoising_start=denoising_start if denoising_value_valid else None
        )
        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Preprocess mask and image
        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=torch.float32)
        #
        mask = self.mask_processor.preprocess(mask_image, height=height, width=width)


        if masked_image_latents is not None:
            masked_image = masked_image_latents
        elif init_image.shape[1] == 4:
            # if images are in latent space, we can't mask it
            masked_image = None
        else:
            masked_image = init_image * (mask < 0.5) #mask>0.5 are mnipulated regions

        ###### get imgs, masks, and masked imgs for each scales
        scale_imgs = []
        scale_masks = []
        scale_masked_imgs = []
        for ii in range(scale_num):
            tmp_scale = (ii + 1)
            tmp_width = x1_width* tmp_scale
            tmp_height = x1_hight* tmp_scale

            if height > width:
                tmp_width = int(tmp_width * aspect_ratio)
            else:
                tmp_height = int(tmp_height * aspect_ratio)
            #
            tmp_image = self.image_processor.preprocess(image, height=tmp_height, width=tmp_width)
            tmp_image = tmp_image.to(dtype=torch.float32)
            tmp_mask = self.mask_processor.preprocess(mask_image, height=tmp_height, width=tmp_width)
            tmp_masked_image = tmp_image * (tmp_mask < 0.5)
            scale_imgs.append(tmp_image.to(device))
            scale_masks.append(tmp_mask.to(device))
            scale_masked_imgs.append(tmp_masked_image.to(device))

        #######

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        add_noise = True if denoising_start is None else False
        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            add_noise=add_noise,
            return_noise=True,
            return_image_latents=return_image_latents,
        )  # unmasked regions

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
            )
        # 8.1 Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # sounds there are no needs for these codes
        # height, width = latents.shape[-2:]
        # height = height * self.vae_scale_factor
        # width = width * self.vae_scale_factor

        # original_size = original_size or (height, width)
        # target_size = target_size or (height, width)

        # 10. Prepare added time ids & embeddings
        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size

        add_text_embeds = pooled_prompt_embeds
        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)

        # 11. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if (
            denoising_end is not None
            and denoising_start is not None
            and denoising_value_valid(denoising_end)
            and denoising_value_valid(denoising_start)
            and denoising_start >= denoising_end
        ):
            raise ValueError(
                f"`denoising_start`: {denoising_start} cannot be larger than or equal to `denoising_end`: "
                + f" {denoising_end} when using type float."
            )
        elif denoising_end is not None and denoising_value_valid(denoising_end):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]


        ###
        out_vis_analysisList = []
        image = self.image_processor.postprocess(masked_image, output_type=output_type)
        output_images = [image[0]]  # for saving images
        if self.save_image_tag:
            image[0].save(f"{self.save_root}/{file_name}_masked_img.png")


        ### first stage
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if num_channels_unet == 4:
                    init_latents_proper = image_latents[:1]
                    init_mask = mask[:1]

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )
                    # init_latents_proper: unmasked regions
                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        ## first stage done
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)


        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        # for the subsequent processing
        pre_img = image.clone()
        image = self.image_processor.postprocess(image, output_type=output_type)
        if show_image:
            plt.figure(figsize=(10, 10))
            plt.imshow(image[0])
            plt.axis('off')  # Turn off axis numbers and ticks
            plt.show()
        output_images.append(image[0])
        if self.save_image_tag:
            image[0].save(f"{self.save_root}/{file_name}_init_edit.png")

        ###
        end_time = time.time()
        print("___"*100)
        print('time for the first stage running is : %s Seconds' % (end_time - start_time))

        # while time
        whole_time = (end_time - start_time)    

        ####################################################### Phase Upscaling #####################################################
   
        if scale_num<=1: run_list = []
        else:
            run_list = [scale_num]
            if run_stage == "three":
                run_list = [int(scale_num/2)+1]+ run_list
            elif run_stage == "S":
                run_list = [current_scale_num for current_scale_num in range(2, scale_num + 1)]

        for idx_,current_scale_num in enumerate(run_list):
        # for current_scale_num in run_list:
            start_time = time.time()

            if self.lowvram:
                latents = latents.to(device)
                self.unet.to(device)
                torch.cuda.empty_cache()
            print("### Phase {} Denoising ###".format(current_scale_num))
            #
            current_height = x1_hight * current_scale_num
            current_width = x1_width * current_scale_num

            if height > width:
                current_width = int(current_width * aspect_ratio)
            else:
                current_height = int(current_height * aspect_ratio)

            # upsample encoding latent from previous stage
            latents = F.interpolate(latents.to(device), size=(
            int(current_height / self.vae_scale_factor), int(current_width / self.vae_scale_factor)), mode='bicubic')

            ################## for editing################## ################## ##################
            # multipatch encoding
            editing_latents_local = self.multi_patch_encoding(latents, pre_img, device, current_height, current_width,
                             scale_masked_imgs, scale_masks, current_scale_num, output_type, show_image,
                             latents.dtype, generator,sigma,
                             window_size=multi_patch_encoding_window_size, 
                             stride=multi_patch_encoding_stride,
                             file_name=file_name)

            latents = editing_latents_local.clone() ##
            # get the correspding mask using nearest intepolation
            cur_mask = torch.nn.functional.interpolate(
                mask[:1], size=(int(current_height / self.vae_scale_factor), int(current_width / self.vae_scale_factor))
            )
            cur_mask = cur_mask.to(device=device, dtype=latents.dtype) #0-1 mask
            confidence_mask_layer = ConfidenceDrivenMaskLayer(size=(2 * current_scale_num - 1), sigma=sigma, iters=7, pad=current_scale_num-1)
            soft_cur_mask = confidence_mask_layer(1-cur_mask)
            cur_mask= cur_mask+soft_cur_mask

            # ####### debug the cur_img and the soft_cur_mask
            # image = self.image_processor.postprocess(soft_cur_mask, output_type=output_type)
            # if show_image:
            #     plt.figure(figsize=(10, 10))
            #     plt.imshow(image[0])
            #     plt.axis('off')  # Turn off axis numbers and ticks
            #     plt.show()
            # output_images.append(image[0])
            # if self.save_image_tag:
            #     image[0].save(f"{self.save_root}/{file_name}_{current_scale_num}_soft_cur_mask.png")

            # #
            # image = self.image_processor.postprocess(cur_mask, output_type=output_type)
            # if show_image:
            #     plt.figure(figsize=(10, 10))
            #     plt.imshow(image[0])
            #     plt.axis('off')  # Turn off axis numbers and ticks
            #     plt.show()
            # output_images.append(image[0])
            # if self.save_image_tag:
            #     image[0].save(f"{self.save_root}/{file_name}_{current_scale_num}_cur_mask.png")

            #########
            ##
            noise_latents = []
            noise = torch.randn_like(latents)
            if self.vis_analysis:
                if needs_upcasting:
                    self.upcast_vae()
                    latents_aa = noise.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
                tmp_image = self.tiled_decode(latents_aa, current_height, current_width)
                image = self.image_processor.postprocess(tmp_image, output_type=output_type)
                if show_image:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image[0])
                    plt.axis('off')  # Turn off axis numbers and ticks
                    plt.show()
                out_vis_analysisList.append((image[0],0))
                # cast back to fp16 if needed
                if needs_upcasting:
                    self.vae.to(dtype=torch.float16)

            # save the noise
            # diffusion
            for ii,timestep in enumerate(timesteps):
                noise_latent = self.scheduler.add_noise(latents, noise, timestep.unsqueeze(0))
                noise_latents.append(noise_latent)
            #
            latents = noise_latents[0]
            #
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    count = torch.zeros_like(latents)
                    value = torch.zeros_like(latents)
                    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * (
                                self.scheduler.config.num_train_timesteps - t) / self.scheduler.config.num_train_timesteps)).cpu()

                    c1 = cosine_factor ** beta_scale_1
                    # global-local consistency denoising
                    latents = latents * (1 - c1) + noise_latents[i] * c1

                    cur_edit_latent = noise_latents[i]
                    # cur_edit_latent = (noise_latents[i] - noise_latents[i].mean()) / noise_latents[i].std() * latents.std() + latents.mean()
                    latents = latents * cur_mask + cur_edit_latent * (1 - cur_mask)
                    #
                    ### debug # visualize the intermediate maps
                    # image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                    # image = self.image_processor.postprocess(image, output_type=output_type)
                    # output_images.append(image[0])
                    # if self.save_image_tag:
                    #     image[0].save(f"{self.save_root}/{file_name}_{current_scale_num}_editing_latent_{t}.png")
                    #######

                    ############################################# MultiDiffusion #############################################
                    # # first get the coordinates of each patch
                    # views = self.get_views(current_height, current_width, stride=stride,
                    #                        window_size=self.unet.config.sample_size, random_jitter=True)

                    # # then group batches based on view_barch_size
                    # views_batch = [views[i: i + view_batch_size] for i in range(0, len(views), view_batch_size)]

                    # # add jitter
                    # jitter_range = (self.unet.config.sample_size - stride) // 4
                    # latents_ = F.pad(latents, (jitter_range, jitter_range, jitter_range, jitter_range), 'constant', 0)

                    # count_local = torch.zeros_like(latents_)
                    # value_local = torch.zeros_like(latents_)

                    # # for all batches, each time we only for (view_barch_size) batch latents
                    # for j, batch_view in enumerate(views_batch):
                    #     vb_size = len(batch_view)

                    #     # get the latents corresponding to the current view coordinates
                    #     latents_for_view = torch.cat(
                    #         [
                    #             latents_[:, :, h_start:h_end, w_start:w_end]
                    #             for h_start, h_end, w_start, w_end in batch_view
                    #         ]
                    #     )

                    #     # expand the latents if we are doing classifier free guidance
                    #     latent_model_input = latents_for_view
                    #     latent_model_input = (
                    #         latent_model_input.repeat_interleave(2, dim=0)
                    #         if do_classifier_free_guidance
                    #         else latent_model_input
                    #     )
                    #     latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    #     prompt_embeds_input = torch.cat([prompt_embeds] * vb_size)
                    #     add_text_embeds_input = torch.cat([add_text_embeds] * vb_size)
                    #     add_time_ids_input = []
                    #     for h_start, h_end, w_start, w_end in batch_view:
                    #         add_time_ids_ = add_time_ids.clone()
                    #         add_time_ids_[:, 2] = h_start * self.vae_scale_factor
                    #         add_time_ids_[:, 3] = w_start * self.vae_scale_factor
                    #         add_time_ids_input.append(add_time_ids_)
                    #     add_time_ids_input = torch.cat(add_time_ids_input)

                    #     # predict the noise residual
                    #     added_cond_kwargs = {"text_embeds": add_text_embeds_input, "time_ids": add_time_ids_input}
                    #     noise_pred = self.unet(
                    #         latent_model_input,
                    #         t,
                    #         encoder_hidden_states=prompt_embeds_input,
                    #         cross_attention_kwargs=cross_attention_kwargs,
                    #         added_cond_kwargs=added_cond_kwargs,
                    #         return_dict=False,
                    #     )[0]

                    #     if do_classifier_free_guidance:
                    #         noise_pred_uncond, noise_pred_text = noise_pred[::2], noise_pred[1::2]
                    #         noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    #     if do_classifier_free_guidance and guidance_rescale > 0.0:
                    #         # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    #         noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text,
                    #                                        guidance_rescale=guidance_rescale)

                    #     # compute the previous noisy sample x_t -> x_t-1
                    #     self.scheduler._init_step_index(t)
                    #     latents_denoised_batch = self.scheduler.step(
                    #         noise_pred, t, latents_for_view, **extra_step_kwargs, return_dict=False)[0]

                    #     # extract value from batch
                    #     for latents_view_denoised, (h_start, h_end, w_start, w_end) in zip(
                    #             latents_denoised_batch.chunk(vb_size), batch_view
                    #     ):
                    #         value_local[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                    #         count_local[:, :, h_start:h_end, w_start:w_end] += 1

                    # # after all views, you may get all the local patches, so the jetter is trying to reduce the boundary issues
                    # value_local = value_local[:, :,
                    #               jitter_range: jitter_range + current_height // self.vae_scale_factor,
                    #               jitter_range: jitter_range + current_width // self.vae_scale_factor]
                    # count_local = count_local[:, :,
                    #               jitter_range: jitter_range + current_height // self.vae_scale_factor,
                    #               jitter_range: jitter_range + current_width // self.vae_scale_factor]

                    # c2 = cosine_factor ** beta_scale_2 *0.5

                    # value += value_local / count_local * (1 - c2)
                    # count += torch.ones_like(value_local) * (1 - c2)

                    ############################################# LOCAL UG #############################################
                    # # P = αt + 1/m*m *(1−αt)
                    # # we emprically set to 2
                    # ug_scale = 2
                    # Alpha_t = self.scheduler.alphas[t.cpu().int()]
                    # P_ = Alpha_t + (1 - Alpha_t) / (ug_scale * ug_scale)
                    # P_sqrt = P_ ** 0.5
                    # #### for memory saving
                    # ug_view_batch_size= view_batch_size//4 
                    # # first get the coordinates of each patch
                    # views = self.get_views(current_height, current_width, stride=stride,
                    #                        window_size=self.unet.config.sample_size*ug_scale, random_jitter=True)

                    # # then group batches based on view_barch_size
                    # views_batch = [views[i: i + ug_view_batch_size] for i in range(0, len(views), ug_view_batch_size)]

                    # # why add jitter for ranges?
                    # jitter_range = (self.unet.config.sample_size*ug_scale - stride) // 4
                    # latents_ = F.pad(latents, (jitter_range, jitter_range, jitter_range, jitter_range), 'constant', 0)

                    # ug_count_local = torch.zeros_like(latents_)
                    # ug_value_local = torch.zeros_like(latents_)

                    # # for all batches, each time we only for (view_barch_size) batch latents
                    # for j, batch_view in enumerate(views_batch):
                    #     vb_size = len(batch_view)

                    #     # get the latents corresponding to the current view coordinates
                    #     latents_for_view = torch.cat(
                    #         [
                    #             latents_[:, :, h_start:h_end, w_start:w_end]
                    #             for h_start, h_end, w_start, w_end in batch_view
                    #         ]
                    #     )

                    #     # expand the latents if we are doing classifier free guidance
                    #     latent_model_input = latents_for_view
                    #     latent_model_input = (
                    #         latent_model_input.repeat_interleave(2, dim=0)
                    #         if do_classifier_free_guidance
                    #         else latent_model_input
                    #     )
                    #     latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    #     prompt_embeds_input = torch.cat([prompt_embeds] * vb_size)
                    #     add_text_embeds_input = torch.cat([add_text_embeds] * vb_size)
                    #     add_time_ids_input = []
                    #     for h_start, h_end, w_start, w_end in batch_view:
                    #         add_time_ids_ = add_time_ids.clone()
                    #         add_time_ids_[:, 2] = h_start * self.vae_scale_factor
                    #         add_time_ids_[:, 3] = w_start * self.vae_scale_factor
                    #         add_time_ids_input.append(add_time_ids_)
                    #     add_time_ids_input = torch.cat(add_time_ids_input)

                    #     # predict the noise residual
                    #     added_cond_kwargs = {"text_embeds": add_text_embeds_input, "time_ids": add_time_ids_input}
                    #     noise_pred = self.unet(
                    #         latent_model_input,
                    #         t,
                    #         encoder_hidden_states=prompt_embeds_input,
                    #         cross_attention_kwargs=cross_attention_kwargs,
                    #         added_cond_kwargs=added_cond_kwargs,
                    #         return_dict=False,
                    #     )[0]

                    #     if do_classifier_free_guidance:
                    #         noise_pred_uncond, noise_pred_text = noise_pred[::2], noise_pred[1::2]
                    #         noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    #     if do_classifier_free_guidance and guidance_rescale > 0.0:
                    #         # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    #         noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text,
                    #                                        guidance_rescale=guidance_rescale)

                    #     # # compute the previous noisy sample x_t -> x_t-1
                    #     # self.scheduler._init_step_index(t)
                    #     # latents_denoised_batch = self.scheduler.step(
                    #     #     noise_pred, t, latents_for_view, **extra_step_kwargs, return_dict=False)[0]

                    #     #######UG ###########
                    #     # get the latents corresponding to the current view coordinates
                    #     down_latents_for_view = torch.cat(
                    #         [
                    #             torch.nn.functional.avg_pool2d(latents_[:, :, h_start:h_end, w_start:w_end],
                    #                                            kernel_size=ug_scale, stride=ug_scale)
                    #             for h_start, h_end, w_start, w_end in batch_view
                    #         ]
                    #     )
                    #     ###########
                    #     # expand the latents if we are doing classifier free guidance
                    #     down_latent_model_input = down_latents_for_view
                    #     down_latent_model_input = (
                    #         down_latent_model_input.repeat_interleave(2, dim=0)
                    #         if do_classifier_free_guidance
                    #         else down_latent_model_input
                    #     )
                    #     down_latent_model_input = self.scheduler.scale_model_input(down_latent_model_input, t)
                    #     down_latent_model_input = down_latent_model_input / P_sqrt

                    #     # prompt_embeds_input = torch.cat([prompt_embeds] * vb_size)
                    #     # add_text_embeds_input = torch.cat([add_text_embeds] * vb_size)
                    #     # add_time_ids_input = []
                    #     # for h_start, h_end, w_start, w_end in batch_view:
                    #     #     add_time_ids_ = add_time_ids.clone()
                    #     #     add_time_ids_[:, 2] = h_start * self.vae_scale_factor
                    #     #     add_time_ids_[:, 3] = w_start * self.vae_scale_factor
                    #     #     add_time_ids_input.append(add_time_ids_)
                    #     # add_time_ids_input = torch.cat(add_time_ids_input)

                    #     # predict the noise residual
                    #     # added_cond_kwargs = {"text_embeds": add_text_embeds_input, "time_ids": add_time_ids_input}
                    #     down_noise_pred = self.unet(
                    #         down_latent_model_input,
                    #         t,
                    #         encoder_hidden_states=prompt_embeds_input,
                    #         cross_attention_kwargs=cross_attention_kwargs,
                    #         added_cond_kwargs=added_cond_kwargs,
                    #         return_dict=False,
                    #     )[0]
                    #     down_noise_pred = down_noise_pred / ug_scale

                    #     if do_classifier_free_guidance:
                    #         down_noise_pred_uncond, down_noise_pred_text = down_noise_pred[::2], down_noise_pred[1::2]
                    #         down_noise_pred = down_noise_pred_uncond + guidance_scale * (down_noise_pred_text - down_noise_pred_uncond)

                    #     if do_classifier_free_guidance and guidance_rescale > 0.0:
                    #         # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    #         down_noise_pred = rescale_noise_cfg(down_noise_pred, down_noise_pred_text,
                    #                                        guidance_rescale=guidance_rescale)

                    #     if t > 500:
                    #         down_value_noise_pred = torch.nn.functional.avg_pool2d(noise_pred, kernel_size=ug_scale,stride=ug_scale)
                    #         up_guidance = torch.nn.functional.interpolate(
                    #             (down_noise_pred - down_value_noise_pred), size=(
                    #                 int(noise_pred.shape[2]), int(noise_pred.shape[3])), mode="nearest"
                    #         )
                    #         # noise_pred = noise_pred + 0.2 * up_guidance
                    #         noise_pred = noise_pred + ug_weight * up_guidance

                            
                    #     # compute the previous noisy sample x_t -> x_t-1
                    #     self.scheduler._init_step_index(t)
                    #     latents_denoised_batch = self.scheduler.step(
                    #         noise_pred, t, latents_for_view, **extra_step_kwargs, return_dict=False)[0]
                    #     ############
                    #     # extract value from batch
                    #     for latents_view_denoised, (h_start, h_end, w_start, w_end) in zip(
                    #             latents_denoised_batch.chunk(vb_size), batch_view
                    #     ):
                    #         ug_value_local[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                    #         ug_count_local[:, :, h_start:h_end, w_start:w_end] += 1

                    # # after all views, you may get all the local patches, so the jetter is trying to reduce the boundary issues
                    # ug_value_local = ug_value_local[:, :,
                    #               jitter_range: jitter_range + current_height // self.vae_scale_factor,
                    #               jitter_range: jitter_range + current_width // self.vae_scale_factor]
                    # ug_count_local = ug_count_local[:, :,
                    #               jitter_range: jitter_range + current_height // self.vae_scale_factor,
                    #               jitter_range: jitter_range + current_width // self.vae_scale_factor]

                    # c2 = cosine_factor ** beta_scale_2 *0.5

                    # value += ug_value_local / ug_count_local * (1 - c2)
                    # count += torch.ones_like(ug_value_local) * (1 - c2)

         ############################################# Dilated Sampling #############################################

                    views = [[h, w] for h in range(current_scale_num) for w in range(current_scale_num)]
                    views_batch = [views[i: i + view_batch_size] for i in range(0, len(views), view_batch_size)]

                    h_pad = (current_scale_num - (latents.size(2) % current_scale_num)) % current_scale_num
                    w_pad = (current_scale_num - (latents.size(3) % current_scale_num)) % current_scale_num
                    latents_ = F.pad(latents, (w_pad, 0, h_pad, 0), 'constant', 0)

                    count_global = torch.zeros_like(latents_)
                    value_global = torch.zeros_like(latents_)

                    c3 = 0.99 * cosine_factor ** cosine_scale_3 + 1e-2
                    std_, mean_ = latents_.std(), latents_.mean()
                    latents_gaussian = gaussian_filter(latents_, kernel_size=(2 * current_scale_num - 1),
                                                       sigma=sigma * c3)
                    latents_gaussian = (latents_gaussian - latents_gaussian.mean()) / latents_gaussian.std() * std_ + mean_

                    for j, batch_view in enumerate(views_batch):
                        latents_for_view = torch.cat(
                            [
                                latents_[:, :, h::current_scale_num, w::current_scale_num]
                                for h, w in batch_view
                            ]
                        )
                        latents_for_view_gaussian = torch.cat(
                            [
                                latents_gaussian[:, :, h::current_scale_num, w::current_scale_num]
                                for h, w in batch_view
                            ]
                        )

                        vb_size = latents_for_view.size(0)

                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = latents_for_view_gaussian
                        latent_model_input = (
                            latent_model_input.repeat_interleave(2, dim=0)
                            if do_classifier_free_guidance
                            else latent_model_input
                        )
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        prompt_embeds_input = torch.cat([prompt_embeds] * vb_size)
                        add_text_embeds_input = torch.cat([add_text_embeds] * vb_size)
                        add_time_ids_input = torch.cat([add_time_ids] * vb_size)

                        # predict the noise residual
                        added_cond_kwargs = {"text_embeds": add_text_embeds_input, "time_ids": add_time_ids_input}
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds_input,
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred[::2], noise_pred[1::2]
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        if do_classifier_free_guidance and guidance_rescale > 0.0:
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text,
                                                           guidance_rescale=guidance_rescale)

                        # compute the previous noisy sample x_t -> x_t-1
                        self.scheduler._init_step_index(t)
                        latents_denoised_batch = self.scheduler.step(
                            noise_pred, t, latents_for_view, **extra_step_kwargs, return_dict=False)[0]

                        # extract value from batch
                        for latents_view_denoised, (h, w) in zip(
                                latents_denoised_batch.chunk(vb_size), batch_view
                        ):
                            value_global[:, :, h::current_scale_num, w::current_scale_num] += latents_view_denoised
                            count_global[:, :, h::current_scale_num, w::current_scale_num] += 1

                    c2 = cosine_factor ** beta_scale_2

                    value_global = value_global[:, :, h_pad:, w_pad:]

                    value += value_global * c2
                    count += torch.ones_like(value_global) * c2

                    ###########################################################
                    latents = torch.where(count > 0, value / count, value)
                    ###################################

                    if self.vis_analysis:
                        if needs_upcasting:
                            self.upcast_vae()
                            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
                        tmp_image = self.tiled_decode(latents, current_height, current_width)
                        image = self.image_processor.postprocess(tmp_image, output_type=output_type)
                        if show_image:
                            plt.figure(figsize=(10, 10))
                            plt.imshow(image[0])
                            plt.axis('off')  # Turn off axis numbers and ticks
                            plt.show()
                        out_vis_analysisList.append((image[0],t))
                        # cast back to fp16 if needed
                        if needs_upcasting:
                            self.vae.to(dtype=torch.float16)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)
        
  
                #########################################################################################################################################
                if self.lowvram:
                    latents = latents.cpu()
                    torch.cuda.empty_cache()
                if not output_type == "latent":
                    # make sure the VAE is in float32 mode, as it overflows in float16
                    needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

                    if self.lowvram:
                        needs_upcasting = False  # use madebyollin/sdxl-vae-fp16-fix in lowvram mode!
                        self.unet.cpu()
                        self.vae.to(device)

                    if needs_upcasting:
                        self.upcast_vae()
                        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

                    print("### Phase {} Decoding ###".format(current_scale_num))
                    if multi_decoder:
                        image = self.tiled_decode(latents, current_height, current_width)
                    else:
                        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

                    ## hard replacing
                    cur_masked_img = scale_masked_imgs[current_scale_num-1]
                    cur_mask = scale_masks[current_scale_num-1]
                    cur_mask = cur_mask.to(device=device, dtype=latents.dtype)  # 0-1 mask

                    confidence_mask_layer = ConfidenceDrivenMaskLayer(size=(2 * current_scale_num*self.vae_scale_factor - 1), sigma=sigma,iters=7, pad=(current_scale_num)*self.vae_scale_factor-1 )
                    soft_cur_mask = confidence_mask_layer(1 - cur_mask)
                    cur_mask = cur_mask + soft_cur_mask

                    #### final out
                    inpainted_image = cur_masked_img * (1-cur_mask) + image * cur_mask

                    if current_scale_num == run_list[-1]:
                        print("process_images_with_poisson!!")
                        # aaa = image * cur_mask
                        # # hard_mask = cur_mask.clone().detach()
                        # inpainted_image = blend(inpainted_image[0].cpu(),image[0].cpu(),cur_mask[0,0].cpu(),
                        #                     torch.tensor([0, 0]).cpu(), True, channels_dim=0,restore_detail=True,data_range=[-1,1]).unsqueeze(0)
                        #
                        inpainted_image = inpainted_image[:,:,:self.orig_size[0],:self.orig_size[1]]
                        image = image[:,:,:self.orig_size[0],:self.orig_size[1]]
                        cur_mask = cur_mask[:,:,:self.orig_size[0],:self.orig_size[1]]

                        inpainted_image = blend(inpainted_image.detach().clone()[0].cpu(), image.detach().clone()[0].cpu(),
                                                cur_mask.detach().clone()[0, 0].cpu(),
                                                torch.tensor([0, 0]).cpu(), True, channels_dim=0, restore_detail=True,
                                                data_range=[-1, 1]).unsqueeze(0)
                        
                        #####
                        # inpainted_image = image
                        # print("process_images_with_poisson!!")
                        # pre_scale_num = run_list[idx_-1]
                        # pre_height = x1_hight * pre_scale_num
                        # pre_width = x1_width * pre_scale_num
                        # small_mask = scale_masks[pre_scale_num - 1]
                        #
                        # small_inpainted_image = F.interpolate(inpainted_image.detach().clone().to(device), size=(pre_height,pre_width),mode='bicubic')
                        #
                        # small_image = F.interpolate(image.detach().clone().to(device), size=(pre_height,pre_width),mode='bicubic')
                        #
                        # small_inpainted_image = blend(small_inpainted_image.detach().clone()[0].cpu(), small_image.detach().clone()[0].cpu(),
                        #                         small_mask.detach().clone()[0, 0].cpu(),
                        #                         torch.tensor([0, 0]).cpu(), True, channels_dim=0, restore_detail=True,
                        #                         data_range=[-1, 1]).unsqueeze(0)
                        #
                        #
                        # small_inpainted_image = F.interpolate(small_inpainted_image, size=(current_height,current_width),mode='bicubic')
                        #
                        # inpainted_image = cur_masked_img * (1 - cur_mask) + (image*0.5+small_inpainted_image*0.5) * cur_mask

                        
                    # cast back to fp16 if needed
                    if needs_upcasting:
                        self.vae.to(dtype=torch.float16)
                else:
                    image = latents

                # for subsequent processing
                pre_img_ = inpainted_image.clone()
                if not output_type == "latent":
                    image = self.image_processor.postprocess(image, output_type=output_type)
                    if show_image:
                        plt.figure(figsize=(10, 10))
                        plt.imshow(image[0])
                        plt.axis('off')  # Turn off axis numbers and ticks
                        plt.show()
                    output_images.append(image[0])
                    if self.save_image_tag:
                        image[0].save(f"{self.save_root}/{file_name}_{current_scale_num}_out.png")

                    inpainted_image = self.image_processor.postprocess(inpainted_image, output_type=output_type)
                    if show_image:
                        plt.figure(figsize=(10, 10))
                        plt.imshow(inpainted_image[0])
                        plt.axis('off')  # Turn off axis numbers and ticks
                        plt.show()
                    output_images.append(inpainted_image[0])

    
            pre_img = pre_img_
            #preserve for the next stage
            latents = editing_latents_local.clone().detach()  # for next step

            end_time = time.time()
            print("___"*100)
            print(f'time for the {current_scale_num} stage running is : %s Seconds' % (end_time - start_time))

            # while time
            whole_time += (end_time - start_time)
            print(f'the whole time for the {current_scale_num} stage is : %s Seconds' % whole_time)

        print("___"*100)
        print('the whole time is : %s Seconds' % whole_time)


        # Offload all models
        self.maybe_free_model_hooks()

        if self.vis_analysis:
            return output_images, out_vis_analysisList

        return output_images



    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def reconstraction_test(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.FloatTensor = None,
        strength: float = 0.9999,
        num_inference_steps: int = 50,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        # return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        ####
        tar_height: Optional[int] = None,  ## image heihgt for editing
        tar_width: Optional[int] = None,
        view_batch_size: int = 16,
        multi_decoder: bool = True,
        stride: Optional[int] = 64,
        beta_scale_1: Optional[float] = 3.,
        beta_scale_2: Optional[float] = 1.,
        cosine_scale_3: Optional[float] = 1.,
        sigma: Optional[float] = 1.0,
        show_image: bool = False,
        multi_patch_encoding_window_size: int = 1024,
        multi_patch_encoding_stride: int = 512,
        # for saving imges,
        file_name = "out_img",
        save_root = "results",
        save_image_tag = False,
        vis_analysis = False,
        run_stage = "two",
        ug_weight=0.2,
        orig_size = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            mask_image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted
                to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
                instead of 3, so the expected shape would be `(B, H, W, 1)`.
            strength (`float`, *optional*, defaults to 0.9999):
                Conceptually, indicates how much to transform the masked portion of the reference `image`. Must be
                between 0 and 1. `image` will be used as a starting point, adding more noise to it the larger the
                `strength`. The number of denoising steps depends on the amount of noise initially added. When
                `strength` is 1, added noise will be maximum and the denoising process will run for the full number of
                iterations specified in `num_inference_steps`. A value of 1, therefore, essentially ignores the masked
                portion of the reference `image`. Note that in the case of `denoising_start` being declared as an
                integer, the value of `strength` will be ignored.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            denoising_start (`float`, *optional*):
                When specified, indicates the fraction (between 0.0 and 1.0) of the total denoising process to be
                bypassed before it is initiated. Consequently, the initial part of the denoising process is skipped and
                it is assumed that the passed `image` is a partly denoised image. Note that when this is specified,
                strength will be ignored. The `denoising_start` parameter is particularly beneficial when this pipeline
                is integrated into a "Mixture of Denoisers" multi-pipeline setup, as detailed in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output).
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise (ca. final 20% of timesteps still needed) and should be
                denoised by a successor pipeline that has `denoising_start` set to 0.8 so that it only denoises the
                final 20% of the scheduler. The denoising_end parameter should ideally be utilized when this pipeline
                forms a part of a "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output).
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            aesthetic_score (`float`, *optional*, defaults to 6.0):
                Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
                Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_aesthetic_score (`float`, *optional*, defaults to 2.5):
                Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). Can be used to
                simulate an aesthetic score of the generated image by influencing the negative text condition.
            ################### UltraDiffEdit specific parameters ####################
            tar_height (`int`):
                The height in pixels of the edited image. This is set to Integer multiples of 1024.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            tar_width (`int`):
                The width in pixels of the generated image.This is set to Integer multiples of 1024.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            view_batch_size (`int`, defaults to 16):
                The batch size for multiple denoising paths. Typically, a larger batch size can result in higher 
                efficiency but comes with increased GPU memory requirements.
            multi_decoder (`bool`, defaults to True):
                Determine whether to use a tiled decoder. Generally, when the resolution exceeds 3072x3072, 
                a tiled decoder becomes necessary.
            stride (`int`, defaults to 64):
                The stride of moving local patches. A smaller stride is better for alleviating seam issues,
                but it also introduces additional computational overhead and inference time.
            beta_scale_1 (`float`, defaults to 3):
                Control the weights of diffused and denoised latent maps in global-local consistent denoising. 
            beta_scale_2 (`float`, defaults to 1):
                Control the weights of patch-based sampling, patch-based upsample guidance sampling, and dilated sampling in patch-based hybrid sampling. 
            cosine_scale_3 (`float`, defaults to 1):
                Control the strength of the gaussion filter. For specific impacts, please refer to Appendix C
                in the DemoFusion paper.
            sigma (`float`, defaults to 1):
                The standard value of the gaussian filter.
            show_image (`bool`, defaults to False):
                Determine whether to show intermediate results during generation.
            file_name (`str`, defaults to ``out_img``):
                The file prefix of the saved output image.
            save_root (`str`, defaults to ``results``):
                The root path of the saved output image.
            save_image_tag (`bool`, defaults to False):
                Determine whether to save intemediate results, such as masked images, and masks.
            vis_analysis (`bool`, defaults to False):
                Determine whether to save intemediate denoised images.
            run_stage (`str`, defaults to ``two``):
                define the phase set, using ``two``  to set the two stages, ``three``  to set the three stages, and ``S``  to set the S stages.
            ug_weight (`float`, defaults to 0.2):
                the weight used for the patch-based upsample guidnace sampling.

        Examples:

        Returns:
            a `list` with the generated images at each phase.
        """
        # 0. Default height and width to unet
        # height = height or self.unet.config.sample_size * self.vae_scale_factor
        # width = width or self.unet.config.sample_size * self.vae_scale_factor

        print(f"we use beta1",  beta_scale_1)
        print(f"we use beta2",  beta_scale_2)
        print(f"we use ug_weight", ug_weight)
        print("the size of the window is :", self.unet.config.sample_size)
        print("the stride of the window is :", stride)
        
        # multi_patch_encoding_window_size= self.unet.config.sample_size*self.vae_scale_factor
        # multi_patch_encoding_stride= stride*self.vae_scale_factor
        print("the multi_patch_encoding_window_size is :", multi_patch_encoding_window_size)
        print("the multi_patch_encoding_stride is :", multi_patch_encoding_stride)

        import time
        start_time = time.time()
        ####set the size for the first stage (initial size for the first stage editing)
        x1_size = self.default_sample_size * self.vae_scale_factor
        x1_width, x1_hight = self.get_start_size( tar_width,tar_height, fix_size=x1_size)
        # the initial size
        width = x1_width
        height = x1_hight

        if orig_size!= None:
            self.orig_size  =orig_size
        else:
            self.orig_size  =[tar_height,tar_width]

        #
        height_scale = tar_height / x1_hight
        width_scale = tar_width / x1_width
        scale_num = int(max(height_scale, width_scale))
        aspect_ratio = min(height_scale, width_scale) / max(height_scale, width_scale)
        #
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        self.lowvram = False
        ####
        self.save_root = save_root
        self.save_image_tag = save_image_tag
        self.vis_analysis = vis_analysis
        ##
        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            strength,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. set timesteps
        def denoising_value_valid(dnv):
            return isinstance(denoising_end, float) and 0 < dnv < 1

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device, denoising_start=denoising_start if denoising_value_valid else None
        )
        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Preprocess mask and image
        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=torch.float32)
        #
        mask = self.mask_processor.preprocess(mask_image, height=height, width=width)


        if masked_image_latents is not None:
            masked_image = masked_image_latents
        elif init_image.shape[1] == 4:
            # if images are in latent space, we can't mask it
            masked_image = None
        else:
            masked_image = init_image * (mask < 0.5) #mask>0.5 are mnipulated regions

        ###### get imgs, masks, and masked imgs for each scales
        scale_imgs = []
        scale_masks = []
        scale_masked_imgs = []
        for ii in range(scale_num):
            tmp_scale = (ii + 1)
            tmp_width = x1_width* tmp_scale
            tmp_height = x1_hight* tmp_scale

            if height > width:
                tmp_width = int(tmp_width * aspect_ratio)
            else:
                tmp_height = int(tmp_height * aspect_ratio)
            #
            tmp_image = self.image_processor.preprocess(image, height=tmp_height, width=tmp_width)
            tmp_image = tmp_image.to(dtype=torch.float32)
            tmp_mask = self.mask_processor.preprocess(mask_image, height=tmp_height, width=tmp_width)
            tmp_masked_image = tmp_image * (tmp_mask < 0.5)
            scale_imgs.append(tmp_image.to(device))
            scale_masks.append(tmp_mask.to(device))
            scale_masked_imgs.append(tmp_masked_image.to(device))

        #######

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        add_noise = True if denoising_start is None else False
        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            add_noise=add_noise,
            return_noise=True,
            return_image_latents=return_image_latents,
        )  # unmasked regions

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
            )
        # 8.1 Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # sounds there are no needs for these codes
        # height, width = latents.shape[-2:]
        # height = height * self.vae_scale_factor
        # width = width * self.vae_scale_factor

        # original_size = original_size or (height, width)
        # target_size = target_size or (height, width)

        # 10. Prepare added time ids & embeddings
        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size

        add_text_embeds = pooled_prompt_embeds
        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)

        # 11. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if (
            denoising_end is not None
            and denoising_start is not None
            and denoising_value_valid(denoising_end)
            and denoising_value_valid(denoising_start)
            and denoising_start >= denoising_end
        ):
            raise ValueError(
                f"`denoising_start`: {denoising_start} cannot be larger than or equal to `denoising_end`: "
                + f" {denoising_end} when using type float."
            )
        elif denoising_end is not None and denoising_value_valid(denoising_end):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]


        ###
        out_vis_analysisList = []
        image = self.image_processor.postprocess(masked_image, output_type=output_type)
        output_images = [image[0]]  # for saving images
        if self.save_image_tag:
            image[0].save(f"{self.save_root}/{file_name}_masked_img.png")


        ### first stage
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if num_channels_unet == 4:
                    init_latents_proper = image_latents[:1]
                    init_mask = mask[:1]

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )
                    # init_latents_proper: unmasked regions
                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        ## first stage done
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)


        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        # for the subsequent processing
        pre_img = image.clone()
        image = self.image_processor.postprocess(image, output_type=output_type)
        if show_image:
            plt.figure(figsize=(10, 10))
            plt.imshow(image[0])
            plt.axis('off')  # Turn off axis numbers and ticks
            plt.show()
        output_images.append(image[0])
        if self.save_image_tag:
            image[0].save(f"{self.save_root}/{file_name}_init_edit.png")

        ###
        end_time = time.time()
        print("___"*100)
        print('time for the first stage running is : %s Seconds' % (end_time - start_time))

        # while time
        whole_time = (end_time - start_time)    

        ####################################################### Phase Upscaling #####################################################
   
        if scale_num<=1: run_list = []
        else:
            run_list = [scale_num]
            if run_stage == "three":
                run_list = [int(scale_num/2)+1]+ run_list
            elif run_stage == "S":
                run_list = [current_scale_num for current_scale_num in range(2, scale_num + 1)]

        for idx_,current_scale_num in enumerate(run_list):
        # for current_scale_num in run_list:
            start_time = time.time()

            if self.lowvram:
                latents = latents.to(device)
                self.unet.to(device)
                torch.cuda.empty_cache()
            print("### Phase {} Denoising ###".format(current_scale_num))
            #
            current_height = x1_hight * current_scale_num
            current_width = x1_width * current_scale_num

            if height > width:
                current_width = int(current_width * aspect_ratio)
            else:
                current_height = int(current_height * aspect_ratio)

            # upsample encoding latent from previous stage
            latents = F.interpolate(latents.to(device), size=(
            int(current_height / self.vae_scale_factor), int(current_width / self.vae_scale_factor)), mode='bicubic')

            ################## for editing################## ################## ##################
            # multipatch encoding
            editing_latents_local = self.multi_patch_encoding(latents, pre_img, device, current_height, current_width,
                             scale_masked_imgs, scale_masks, current_scale_num, output_type, show_image,
                             latents.dtype, generator,sigma,
                             window_size=multi_patch_encoding_window_size, 
                             stride=multi_patch_encoding_stride,
                             file_name=file_name)

            latents = editing_latents_local.clone() ##
            
  
            #########################################################################################################################################
            if self.lowvram:
                latents = latents.cpu()
                torch.cuda.empty_cache()
            if not output_type == "latent":
                # make sure the VAE is in float32 mode, as it overflows in float16
                needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

                if self.lowvram:
                    needs_upcasting = False  # use madebyollin/sdxl-vae-fp16-fix in lowvram mode!
                    self.unet.cpu()
                    self.vae.to(device)

                if needs_upcasting:
                    self.upcast_vae()
                    latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

                print("### Phase {} Decoding ###".format(current_scale_num))
                if multi_decoder:
                    image = self.tiled_decode(latents, current_height, current_width)
                else:
                    image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            image = self.image_processor.postprocess(image, output_type=output_type)[0]
            output_images.append(image)    # cast back to fp16 if needed            
            end_time = time.time()
            print("___"*100)
            print(f'time for the {current_scale_num} stage running is : %s Seconds' % (end_time - start_time))

            # while time
            whole_time += (end_time - start_time)
            print(f'the whole time for the {current_scale_num} stage is : %s Seconds' % whole_time)

        print("___"*100)
        print('the whole time is : %s Seconds' % whole_time)

        # Offload all models
        self.maybe_free_model_hooks()

        if self.vis_analysis:
            return output_images, out_vis_analysisList

        return output_images


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def refine_editing(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            image: PipelineImageInput = None,
            content_img: PipelineImageInput = None,
            mask_image: PipelineImageInput = None,
            masked_image_latents: torch.FloatTensor = None,
            # height: Optional[int] = None,
            # width: Optional[int] = None,
            strength: float = 0.9999,
            num_inference_steps: int = 50,
            denoising_start: Optional[float] = None,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Tuple[int, int] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Tuple[int, int] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
            aesthetic_score: float = 6.0,
            negative_aesthetic_score: float = 2.5,
            ####
            # image_lr: Optional[torch.FloatTensor] = None,
            tar_height: Optional[int] = None,
            tar_width: Optional[int] = None,
            view_batch_size: int = 16,
            multi_decoder: bool = True,
            stride: Optional[int] = 64,
            multi_patch_encoding_window_size: int = 1024,
            multi_patch_encoding_stride: int = 512,
            beta_scale_1: Optional[float] = 3.,
            beta_scale_2: Optional[float] = 1.,
            cosine_scale_3: Optional[float] = 1.,
            sigma: Optional[float] = 1.0,
            show_image: bool = False,
            lowvram: bool = False,
            # for saving imges,
            file_name="out_img",
            save_root="results",
            save_image_tag=False,
            run_stage ="two",
            ug_weight = 0.2,
            orig_size = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            mask_image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted
                to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
                instead of 3, so the expected shape would be `(B, H, W, 1)`.
            strength (`float`, *optional*, defaults to 0.9999):
                Conceptually, indicates how much to transform the masked portion of the reference `image`. Must be
                between 0 and 1. `image` will be used as a starting point, adding more noise to it the larger the
                `strength`. The number of denoising steps depends on the amount of noise initially added. When
                `strength` is 1, added noise will be maximum and the denoising process will run for the full number of
                iterations specified in `num_inference_steps`. A value of 1, therefore, essentially ignores the masked
                portion of the reference `image`. Note that in the case of `denoising_start` being declared as an
                integer, the value of `strength` will be ignored.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            denoising_start (`float`, *optional*):
                When specified, indicates the fraction (between 0.0 and 1.0) of the total denoising process to be
                bypassed before it is initiated. Consequently, the initial part of the denoising process is skipped and
                it is assumed that the passed `image` is a partly denoised image. Note that when this is specified,
                strength will be ignored. The `denoising_start` parameter is particularly beneficial when this pipeline
                is integrated into a "Mixture of Denoisers" multi-pipeline setup, as detailed in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output).
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise (ca. final 20% of timesteps still needed) and should be
                denoised by a successor pipeline that has `denoising_start` set to 0.8 so that it only denoises the
                final 20% of the scheduler. The denoising_end parameter should ideally be utilized when this pipeline
                forms a part of a "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output).
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            aesthetic_score (`float`, *optional*, defaults to 6.0):
                Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
                Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_aesthetic_score (`float`, *optional*, defaults to 2.5):
                Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). Can be used to
                simulate an aesthetic score of the generated image by influencing the negative text condition.
            ################### UltraDiffEdit specific parameters ####################
            tar_height (`int`):
                The height in pixels of the edited image. This is set to Integer multiples of 1024.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            tar_width (`int`):
                The width in pixels of the generated image.This is set to Integer multiples of 1024.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            view_batch_size (`int`, defaults to 16):
                The batch size for multiple denoising paths. Typically, a larger batch size can result in higher 
                efficiency but comes with increased GPU memory requirements.
            multi_decoder (`bool`, defaults to True):
                Determine whether to use a tiled decoder. Generally, when the resolution exceeds 3072x3072, 
                a tiled decoder becomes necessary.
            stride (`int`, defaults to 64):
                The stride of moving local patches. A smaller stride is better for alleviating seam issues,
                but it also introduces additional computational overhead and inference time.
            beta_scale_1 (`float`, defaults to 3):
                Control the weights of diffused and denoised latent maps in global-local consistent denoising. 
            beta_scale_2 (`float`, defaults to 1):
                Control the weights of patch-based sampling, patch-based upsample guidance sampling, and dilated sampling in patch-based hybrid sampling. 
            cosine_scale_3 (`float`, defaults to 1):
                Control the strength of the gaussion filter. For specific impacts, please refer to Appendix C
                in the DemoFusion paper.
            sigma (`float`, defaults to 1):
                The standard value of the gaussian filter.
            show_image (`bool`, defaults to False):
                Determine whether to show intermediate results during generation.
            file_name (`str`, defaults to ``out_img``):
                The file prefix of the saved output image.
            save_root (`str`, defaults to ``results``):
                The root path of the saved output image.
            save_image_tag (`bool`, defaults to False):
                Determine whether to save intemediate results, such as masked images, and masks.
            run_stage (`str`, defaults to ``two``):
                define the phase set, using ``two``  to set the two stages, ``three``  to set the three stages, and ``S``  to set the S stages.
            ug_weight (`float`, defaults to 0.2):
                the weight used for the patch-based upsample guidnace sampling.
        Examples:

        Returns:
                a `list` with the generated images at each phase.

        """
        # 0. Default height and width to unet

        print(f"we use beta1",  beta_scale_1)
        print(f"we use beta2",  beta_scale_2)
        print(f"we use ug_weight", ug_weight)
        print("the size of the window is :", self.unet.config.sample_size)
        print("the stride of the window is :", stride)
        
        # multi_patch_encoding_window_size= self.unet.config.sample_size*self.vae_scale_factor
        # multi_patch_encoding_stride= stride*self.vae_scale_factor
        print("the multi_patch_encoding_window_size is :", multi_patch_encoding_window_size)
        print("the multi_patch_encoding_stride is :", multi_patch_encoding_stride)

        ####
        x1_size = self.default_sample_size * self.vae_scale_factor
        x1_width, x1_hight = self.get_start_size(tar_width, tar_height, fix_size=x1_size)
        #
        width = x1_width
        height = x1_hight
        #
        height_scale = tar_height / x1_hight
        width_scale = tar_width / x1_width
        scale_num = int(max(height_scale, width_scale))
        aspect_ratio = min(height_scale, width_scale) / max(height_scale, width_scale)

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        if orig_size!= None:
            self.orig_size  =orig_size
        else:
            self.orig_size  =[tar_height,tar_width]
        self.lowvram = lowvram
        ####
        self.save_root = save_root
        self.save_image_tag = save_image_tag
        ##
        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            strength,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. set timesteps
        def denoising_value_valid(dnv):
            return isinstance(denoising_end, float) and 0 < dnv < 1

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device, denoising_start=denoising_start if denoising_value_valid else None
        )
        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Preprocess mask and image
        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=torch.float32)

        # content_im = self.image_processor.preprocess(content_img, height=height, width=width)
        # content_im = content_im.to(dtype=torch.float32)
        #
        mask = self.mask_processor.preprocess(mask_image, height=height, width=width)

        if masked_image_latents is not None:
            masked_image = masked_image_latents
        elif init_image.shape[1] == 4:
            # if images are in latent space, we can't mask it
            masked_image = None
        else:
            masked_image = init_image * (mask < 0.5)  # mask>0.5 are mnipulated regions

        #
        _init_image = self.image_processor.preprocess(image, height=tar_height, width=tar_width)
        _init_image = _init_image.to(dtype=torch.float32).to(device)
        #
        content_im = self.image_processor.preprocess(content_img, height=tar_height, width=tar_width)
        content_im = content_im.to(dtype=torch.float32).to(device)
        #
        _mask = self.mask_processor.preprocess(mask_image, height=tar_height, width=tar_width).to(device)
        pre_img = _init_image * (_mask < 0.5) + content_im* (_mask >= 0.5)

        ###### each size tmp imgs, masks, and masked imgs
        scale_imgs = []
        scale_masks = []
        scale_masked_imgs = []
        for ii in range(scale_num):
            tmp_scale = (ii + 1)

            tmp_width = x1_width * tmp_scale
            tmp_height = x1_hight * tmp_scale

            if height > width:
                tmp_width = int(tmp_width * aspect_ratio)
            else:
                tmp_height = int(tmp_height * aspect_ratio)

            tmp_image = self.image_processor.preprocess(image, height=tmp_height, width=tmp_width)
            tmp_image = tmp_image.to(dtype=torch.float32)
            tmp_mask = self.mask_processor.preprocess(mask_image, height=tmp_height, width=tmp_width)
            tmp_masked_image = tmp_image * (tmp_mask < 0.5)
            scale_imgs.append(tmp_image.to(device))
            scale_masks.append(tmp_mask.to(device))
            scale_masked_imgs.append(tmp_masked_image.to(device))

        #######
        #
        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        add_noise = True if denoising_start is None else False
        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            add_noise=add_noise,
            return_noise=True,
            return_image_latents=return_image_latents,
        )  # unmasked regions

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents + num_channels_masked_image + num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
            )
        # 8.1 Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # sounds there are no needs for these codes
        # height, width = latents.shape[-2:]
        # height = height * self.vae_scale_factor
        # width = width * self.vae_scale_factor

        # original_size = original_size or (height, width)
        # target_size = target_size or (height, width)

        # 10. Prepare added time ids & embeddings
        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size

        add_text_embeds = pooled_prompt_embeds
        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)
        #
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
        #
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)
        #
        # 11. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        #

        image = self.image_processor.postprocess(masked_image, output_type=output_type)
        output_images = [image[0]]  # for saving images
        if self.save_image_tag:
            image[0].save(f"{self.save_root}/{file_name}_masked_img.png")
        #


        ####################################################### Phase Upscaling #####################################################

        run_list = [scale_num]
        if run_stage == "three":
            if  scale_num> 2: run_list = [int(scale_num/2)+1]+ run_list
        elif run_stage == "S":
            if  scale_num> 2: run_list = [current_scale_num for current_scale_num in range(2, scale_num + 1)]
        

        for idx_,current_scale_num in enumerate(run_list):
            if self.lowvram:
                latents = latents.to(device)
                self.unet.to(device)
                torch.cuda.empty_cache()
            print("### Phase {} Denoising ###".format(current_scale_num))
            #
            current_height = x1_hight * current_scale_num
            current_width = x1_width * current_scale_num

            if height > width:
                current_width = int(current_width * aspect_ratio)
            else:
                current_height = int(current_height * aspect_ratio)

            latents = F.interpolate(latents.to(device), size=(
                int(current_height / self.vae_scale_factor), int(current_width / self.vae_scale_factor)),
                                    mode='bicubic')

            ################## for editing################## ################## ##################
            # upsample encoding latent from previous stage
            editing_latents_local = self.multi_patch_encoding(latents, pre_img, device, current_height, current_width,
                                                              scale_masked_imgs, scale_masks, current_scale_num,
                                                              output_type, show_image,
                                                              latents.dtype, generator, sigma,
                                                              window_size=multi_patch_encoding_window_size, 
                                                              stride=multi_patch_encoding_stride,
                                                              file_name=file_name)

            latents = editing_latents_local.clone()  ## 生成的 encoding 应该是不需要的

            cur_mask = torch.nn.functional.interpolate(
                mask[:1], size=(int(current_height / self.vae_scale_factor), int(current_width / self.vae_scale_factor))
            )
            cur_mask = cur_mask.to(device=device, dtype=latents.dtype)  # 0-1 mask
            confidence_mask_layer = ConfidenceDrivenMaskLayer(size=(2 * current_scale_num - 1), sigma=sigma, iters=7,
                                                              pad=current_scale_num - 1)
            soft_cur_mask = confidence_mask_layer(1 - cur_mask)
            cur_mask = cur_mask + soft_cur_mask

            # cur_mask = (1.0 - soft_cur_mask) * cur_mask

            # ####### debug the cur_img and the soft_cur_mask
            # image = self.image_processor.postprocess(soft_cur_mask, output_type=output_type)
            # # if show_image:
            # #     plt.figure(figsize=(10, 10))
            # #     plt.imshow(image[0])
            # #     plt.axis('off')  # Turn off axis numbers and ticks
            # #     plt.show()
            # output_images.append(image[0])
            # if self.save_image_tag:
            #     image[0].save(f"{self.save_root}/{file_name}_{current_scale_num}_soft_cur_mask.png")

            # #
            # image = self.image_processor.postprocess(cur_mask, output_type=output_type)
            # # if show_image:
            # #     plt.figure(figsize=(10, 10))
            # #     plt.imshow(image[0])
            # #     plt.axis('off')  # Turn off axis numbers and ticks
            # #     plt.show()
            # output_images.append(image[0])
            # if self.save_image_tag:
            #     image[0].save(f"{self.save_root}/{file_name}_{current_scale_num}_cur_mask.png")

            #########

            ##################### ################## ################## ##################

            ##
            noise_latents = []
            noise = torch.randn_like(latents)
      
            # save the noise
            for ii, timestep in enumerate(timesteps):
                noise_latent = self.scheduler.add_noise(latents, noise, timestep.unsqueeze(0))
                noise_latents.append(noise_latent)

            latents = noise_latents[0]

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    count = torch.zeros_like(latents)
                    value = torch.zeros_like(latents)
                    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * (
                            self.scheduler.config.num_train_timesteps - t) / self.scheduler.config.num_train_timesteps)).cpu()

                    c1 = cosine_factor ** beta_scale_1
                    latents = latents * (1 - c1) + noise_latents[i] * c1
                    #
                    cur_edit_latent = noise_latents[i]
                    latents = latents * cur_mask + cur_edit_latent * (1 - cur_mask)
                    #
                    ### debug
                    # image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                    # image = self.image_processor.postprocess(image, output_type=output_type)
                    # output_images.append(image[0])
                    # if self.save_image_tag:
                    #     image[0].save(f"{self.save_root}/{file_name}_{current_scale_num}_editing_latent_{t}.png")
                    # #

                    ############################################# MultiDiffusion #############################################
                    # first get the coordinates of each patch
                    views = self.get_views(current_height, current_width, stride=stride,
                                           window_size=self.unet.config.sample_size, random_jitter=True)

                    # then group batches based on view_barch_size
                    views_batch = [views[i: i + view_batch_size] for i in range(0, len(views), view_batch_size)]

                    # why add jitter for ranges?
                    jitter_range = (self.unet.config.sample_size - stride) // 4
                    latents_ = F.pad(latents, (jitter_range, jitter_range, jitter_range, jitter_range), 'constant', 0)

                    count_local = torch.zeros_like(latents_)
                    value_local = torch.zeros_like(latents_)

                    # for all batches, each time we only for (view_barch_size) batch latents
                    for j, batch_view in enumerate(views_batch):
                        vb_size = len(batch_view)

                        # get the latents corresponding to the current view coordinates
                        latents_for_view = torch.cat(
                            [
                                latents_[:, :, h_start:h_end, w_start:w_end]
                                for h_start, h_end, w_start, w_end in batch_view
                            ]
                        )

                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = latents_for_view
                        latent_model_input = (
                            latent_model_input.repeat_interleave(2, dim=0)
                            if do_classifier_free_guidance
                            else latent_model_input
                        )
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        prompt_embeds_input = torch.cat([prompt_embeds] * vb_size)
                        add_text_embeds_input = torch.cat([add_text_embeds] * vb_size)
                        add_time_ids_input = []
                        for h_start, h_end, w_start, w_end in batch_view:
                            add_time_ids_ = add_time_ids.clone()
                            add_time_ids_[:, 2] = h_start * self.vae_scale_factor
                            add_time_ids_[:, 3] = w_start * self.vae_scale_factor
                            add_time_ids_input.append(add_time_ids_)
                        add_time_ids_input = torch.cat(add_time_ids_input)

                        # predict the noise residual
                        added_cond_kwargs = {"text_embeds": add_text_embeds_input, "time_ids": add_time_ids_input}
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds_input,
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred[::2], noise_pred[1::2]
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        if do_classifier_free_guidance and guidance_rescale > 0.0:
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text,
                                                           guidance_rescale=guidance_rescale)

                        # compute the previous noisy sample x_t -> x_t-1
                        self.scheduler._init_step_index(t)
                        latents_denoised_batch = self.scheduler.step(
                            noise_pred, t, latents_for_view, **extra_step_kwargs, return_dict=False)[0]

                        # extract value from batch
                        for latents_view_denoised, (h_start, h_end, w_start, w_end) in zip(
                                latents_denoised_batch.chunk(vb_size), batch_view
                        ):
                            value_local[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                            count_local[:, :, h_start:h_end, w_start:w_end] += 1

                    # after all views, you may get all the local patches, so the jetter is trying to reduce the boundary issues
                    value_local = value_local[:, :,
                                  jitter_range: jitter_range + current_height // self.vae_scale_factor,
                                  jitter_range: jitter_range + current_width // self.vae_scale_factor]
                    count_local = count_local[:, :,
                                  jitter_range: jitter_range + current_height // self.vae_scale_factor,
                                  jitter_range: jitter_range + current_width // self.vae_scale_factor]

                    c2 = cosine_factor ** beta_scale_2 * 0.5

                    value += value_local / count_local * (1 - c2)
                    count += torch.ones_like(value_local) * (1 - c2)

                    ############################################# LOCAL UG #############################################
                    # # P = αt + 1/m*m *(1−αt)
                    ug_scale = 2
                    Alpha_t = self.scheduler.alphas[t.cpu().int()]
                    P_ = Alpha_t + (1 - Alpha_t) / (ug_scale * ug_scale)
                    P_sqrt = P_ ** 0.5
                    #### for memory saving
                    ug_view_batch_size = view_batch_size // 4
                    # first get the coordinates of each patch
                    views = self.get_views(current_height, current_width, stride=stride,
                                           window_size=self.unet.config.sample_size * ug_scale, random_jitter=True)

                    # views = self.views_filter(views, cur_mask)

                    # then group batches based on view_barch_size
                    views_batch = [views[i: i + ug_view_batch_size] for i in range(0, len(views), ug_view_batch_size)]

                    # why add jitter for ranges?
                    jitter_range = (self.unet.config.sample_size * ug_scale - stride) // 4
                    latents_ = F.pad(latents, (jitter_range, jitter_range, jitter_range, jitter_range), 'constant', 0)

                    ug_count_local = torch.zeros_like(latents_)
                    ug_value_local = torch.zeros_like(latents_)

                    # for all batches, each time we only for (view_barch_size) batch latents
                    for j, batch_view in enumerate(views_batch):
                        vb_size = len(batch_view)

                        # get the latents corresponding to the current view coordinates
                        latents_for_view = torch.cat(
                            [
                                latents_[:, :, h_start:h_end, w_start:w_end]
                                for h_start, h_end, w_start, w_end in batch_view
                            ]
                        )

                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = latents_for_view
                        latent_model_input = (
                            latent_model_input.repeat_interleave(2, dim=0)
                            if do_classifier_free_guidance
                            else latent_model_input
                        )
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        prompt_embeds_input = torch.cat([prompt_embeds] * vb_size)
                        add_text_embeds_input = torch.cat([add_text_embeds] * vb_size)
                        add_time_ids_input = []
                        for h_start, h_end, w_start, w_end in batch_view:
                            add_time_ids_ = add_time_ids.clone()
                            add_time_ids_[:, 2] = h_start * self.vae_scale_factor
                            add_time_ids_[:, 3] = w_start * self.vae_scale_factor
                            add_time_ids_input.append(add_time_ids_)
                        add_time_ids_input = torch.cat(add_time_ids_input)

                        # predict the noise residual
                        added_cond_kwargs = {"text_embeds": add_text_embeds_input, "time_ids": add_time_ids_input}
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds_input,
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred[::2], noise_pred[1::2]
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        if do_classifier_free_guidance and guidance_rescale > 0.0:
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text,
                                                           guidance_rescale=guidance_rescale)

                        # # compute the previous noisy sample x_t -> x_t-1
                        # self.scheduler._init_step_index(t)
                        # latents_denoised_batch = self.scheduler.step(
                        #     noise_pred, t, latents_for_view, **extra_step_kwargs, return_dict=False)[0]

                        #######UG ###########
                        # get the latents corresponding to the current view coordinates
                        down_latents_for_view = torch.cat(
                            [
                                torch.nn.functional.avg_pool2d(latents_[:, :, h_start:h_end, w_start:w_end],
                                                               kernel_size=ug_scale, stride=ug_scale)
                                for h_start, h_end, w_start, w_end in batch_view
                            ]
                        )
                        ###########
                        # expand the latents if we are doing classifier free guidance
                        down_latent_model_input = down_latents_for_view
                        down_latent_model_input = (
                            down_latent_model_input.repeat_interleave(2, dim=0)
                            if do_classifier_free_guidance
                            else down_latent_model_input
                        )
                        down_latent_model_input = self.scheduler.scale_model_input(down_latent_model_input, t)
                        down_latent_model_input = down_latent_model_input / P_sqrt

                        # prompt_embeds_input = torch.cat([prompt_embeds] * vb_size)
                        # add_text_embeds_input = torch.cat([add_text_embeds] * vb_size)
                        # add_time_ids_input = []
                        # for h_start, h_end, w_start, w_end in batch_view:
                        #     add_time_ids_ = add_time_ids.clone()
                        #     add_time_ids_[:, 2] = h_start * self.vae_scale_factor
                        #     add_time_ids_[:, 3] = w_start * self.vae_scale_factor
                        #     add_time_ids_input.append(add_time_ids_)
                        # add_time_ids_input = torch.cat(add_time_ids_input)

                        # predict the noise residual
                        # added_cond_kwargs = {"text_embeds": add_text_embeds_input, "time_ids": add_time_ids_input}
                        down_noise_pred = self.unet(
                            down_latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds_input,
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]
                        down_noise_pred = down_noise_pred / ug_scale

                        if do_classifier_free_guidance:
                            down_noise_pred_uncond, down_noise_pred_text = down_noise_pred[::2], down_noise_pred[1::2]
                            down_noise_pred = down_noise_pred_uncond + guidance_scale * (
                                        down_noise_pred_text - down_noise_pred_uncond)

                        if do_classifier_free_guidance and guidance_rescale > 0.0:
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            down_noise_pred = rescale_noise_cfg(down_noise_pred, down_noise_pred_text,
                                                                guidance_rescale=guidance_rescale)

                        if t > 500:
                            down_value_noise_pred = torch.nn.functional.avg_pool2d(noise_pred, kernel_size=ug_scale,
                                                                                   stride=ug_scale)
                            up_guidance = torch.nn.functional.interpolate(
                                (down_noise_pred - down_value_noise_pred), size=(
                                    int(noise_pred.shape[2]), int(noise_pred.shape[3])), mode="nearest"
                            )
                            # noise_pred = noise_pred + 0.2 * up_guidance
                            noise_pred = noise_pred + ug_weight * up_guidance

                        # compute the previous noisy sample x_t -> x_t-1
                        self.scheduler._init_step_index(t)
                        latents_denoised_batch = self.scheduler.step(
                            noise_pred, t, latents_for_view, **extra_step_kwargs, return_dict=False)[0]
                        ############
                        # extract value from batch
                        for latents_view_denoised, (h_start, h_end, w_start, w_end) in zip(
                                latents_denoised_batch.chunk(vb_size), batch_view
                        ):
                            ug_value_local[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                            ug_count_local[:, :, h_start:h_end, w_start:w_end] += 1

                    # after all views, you may get all the local patches, so the jetter is trying to reduce the boundary issues
                    ug_value_local = ug_value_local[:, :,
                                     jitter_range: jitter_range + current_height // self.vae_scale_factor,
                                     jitter_range: jitter_range + current_width // self.vae_scale_factor]
                    ug_count_local = ug_count_local[:, :,
                                     jitter_range: jitter_range + current_height // self.vae_scale_factor,
                                     jitter_range: jitter_range + current_width // self.vae_scale_factor]

                    c2 = cosine_factor ** beta_scale_2 * 0.5

                    value += ug_value_local / ug_count_local * (1 - c2)
                    count += torch.ones_like(ug_value_local) * (1 - c2)

                    ############################################# Dilated Sampling #############################################

                    views = [[h, w] for h in range(current_scale_num) for w in range(current_scale_num)]
                    views_batch = [views[i: i + view_batch_size] for i in range(0, len(views), view_batch_size)]

                    h_pad = (current_scale_num - (latents.size(2) % current_scale_num)) % current_scale_num
                    w_pad = (current_scale_num - (latents.size(3) % current_scale_num)) % current_scale_num
                    latents_ = F.pad(latents, (w_pad, 0, h_pad, 0), 'constant', 0)

                    count_global = torch.zeros_like(latents_)
                    value_global = torch.zeros_like(latents_)

                    c3 = 0.99 * cosine_factor ** cosine_scale_3 + 1e-2
                    std_, mean_ = latents_.std(), latents_.mean()
                    latents_gaussian = gaussian_filter(latents_, kernel_size=(2 * current_scale_num - 1),
                                                       sigma=sigma * c3)
                    ##
                    latents_gaussian = (latents_gaussian - latents_gaussian.mean()) / latents_gaussian.std() * std_ + mean_

                    for j, batch_view in enumerate(views_batch):
                        latents_for_view = torch.cat(
                            [
                                latents_[:, :, h::current_scale_num, w::current_scale_num]
                                for h, w in batch_view
                            ]
                        )
                        latents_for_view_gaussian = torch.cat(
                            [
                                latents_gaussian[:, :, h::current_scale_num, w::current_scale_num]
                                for h, w in batch_view
                            ]
                        )

                        vb_size = latents_for_view.size(0)

                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = latents_for_view_gaussian
                        latent_model_input = (
                            latent_model_input.repeat_interleave(2, dim=0)
                            if do_classifier_free_guidance
                            else latent_model_input
                        )
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        prompt_embeds_input = torch.cat([prompt_embeds] * vb_size)
                        add_text_embeds_input = torch.cat([add_text_embeds] * vb_size)
                        add_time_ids_input = torch.cat([add_time_ids] * vb_size)

                        # predict the noise residual
                        added_cond_kwargs = {"text_embeds": add_text_embeds_input, "time_ids": add_time_ids_input}
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds_input,
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred[::2], noise_pred[1::2]
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        if do_classifier_free_guidance and guidance_rescale > 0.0:
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text,
                                                           guidance_rescale=guidance_rescale)

                        # compute the previous noisy sample x_t -> x_t-1
                        self.scheduler._init_step_index(t)
                        latents_denoised_batch = self.scheduler.step(
                            noise_pred, t, latents_for_view, **extra_step_kwargs, return_dict=False)[0]

                        # extract value from batch
                        for latents_view_denoised, (h, w) in zip(
                                latents_denoised_batch.chunk(vb_size), batch_view
                        ):
                            value_global[:, :, h::current_scale_num, w::current_scale_num] += latents_view_denoised
                            count_global[:, :, h::current_scale_num, w::current_scale_num] += 1

                    c2 = cosine_factor ** beta_scale_2

                    value_global = value_global[:, :, h_pad:, w_pad:]

                    value += value_global * c2
                    count += torch.ones_like(value_global) * c2

                    ###########################################################
                    latents = torch.where(count > 0, value / count, value)
                    ###################################

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)


                #########################################################################################################################################
 
                if self.lowvram:
                    latents = latents.cpu()
                    torch.cuda.empty_cache()
                if not output_type == "latent":
                    # make sure the VAE is in float32 mode, as it overflows in float16
                    needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

                    if self.lowvram:
                        needs_upcasting = False  # use madebyollin/sdxl-vae-fp16-fix in lowvram mode!
                        self.unet.cpu()
                        self.vae.to(device)

                    if needs_upcasting:
                        self.upcast_vae()
                        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

                    print("### Phase {} Decoding ###".format(current_scale_num))
                    if multi_decoder:
                        image = self.tiled_decode(latents, current_height, current_width)
                    else:
                        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

                    ## hard replacing
                    cur_masked_img = scale_masked_imgs[current_scale_num - 1]
                    cur_mask = scale_masks[current_scale_num - 1]
                    cur_mask = cur_mask.to(device=device, dtype=latents.dtype)  # 0-1 mask

                    confidence_mask_layer = ConfidenceDrivenMaskLayer(
                        size=(2 * current_scale_num * self.vae_scale_factor - 1), sigma=sigma, iters=7,
                        pad=(current_scale_num) * self.vae_scale_factor - 1)
                    soft_cur_mask = confidence_mask_layer(1 - cur_mask)
                    cur_mask = cur_mask + soft_cur_mask

                    ##
                    inpainted_image = cur_masked_img * (1 - cur_mask) + image * cur_mask

                    if current_scale_num == run_list[-1]:
                        print("process_images_with_poisson!!")
                        #
                        inpainted_image = inpainted_image[:,:,:self.orig_size[0],:self.orig_size[1]]
                        image = image[:,:,:self.orig_size[0],:self.orig_size[1]]
                        cur_mask = cur_mask[:,:,:self.orig_size[0],:self.orig_size[1]]

                        inpainted_image = blend(inpainted_image.detach().clone()[0].cpu(), image.detach().clone()[0].cpu(),
                                                cur_mask.detach().clone()[0, 0].cpu(),
                                                torch.tensor([0, 0]).cpu(), True, channels_dim=0, restore_detail=True,
                                                data_range=[-1, 1]).unsqueeze(0)
                        ####                        
                        # inpainted_image = image
                        # print("process_images_with_poisson!!")
                        # pre_scale_num = run_list[idx_-1]
                        # pre_height = x1_hight * pre_scale_num
                        # pre_width = x1_width * pre_scale_num
                        # small_mask = scale_masks[pre_scale_num - 1]
                        #
                        # small_inpainted_image = F.interpolate(inpainted_image.detach().clone().to(device), size=(pre_height,pre_width),mode='bicubic')
                        #
                        # small_image = F.interpolate(image.detach().clone().to(device), size=(pre_height,pre_width),mode='bicubic')
                        #
                        # small_inpainted_image = blend(small_inpainted_image.detach().clone()[0].cpu(), small_image.detach().clone()[0].cpu(),
                        #                         small_mask.detach().clone()[0, 0].cpu(),
                        #                         torch.tensor([0, 0]).cpu(), True, channels_dim=0, restore_detail=True,
                        #                         data_range=[-1, 1]).unsqueeze(0)
                        #
                        #
                        # small_inpainted_image = F.interpolate(small_inpainted_image, size=(current_height,current_width),mode='bicubic')
                        #
                        # inpainted_image = cur_masked_img * (1 - cur_mask) + (image*0.5+small_inpainted_image*0.5) * cur_mask




                    # cast back to fp16 if needed
                    if needs_upcasting:
                        self.vae.to(dtype=torch.float16)
                else:
                    image = latents

                # for subsequent processing
                pre_img_ = inpainted_image.clone()
                if not output_type == "latent":
                    image = self.image_processor.postprocess(image, output_type=output_type)
                    if show_image:
                        plt.figure(figsize=(10, 10))
                        plt.imshow(image[0])
                        plt.axis('off')  # Turn off axis numbers and ticks
                        plt.show()
                    output_images.append(image[0])
                    if self.save_image_tag:
                        image[0].save(f"{self.save_root}/{file_name}_{current_scale_num}_out.png")

                    inpainted_image = self.image_processor.postprocess(inpainted_image, output_type=output_type)
                    if show_image:
                        plt.figure(figsize=(10, 10))
                        plt.imshow(inpainted_image[0])
                        plt.axis('off')  # Turn off axis numbers and ticks
                        plt.show()
                    output_images.append(inpainted_image[0])
                    if self.save_image_tag:
                        image[0].save(f"{self.save_root}/{file_name}_{current_scale_num}_edit.png")
                    ##

                    output_images.append(inpainted_image)

            pre_img = pre_img_
            # preserve for the next stage
            # pre_editing_latents_global = editing_latents_local.clone().detach()  # for next step
            latents = editing_latents_local.clone().detach()  # for next step

        # Offload all models
        self.maybe_free_model_hooks()


        return output_images

    # Overrride to properly handle the loading and unloading of the additional text encoder.
    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.load_lora_weights
    def load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
        # We could have accessed the unet config from `lora_state_dict()` too. We pass
        # it here explicitly to be able to tell that it's coming from an SDXL
        # pipeline.

        # Remove any existing hooks.
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module
        else:
            raise ImportError("Offloading requires `accelerate v0.17.0` or higher.")

        is_model_cpu_offload = False
        is_sequential_cpu_offload = False
        recursive = False
        for _, component in self.components.items():
            if isinstance(component, torch.nn.Module):
                if hasattr(component, "_hf_hook"):
                    is_model_cpu_offload = isinstance(getattr(component, "_hf_hook"), CpuOffload)
                    is_sequential_cpu_offload = isinstance(getattr(component, "_hf_hook"), AlignDevicesHook)
                    logger.info(
                        "Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again."
                    )
                    recursive = is_sequential_cpu_offload
                    remove_hook_from_module(component, recurse=recursive)
        state_dict, network_alphas = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            unet_config=self.unet.config,
            **kwargs,
        )
        self.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=self.unet)

        text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
        if len(text_encoder_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder,
                prefix="text_encoder",
                lora_scale=self.lora_scale,
            )

        text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
        if len(text_encoder_2_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_2_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder_2,
                prefix="text_encoder_2",
                lora_scale=self.lora_scale,
            )

        # Offload back.
        if is_model_cpu_offload:
            self.enable_model_cpu_offload()
        elif is_sequential_cpu_offload:
            self.enable_sequential_cpu_offload()

    @classmethod
    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.save_lora_weights
    def save_lora_weights(
        self,
        save_directory: Union[str, os.PathLike],
        unet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        text_encoder_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        text_encoder_2_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
    ):
        state_dict = {}

        def pack_weights(layers, prefix):
            layers_weights = layers.state_dict() if isinstance(layers, torch.nn.Module) else layers
            layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in layers_weights.items()}
            return layers_state_dict

        if not (unet_lora_layers or text_encoder_lora_layers or text_encoder_2_lora_layers):
            raise ValueError(
                "You must pass at least one of `unet_lora_layers`, `text_encoder_lora_layers` or `text_encoder_2_lora_layers`."
            )

        if unet_lora_layers:
            state_dict.update(pack_weights(unet_lora_layers, "unet"))

        if text_encoder_lora_layers and text_encoder_2_lora_layers:
            state_dict.update(pack_weights(text_encoder_lora_layers, "text_encoder"))
            state_dict.update(pack_weights(text_encoder_2_lora_layers, "text_encoder_2"))

        self.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
        )

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._remove_text_encoder_monkey_patch
    def _remove_text_encoder_monkey_patch(self):
        self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder)
        self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder_2)
