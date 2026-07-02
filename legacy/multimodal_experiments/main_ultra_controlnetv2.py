# import gradio as gr
from diffusers import ControlNetModel, AutoencoderKL

# from gradio_imageslider import ImageSlider
import torch, gc
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import Util.utils_train  as ut
from Util.img_pad_crop import pad_image_to_multiple_num,crop_image_to_original,pad_image_to_multiple_factors
from Util.caption_read import read_lines_from_file
import math

def load_and_process_image(pil_image):
    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = transform(pil_image)
    image = image.unsqueeze(0).half()
    return image


def pad_image(image):
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = Image.new(image.mode, (w, w), (0, 0, 0))
        pad_w = 0
        pad_h = (w - h) // 2
        new_image.paste(image, (0, pad_h))
        return new_image
    else:
        new_image = Image.new(image.mode, (h, h), (0, 0, 0))
        pad_w = (h - w) // 2
        pad_h = 0
        new_image.paste(image, (pad_w, 0))
        return new_image

# def generate_images(prompt, negative_prompt, controlnet_conditioning_scale, height, width, num_inference_steps, guidance_scale, cosine_scale_1, cosine_scale_2, cosine_scale_3, sigma, view_batch_size, stride, seed, input_image):
#
#     padded_image = pad_image(input_image).resize((1024, 1024)).convert("RGB")
#     image_lr = load_and_process_image(padded_image).to('cuda')
#     controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
#     vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
#     pipe = DemoFusionSDXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
#     pipe = pipe.to("cuda")
#     generator = torch.Generator(device='cuda')
#     generator = generator.manual_seed(int(seed))
#     # get canny image
#     canny_image = np.array(padded_image)
#     canny_image = cv2.Canny(canny_image, 100, 200)
#     canny_image = canny_image[:, :, None]
#     canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
#     canny_image = Image.fromarray(canny_image)
#     images = pipe(prompt, negative_prompt=negative_prompt, controlnet_conditioning_scale=controlnet_conditioning_scale,
#                   image_lr=image_lr, condition_image=canny_image, generator=generator,
#                   height=int(height), width=int(width), view_batch_size=int(view_batch_size), stride=int(stride),
#                   num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
#                   cosine_scale_1=cosine_scale_1, cosine_scale_2=cosine_scale_2, cosine_scale_3=cosine_scale_3, sigma=sigma,
#                   multi_decoder=True, show_image=False, lowvram=False
#                  )
#     for i, image in enumerate(images):
#       image.save('image_'+str(i)+'.png')
#     pipe = None
#     gc.collect()
#     torch.cuda.empty_cache()
#     return (images[0], images[-1])
#
# with gr.Blocks(title=f"DemoFusion") as demo:
#     with gr.Column():
#       with gr.Row():
#         with gr.Group():
#           image_input = gr.Image(type="pil", label="Input Image")
#           prompt = gr.Textbox(label="Prompt (Note: an accurate prompt to describe the content and style of the input will significantly improve performance.)", value="8k high definition, high details")
#           negative_prompt = gr.Textbox(label="Negative Prompt", value="blurry, ugly, duplicate, poorly drawn, deformed, mosaic")
#           controlnet_conditioning_scale = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, label="ControlNet Conditioning Scale")
#           width = gr.Slider(minimum=1024, maximum=4096, step=1024, value=2048, label="Width")
#           height = gr.Slider(minimum=1024, maximum=4096, step=1024, value=2048, label="Height")
#           num_inference_steps = gr.Slider(minimum=10, maximum=100, step=1, value=50, label="Num Inference Steps")
#           guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.1, value=7.5, label="Guidance Scale")
#           cosine_scale_1 = gr.Slider(minimum=0, maximum=5, step=0.1, value=3, label="Cosine Scale 1")
#           cosine_scale_2 = gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Cosine Scale 2")
#           cosine_scale_3 = gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Cosine Scale 3")
#           sigma = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.8, label="Sigma")
#           view_batch_size = gr.Slider(minimum=4, maximum=32, step=4, value=16, label="View Batch Size")
#           stride = gr.Slider(minimum=8, maximum=96, step=8, value=64, label="Stride")
#           seed = gr.Number(label="Seed", value=2013)
#           button = gr.Button()
#         output_images = ImageSlider(show_label=False)
#     button.click(fn=generate_images, inputs=[prompt, negative_prompt, controlnet_conditioning_scale, height, width, num_inference_steps, guidance_scale, cosine_scale_1, cosine_scale_2, cosine_scale_3, sigma, view_batch_size, stride, seed, image_input], outputs=[output_images], show_progress=True)
# demo.queue().launch(inline=False, share=True, debug=True)



def v1():
    import time
    import os
    from diffusers.utils import load_image
    from diffusers import ControlNetModel, DDIMScheduler

    # from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline
    from pipeline_demofusion_sdxl_controlnet import DemoFusionSDXLControlNetPipeline
    # from pipeline_anysize_inpaint_controlnet import DemoFusionSDXLControlNetPipeline
    # from pipeline_anysize_inpaint_controlnetv2 import DemoFusionSDXLControlNetPipeline
    # from pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline

    name_ = str(time.time())

    os.makedirs("./results", exist_ok=True)

    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    # mask_path = "/home/onelong/Longlongaaago/dataset/DemoFusion_data/demofusion_img_demo/A freshly baked loaf of bread on a rustic kitchen counter./img_1024_mask.png"
    # img_path = "/home/onelong/Longlongaaago/dataset/DemoFusion_data/demofusion_img_demo/A freshly baked loaf of bread on a rustic kitchen counter./img_4096.jpg"

    init_image = load_image(img_url).convert("RGB").resize((1024, 1024))
    mask_image = load_image(mask_url).convert("RGB").resize((1024, 1024))
    init_image.save(f"results/{name_}_input.png")
    mask_image.save(f"results/{name_}_mask.png")

    padded_image = pad_image(init_image).resize((1024, 1024)).convert("RGB")

    image_lr = load_and_process_image(padded_image).to('cuda')
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = DemoFusionSDXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                            controlnet=controlnet, vae=vae, torch_dtype=torch.float16)

    pipe = pipe.to("cuda")
    generator = torch.Generator(device='cuda')

    prompt = "8k high definition, high details."
    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic."
    controlnet_conditioning_scale = 0.5  # [0,1]  ControlNet Conditioning Scale

    width = 4096  # gr.Slider(minimum=1024, maximum=4096, step=1024, value=2048, label="Width")
    height = 4096  # gr.Slider(minimum=1024, maximum=4096, step=1024, value=2048, label="Height")
    num_inference_steps = 50  # = gr.Slider(minimum=10, maximum=100, step=1, value=50, label="Num Inference Steps")
    guidance_scale = 7.5  # gr.Slider(minimum=1, maximum=20, step=0.1, value=7.5, label="Guidance Scale")
    cosine_scale_1 = 3  # gr.Slider(minimum=0, maximum=5, step=0.1, value=3, label="Cosine Scale 1")
    cosine_scale_2 = 1  # gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Cosine Scale 2")
    cosine_scale_3 = 1  # gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Cosine Scale 3")
    sigma = 0.8  # gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.8, label="Sigma")
    view_batch_size = 16  # gr.Slider(minimum=4, maximum=32, step=4, value=16, label="View Batch Size")
    stride = 64  # gr.Slider(minimum=8, maximum=96, step=8, value=64, label="Stride")
    seed = 2013  # gr.Number(label="Seed", value=2013)
    generator = generator.manual_seed(int(seed))

    # name_
    # get canny image
    canny_image = np.array(padded_image)
    canny_image = cv2.Canny(canny_image, 100, 200)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)

    images = pipe(prompt, negative_prompt=negative_prompt, controlnet_conditioning_scale=controlnet_conditioning_scale,
                  image_lr=image_lr,
                  condition_image=canny_image, generator=generator,

                  # image=init_image, mask_image=mask_image,
                  # pag_scale = 5,
                  ##
                  height = height, width = width,
                  # tar_height=height,
                  # tar_width=width,

                  # height=int(height), width=int(width),
                  view_batch_size=int(view_batch_size), stride=int(stride),
                  num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                  cosine_scale_1=cosine_scale_1, cosine_scale_2=cosine_scale_2, cosine_scale_3=cosine_scale_3,
                  sigma=sigma,
                  multi_decoder=True, show_image=True, lowvram=False
                  )
    # images = pipe(prompt, negative_prompt=negative_prompt, controlnet_conditioning_scale=controlnet_conditioning_scale,
    #               image_lr=image_lr,
    #               condition_image=canny_image, generator=generator,
    #               mask_image=mask_image,
    #               # image=init_image, mask_image=mask_image,
    #               # pag_scale = 5,
    #               ##
    #               height=height, width=width,
    #               # tar_height=height,
    #               # tar_width=width,
    #
    #               # height=int(height), width=int(width),
    #               view_batch_size=int(view_batch_size), stride=int(stride),
    #               num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
    #               cosine_scale_1=cosine_scale_1, cosine_scale_2=cosine_scale_2, cosine_scale_3=cosine_scale_3,
    #               sigma=sigma,
    #               multi_decoder=True, show_image=True, lowvram=False
    #               )
    # images = pipe(prompt, negative_prompt=negative_prompt, controlnet_conditioning_scale=controlnet_conditioning_scale,
    #               # image_lr=image_lr,
    #               condition_image=canny_image, generator=generator,
    #
    #               image=init_image, mask_image=mask_image,
    #               # pag_scale = 5,
    #               ##
    #               # height = height, width = width,
    #               tar_height=height,
    #               tar_width=width,
    #
    #               # height=int(height), width=int(width),
    #               view_batch_size=int(view_batch_size), stride=int(stride),
    #               num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
    #               cosine_scale_1=cosine_scale_1, cosine_scale_2=cosine_scale_2, cosine_scale_3=cosine_scale_3,
    #               sigma=sigma,
    #               multi_decoder=True, show_image=True, lowvram=False
    #               )

    for i, image in enumerate(images):
        image.save(f'results/image_{name_}_' + str(i) + '.png')
    pipe = None
    gc.collect()
    torch.cuda.empty_cache()

    # return (images[0], images[-1])


def get_depth_map(image,feature_extractor,depth_estimator):
        image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image



def ultra_depth():
    from pipeline_anysize_controlnet_inpaint_sd_xlv3 import StableDiffusionXLControlNetInpaintPipeline

    from transformers import DPTFeatureExtractor, DPTForDepthEstimation
    from diffusers import ControlNetModel,  AutoencoderKL
    from diffusers.utils import load_image

    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()


    prompt = "stormtrooper lecture, photorealistic"
    init_image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
    controlnet_conditioning_scale = 0.5  # recommended for good generalization

    depth_image = get_depth_map(init_image,feature_extractor,depth_estimator)
    width = 2048  # gr.Slider(minimum=1024, maximum=4096, step=1024, value=2048, label="Width")
    height = 2048  # gr.Slider(minimum=1024, maximum=4096, step=1024, value=2048, label="Height")

    mask_image = load_image(
        "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png")
    mask_image = mask_image.resize((2048, 2048))


    image = pipe(
        prompt,
        # image=depth_image,
        num_inference_steps=30,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        image=init_image,
        mask_image=mask_image,
        control_image=depth_image,

        tar_height=height,
        tar_width=width,
        save_image_tag=True,
        show_image=True
    )

    os.makedirs("control_depth_out", exist_ok=True)

    for ii, img in enumerate(image):
        img.save(f"control_depth_out/{ii}_finalout.png")


def make_canny_condition(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]

    image = np.concatenate([image, image, image], axis=2)

    image = Image.fromarray(image)

    return image

def ultra_controlnet_canny():
    from diffusers import ControlNetModel, DDIMScheduler
    # from pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
    # from pipeline_anysize_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
    from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, EulerAncestralDiscreteScheduler, \
        UniPCMultistepScheduler
    from diffusers import EulerDiscreteScheduler
    # from pipeline_anysize_controlnet_inpaint_sd_xlv2 import StableDiffusionXLControlNetInpaintPipeline
    from pipeline_anysize_controlnet_inpaint_sd_xlv3 import StableDiffusionXLControlNetInpaintPipeline
    # from pipeline_anysize_controlnet_inpaint_sd_xlv4 import StableDiffusionXLControlNetInpaintPipeline

    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np
    import torch
    import cv2

    init_image = load_image(
        "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png")
    init_image = init_image.resize((2048, 2048))

    generator = torch.Generator(device="cpu").manual_seed(5)

    mask_image = load_image(
        "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png")
    mask_image = mask_image.resize((2048, 2048))



    control_image = make_canny_condition(init_image)

    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                                      controlnet=controlnet, torch_dtype=torch.float16)
    # pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe.enable_model_cpu_offload()
    width = 2048  # gr.Slider(minimum=1024, maximum=4096, step=1024, value=2048, label="Width")
    height = 2048  # gr.Slider(minimum=1024, maximum=4096, step=1024, value=2048, label="Height")
    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

    # generate image
    image = pipe(

        "a handsome man with ray-ban sunglasses, 4k image",
        negative_prompt = negative_prompt,
        num_inference_steps=20,
        generator=generator,
        eta=1.0,
        image=init_image,
        mask_image=mask_image,
        control_image=control_image,

        tar_height=height,
        tar_width=width,
        save_image_tag=True,
        show_image=True
    )

    os.makedirs("control_out", exist_ok=True)

    for ii, img in enumerate(image):
        img.save(f"control_out/{ii}_finalout.png")


def get_start_size(width,height,fix_size=1024):
    w, h = width,height
    aspect_ratio = w / h
    size = (min(fix_size, int(fix_size * aspect_ratio)),
            min(fix_size, int(fix_size / aspect_ratio)))

    return size


def main():
    """
    real image editing or inpainting
    直接对整个图片文件夹进行推理

    """
    import torch

    from diffusers.utils import load_image
    import time
    import argparse

    parser = argparse.ArgumentParser(description="UltraDiffEdit trainer")
    parser.add_argument("--file_root",
                        default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DIV2K_edit/img",
                        type=str, help="path to the input images")
    parser.add_argument("--mask_root",
                        default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DIV2K_edit/mask_dilate",
                        type=str, help="path to the masks")
    parser.add_argument("--caption_root",
                        default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DIV2K_edit/caption",
                        type=str, help="path to the captions")
    parser.add_argument("--view_batch_size", type=int, default=16, help="the batch size to use")
    parser.add_argument("--start_idx", type=int, default=10, help="the indx of the image to start")

    parser.add_argument("--end_idx", type=int, default=10000, help="the index of the image to end")

    parser.add_argument('--arch', type=str, default='pipeline_anysize_controlnet_inpaint_sd_xlv3',
                        help='models architectures (pipeline_anysize_controlnet_inpaint_sd_xlv3)')

    parser.add_argument('--ckpt', type=str, default=None, help='the checkpoint path pf the model')
    parser.add_argument('--control_net_ckpt', type=str, default=None, help='the checkpoint path pf the model')

    parser.add_argument('--save_root', type=str, default='./results',
                        help='root for saving the images')
    args = parser.parse_args()

    name_t = str(time.time())
    os.makedirs(args.save_root, exist_ok=True)


    if args.ckpt == None:
        args.ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
        args.control_net_ckpt = "diffusers/controlnet-canny-sdxl-1.0"
    else:
        # ckpt_ = "/home/longlong/longlong/Models/Diffusers/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
        # ckpt_ = "/home/onelong/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
        # ckpt = "/home/x/xiantaj/longlong/workdir/Models/Diffusers/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"

        # args.control_net_ckpt = "/gpfs/fs0/scratch/x/xiantaj/longlong/cache/huggingface/hub/models--diffusers--controlnet-canny-sdxl-1.0/snapshots/eb115a19a10d14909256db740ed109532ab1483c"
        # args.control_net_ckpt =  "/gpfs/fs0/scratch/x/xiantaj/longlong/cache/huggingface/hub/models--diffusers--controlnet-depth-sdxl-1.0/snapshots/17bb97973f29801224cd66f192c5ffacf82648b4"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("model load from  ", args.ckpt)

    img_list = []
    mask_list = []
    caption_list = []
    ut.listdir(args.file_root, img_list)
    ut.listdir(args.mask_root, mask_list)
    ut.listdir(args.caption_root, caption_list)

    generator = torch.Generator(device='cpu')  # random seed generator
    generator = generator.manual_seed(5)

    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
    for ii, (img_path, mask_path, caption_path) in enumerate(zip(img_list, mask_list, caption_list)):
        # name_ = f"{ii}"
        name_ = os.path.basename(img_path)
        name_ = str(name_).strip().split(".")[0]
        if ii < args.start_idx or ii >= args.end_idx: continue

        print("img_path:", img_path)
        print("mask_path:", mask_path)
        print("caption_path:", caption_path)

        init_image = load_image(img_path).convert("RGB")
        init_image.save(f"{args.save_root}/{name_}_oriinput.png")

        width, height = init_image.size
        # get the first stage
        x1_width, x1_hight  = get_start_size(width,height)
        x1_init_image = init_image.resize((x1_width, x1_hight))
        print("the x1 image size is: ",x1_width, x1_hight )

        x1_control_image = make_canny_condition(x1_init_image)

        x1_mask_image = load_image(mask_path).convert("RGB").resize((x1_width, x1_hight))
        prompt = read_lines_from_file(caption_path)[0]
        print("the prompt is: ", prompt)
        #
        from diffusers import ControlNetModel, DDIMScheduler
        from pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
        #
        controlnet = ControlNetModel.from_pretrained(args.control_net_ckpt , torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(args.ckpt,
                                                                          controlnet=controlnet,
                                                                          torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()

        # generate image
        ini_out_image = pipe(
            prompt,
            num_inference_steps=20,
            generator=generator,
            eta=1.0,
            image=x1_init_image,
            mask_image=x1_mask_image,
            control_image=x1_control_image,
        ).images[0]
        ini_out_image.save(f"{args.save_root}/{name_}_first_out.png")

        ### stage two
        control_image = x1_control_image.resize((width, height))
        control_image.save(f"{args.save_root}/{name_}_canny_img.png")

        print("the original image size is: ", width, height)
        mask_image = load_image(mask_path).convert("RGB").resize((width, height))
        mask_image.save(f"{args.save_root}/{name_}_orimask.png")
        ini_out_image = ini_out_image.resize((width, height))

        max_size = max(width, height)
        max_scale = math.ceil(float(max_size) / 1024)

        # x1_width, x1_hight  = get_start_size(width,height)

        min_padding = math.lcm(max_scale * 8, 1024)
        # Pad the image
        init_image, orig_size = pad_image_to_multiple_num(init_image, min_padding, color=(255, 255, 255))
        ini_out_image, orig_size = pad_image_to_multiple_num(ini_out_image, min_padding, color=(255, 255, 255))
        mask_image, orig_size = pad_image_to_multiple_num(mask_image, min_padding, color=(0, 0, 0))

        width, height = init_image.size
        print("the padded image size is: ", width, height)

        start = time.time()

        from pipeline_anysize_inpaintv15_revise import StableAnysizeInpaintPipeline
        print("we use pipeline_anysize_inpaintv15_revise")

        pipe = StableAnysizeInpaintPipeline.from_pretrained(
            # "stabilityai/stable-diffusion-xl-base-1.0",
            pretrained_model_name_or_path=args.ckpt,
            # torch_dtype = torch.float32,
            # custom_pipeline="multimodalart/sdxl_perturbed_attention_guidance",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        pipe.to("cuda")

        images = pipe.refine_editing(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            content_img = ini_out_image,
            mask_image=mask_image, num_inference_steps=50, strength=0.80,
            # pag_scale = 5,
            ##
            # height = height, width = width,
            tar_height=height,
            tar_width=width,
            view_batch_size=args.view_batch_size,
            stride=64,
            cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, sigma=0.8,
            multi_decoder=True, show_image=False,
            save_image_tag=False,
            file_name=name_,
            save_root=args.save_root,
        )

        for ii, img in enumerate(images):
            # img.save(f"results/{name_}_{ii}_out.png")
            w, h = img.size
            ratio_w = w / width * orig_size[0]
            ratio_h = h / height * orig_size[1]
            #
            img = crop_image_to_original(img, (ratio_w, ratio_h))
            # img.save(f"{args.save_root}/{name_}_{ii}_out.png")

            if ii == len(images) - 1:
                img.save(f"{args.save_root}/{name_}_finalout.png")

        end = time.time()
        print('time for running is : %s Seconds' % (end - start))


def main_v2():
    """
    real image editing or inpainting
    直接对整个图片文件夹进行推理

    """
    import torch

    from diffusers.utils import load_image
    import time
    import argparse

    parser = argparse.ArgumentParser(description="UltraDiffEdit trainer")
    parser.add_argument("--file_root",
                        default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DIV2K_edit/img",
                        type=str, help="path to the input images")
    parser.add_argument("--conditional_root",
                        default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DIV2K_edit/img",
                        type=str, help="path to the input images")
    parser.add_argument("--mask_root",
                        default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DIV2K_edit/mask_dilate",
                        type=str, help="path to the masks")
    parser.add_argument("--caption_root",
                        default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DIV2K_edit/caption",
                        type=str, help="path to the captions")
    parser.add_argument("--view_batch_size", type=int, default=16, help="the batch size to use")
    parser.add_argument("--start_idx", type=int, default=10, help="the indx of the image to start")

    parser.add_argument("--end_idx", type=int, default=10000, help="the index of the image to end")

    parser.add_argument('--arch', type=str, default='pipeline_anysize_controlnet_inpaint_sd_xlv3',
                        help='models architectures (pipeline_anysize_controlnet_inpaint_sd_xlv3)')

    parser.add_argument('--ckpt', type=str, default=None, help='the checkpoint path pf the model')
    parser.add_argument('--control_net_ckpt', type=str, default=None, help='the checkpoint path pf the model')

    parser.add_argument('--save_root', type=str, default='./results',
                        help='root for saving the images')
    parser.add_argument("--beta1", type=float, default=3, help="the beta1 weight")
    parser.add_argument("--beta2", type=float, default=1, help="the beta2 weight")
    parser.add_argument("--ug_weight", type=float, default=0.2, help="the ug_weight weight")
    parser.add_argument('--stages', type=str, default='two',help='[two, three, S] stages setting for inference')

    args = parser.parse_args()

    name_t = str(time.time())
    os.makedirs(args.save_root, exist_ok=True)

    if args.ckpt == None:
        args.ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
        args.control_net_ckpt = "diffusers/controlnet-canny-sdxl-1.0"
    else:
        # ckpt_ = "/home/longlong/longlong/Models/Diffusers/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
        # ckpt_ = "/home/onelong/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
        # ckpt = "/home/x/xiantaj/longlong/workdir/Models/Diffusers/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"

        # args.control_net_ckpt = "/gpfs/fs0/scratch/x/xiantaj/longlong/cache/huggingface/hub/models--diffusers--controlnet-canny-sdxl-1.0/snapshots/eb115a19a10d14909256db740ed109532ab1483c"
        # args.control_net_ckpt =  "/gpfs/fs0/scratch/x/xiantaj/longlong/cache/huggingface/hub/models--diffusers--controlnet-depth-sdxl-1.0/snapshots/17bb97973f29801224cd66f192c5ffacf82648b4"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("model load from  ", args.ckpt)

    img_list = []
    condi_img_list = []

    mask_list = []
    caption_list = []
    
    ut.listdir(args.conditional_root, condi_img_list)
    ut.listdir(args.file_root, img_list)
    ut.listdir(args.mask_root, mask_list)
    ut.listdir(args.caption_root, caption_list)

    generator = torch.Generator(device='cpu')  # random seed generator
    generator = generator.manual_seed(5)

    my_caption_list = [
        "A futuristic glass dome in the woods",
        "A cozy log cabin with smoke coming out of the chimney in the woods",
        "An abandoned medieval castle in the woods",
        "A giant mushroom house in the woods",
        "A spaceship landing site in the woods",
        "A traditional Japanese tea house in the woods",
        "A mystical glowing crystal in the woods",
        "A towering wooden watchtower in the woods",
        "A rustic treehouse nestled high in the woods",
        "A bustling elf village in the woods",
        "A stone circle with glowing runes in the woods",
        "A modern minimalist cube-shaped house in the woods",
        "A colorful hot air balloon tied down in the woods",
        "A mysterious glowing portal in the woods",
        "A superhero flying through the woods",
        "A small tent with a campfire in the woods",
        "A Viking longhouse with a thatched roof in the woods",
        "A giant clock tower standing alone in the woods",
        "A fantasy dragon’s nest in the woods",
        "A ruined ancient temple reclaimed by nature in the woods",
        "A massive library hidden in the woods",
        "A magical fairy circle with glowing lights in the woods",
        "A robotic outpost blending into the woods",
        "A tropical bamboo hut surrounded by greenery in the woods",
        "A luxurious mansion with a pool in the woods",
        "A glowing alien artifact hovering in the woods",
        "A quaint bakery with a wooden sign in the woods",
        "A Victorian greenhouse filled with exotic plants in the woods",
        "A massive bear cave with glowing eyes in the woods",
        "A whimsical candy house inspired by a fairytale in the woods",
        "A charming thatched cottage in the woods",
        "A secluded wooden cabin in the woods",
        "A small stone house with ivy growing on it in the woods",
        "A cozy A-frame cabin in the woods",
        "A picturesque chalet with a balcony in the woods",
        "A log house surrounded by colorful flowers in the woods",
        "A whimsical gingerbread house in the woods",
        "An old farmhouse with broken windows in the woods",
        "A tiny hobbit hole tucked into a hill in the woods",
        "A wooden windmill spinning gently in the woods",
        "A rustic barn covered in moss in the woods",
        "An abandoned hunting lodge in the woods",
        "A secret bunker hidden in the woods",
        "A charming brick cottage with a chimney in the woods",
        "A small chapel with stained glass windows in the woods",
        "A colorful gypsy wagon parked in the woods",
        "A snow-covered winter cabin in the woods",
        "A luxurious treehouse overlooking the woods",
        "A fairytale castle with towers peeking through the woods",
        "A miner’s hut with tools scattered around in the woods",
        "A spooky, rundown shack in the woods",
        "A classic hunting cabin with mounted antlers in the woods",
        "A modern tiny house with large glass windows in the woods",
        "A mysterious cave entrance with torches in the woods",
        "A rustic boathouse near a pond in the woods",
        "A small chapel made of wood and stone in the woods",
        "A mountain lodge with a large balcony in the woods",
        "A hidden artist’s studio surrounded by easels in the woods",
        "A charming Victorian-style gazebo in the woods",
        "An enchanted witch’s hut with a crooked roof in the woods"
    ]

    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
    for ii, (img_path, mask_path, caption_path) in enumerate(zip(img_list, mask_list, caption_list)):
        # name_ = f"{ii}"
        name_ = os.path.basename(img_path)
        name_ = str(name_).strip().split(".")[0]
        if ii < args.start_idx or ii >= args.end_idx: continue

        ###
        my_caption_list = my_caption_list[::-1]
        for prompt,condi_path in zip(my_caption_list,condi_img_list):

            print("prompt:", prompt)
            print("img_path:", img_path) 
            print("mask_path:", mask_path)
            # print("caption_path:", caption_path)
            print("condi_path:", condi_path)

            condi_image = load_image(condi_path).convert("RGB")
            condi_image.save(f"{args.save_root}/{name_}_condi_image.png")

            init_image = load_image(img_path).convert("RGB")
            init_image.save(f"{args.save_root}/{name_}_oriinput.png")

            width, height = init_image.size
            # get the first stage
            x1_width, x1_hight = get_start_size(width, height)

            x1_condi_image = condi_image.resize((x1_width, x1_hight))

            x1_init_image = init_image.resize((x1_width, x1_hight))
            print("the x1 image size is: ", x1_width, x1_hight)

            # x1_control_image = make_canny_condition(x1_init_image)
            x1_control_image = make_canny_condition(x1_condi_image)

            x1_mask_image = load_image(mask_path).convert("RGB").resize((x1_width, x1_hight))
            # prompt = read_lines_from_file(caption_path)[0]
            # print("the prompt is: ", prompt)


            #
            from diffusers import ControlNetModel, DDIMScheduler
            from pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
            #
            controlnet = ControlNetModel.from_pretrained(args.control_net_ckpt, torch_dtype=torch.float16)
            controlnet.to("cuda")
            pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(args.ckpt,
                                                                            controlnet=controlnet,
                                                                                   torch_dtype=torch.float16)
            # pipe.enable_model_cpu_offload()
            pipe.to("cuda")


            # generate image
            ini_out_image = pipe(
                prompt,
                num_inference_steps=20,
                generator=generator,
                eta=1.0,
                image=x1_init_image,
                mask_image=x1_mask_image,
                control_image=x1_control_image,
            ).images[0]
            ini_out_image.save(f"{args.save_root}/{name_}_first_out.png")

            ### stage two
            control_image = x1_control_image.resize((width, height))
            control_image.save(f"{args.save_root}/{name_}_canny_img.png")

            print("the original image size is: ", width, height)
            mask_image = load_image(mask_path).convert("RGB").resize((width, height))
            mask_image.save(f"{args.save_root}/{name_}_orimask.png")
            ini_out_image = ini_out_image.resize((width, height))

            max_size = max(width, height)
            max_scale = math.ceil(float(max_size) / 1024)

            # x1_width, x1_hight  = get_start_size(width,height)

            min_padding = math.lcm(max_scale * 8, 1024)
            # Pad the image
            init_image, orig_size = pad_image_to_multiple_num(init_image, min_padding, color=(255, 255, 255))
            ini_out_image, orig_size = pad_image_to_multiple_num(ini_out_image, min_padding, color=(255, 255, 255))
            mask_image, orig_size = pad_image_to_multiple_num(mask_image, min_padding, color=(0, 0, 0))

            width, height = init_image.size
            print("the padded image size is: ", width, height)

            start = time.time()
            from pipeline_ultradiffedit_sdxl import  StableAnysizeInpaintPipeline  #
            # from pipeline_anysize_inpaintv15_revise import StableAnysizeInpaintPipeline
            print("we use pipeline_anysize_inpaintv15_revise")

            pipe = StableAnysizeInpaintPipeline.from_pretrained(
                # "stabilityai/stable-diffusion-xl-base-1.0",
                pretrained_model_name_or_path=args.ckpt,
                # torch_dtype = torch.float32,
                # custom_pipeline="multimodalart/sdxl_perturbed_attention_guidance",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            pipe.to("cuda")

            images = pipe.refine_editing(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                content_img=ini_out_image,
                mask_image=mask_image, num_inference_steps=50, strength=0.80,
                # pag_scale = 5,
                ##
                # height = height, width = width,
                tar_height=height,
                tar_width=width,
                view_batch_size=args.view_batch_size,
                stride=64,
                # cosine_scale_1=3, cosine_scale_2=1,
                cosine_scale_3=1, sigma=0.8,
                multi_decoder=True, show_image=False,
                save_image_tag=False,
                file_name=name_,
                save_root=args.save_root,

                generator=generator,

                beta_scale_1=args.beta1, beta_scale_2=args.beta2,

                run_stage=args.stages,
                ug_weight=args.ug_weight,
                orig_size=[orig_size[1], orig_size[0]]
            )

            for ii, img in enumerate(images):
                # img.save(f"results/{name_}_{ii}_out.png")
                w, h = img.size
                ratio_w = w / width * orig_size[0]
                ratio_h = h / height * orig_size[1]
                #
                img = crop_image_to_original(img, (ratio_w, ratio_h))
                # img.save(f"{args.save_root}/{name_}_{ii}_out.png")

                if ii == len(images) - 1:
                    img.save(f"{args.save_root}/{name_}_finalout.png")

            end = time.time()
            print('time for running is : %s Seconds' % (end - start))


if __name__ == '__main__':
    # ultra_controlnet_canny()
    # ultra_depth()
    # v1()
    # main()
    main_v2()
