


def run_demo():
    """
    Minimal UltraDiffEdit demo.
    """
    from diffusers.utils import load_image
    from pipeline_ultradiffedit_sdxl import StableAnysizeInpaintPipeline
    import time
    import torch
    import os

    name_ = str(time.time())
    os.makedirs("./results", exist_ok=True)
    pipe = StableAnysizeInpaintPipeline.from_pretrained(
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
            run_stage="two",
            ug_weight = 0.2,

        )

    images[-1].save(f"results/{name_}_finalout.png")

    end = time.time()
    print('time for running is : %s Seconds' % (end - start))


if __name__ == '__main__':
    run_demo()
