import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline
import os
import Util.utils_train  as ut
from Util.img_pad_crop import pad_image_to_multiple_num,crop_image_to_original,pad_image_to_square,get_start_size
from Util.caption_read import read_lines_from_file
import math





def demo():
    from diffusers.utils import load_image
    from diffusers.image_processor import VaeImageProcessor
    import time
    name_ = str(time.time())
    #
    model_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DemoFusionSDXLPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    #
    # mask_path = "/home/onelong/Longlongaaago/dataset/DemoFusion_data/demofusion_img_demo/A freshly baked loaf of bread on a rustic kitchen counter./img_1024_mask.png"
    # img_path = "/home/onelong/Longlongaaago/dataset/DemoFusion_data/demofusion_img_demo/A freshly baked loaf of bread on a rustic kitchen counter./img_4096.jpg"
    # # img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    # # ori_init_image = load_image(img_url).convert("RGB")
    # ori_init_image = load_image(img_path).convert("RGB")
    # image_processor = VaeImageProcessor()
    # init_image = image_processor.preprocess(ori_init_image, height=1024, width=1024).to("cuda",dtype=torch.float16)
    #
    # width, height = ori_init_image.size
    # ori_mask_image = load_image(mask_path).convert("RGB").resize((width, height))
    # mask_image = image_processor.preprocess(ori_mask_image, height=1024, width=1024).to("cuda",dtype=torch.float16)
    # inpainted_image = init_image * (1 - mask_image) + init_image * mask_image
    #
    # # prompt = "Envision a portrait of an elderly woman, her face a canvas of time, framed by a headscarf with muted tones of rust and cream. Her eyes, blue like faded denim. Her attire, simple yet dignified."
    # # prompt = "two tigers."
    # prompt = "a cute cat sitting on a bench"
    # negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
    #
    # generator = torch.Generator(device='cuda') #random seed generator
    # generator = generator.manual_seed(5)
    #
    # images = pipe(prompt, negative_prompt=negative_prompt, generator=generator,
    #               height=4096, width=4096, view_batch_size=16, stride=64,
    #               num_inference_steps=50, guidance_scale = 7.5,
    #               cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, sigma=0.8,
    #               multi_decoder=True, show_image=True,
    #               image_lr = init_image
    #              )

    ##### generationg

    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    init_image = load_image(img_url).convert("RGB")
    image_processor = VaeImageProcessor()
    init_image = image_processor.preprocess(init_image, height=1024, width=1024).to("cuda", dtype=torch.float16)

    prompt = "Envision a portrait of an elderly woman, her face a canvas of time, framed by a headscarf with muted tones of rust and cream. Her eyes, blue like faded denim. Her attire, simple yet dignified."
    prompt = "two tigers."
    prompt = "a cute cat sitting on a bench"
    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

    generator = torch.Generator(device='cuda')  # random seed generator
    # generator = generator.manual_seed(2013)
    generator = generator.manual_seed(5)

    images = pipe(prompt, negative_prompt=negative_prompt, generator=generator,
                  height=3072, width=3072, view_batch_size=16, stride=64,
                  num_inference_steps=50, guidance_scale=7.5,
                  cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, sigma=0.8,
                  multi_decoder=True, show_image=True,
                  # image_lr=init_image
                  )

    for ii, img in enumerate(images):
        img.save(f"results/{name_}_{ii}_out.png")

        width, height = img.size
        # mask_image = image_processor.preprocess(ori_mask_image, height=width, width=height).to("cuda", dtype=torch.float16)
        # img_ = image_processor.preprocess(img, height=width, width=height).to("cuda", dtype=torch.float32)
        # ori_img_ = image_processor.preprocess(ori_init_image, height=width, width=height).to("cuda", dtype=torch.float32)
        #
        # inpainted_image = ori_img_ * (mask_image < 0.5) + img_ *  (mask_image >= 0.5)
        #
        # image = image_processor.postprocess(inpainted_image, output_type="pil")
        # image[0].save(f"results/{name_}_{ii}_inpaint_out.png")




def model_inference_real():

    """
    demofision comparison
    real image editing or inpainting
    直接对整个图片文件夹进行推理

    """
    from diffusers.utils import load_image
    from diffusers.image_processor import VaeImageProcessor
    from pipeline_demofusion_sdxl_edit import DemoFusionSDXLPipeline

    import time
    name_ = str(time.time())

    from diffusers.utils import load_image
    import time
    import argparse

    parser = argparse.ArgumentParser(description="UltraDiffEdit trainer")
    # parser.add_argument("--file_root", default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/image", type=str, help="path to the input images")
    # parser.add_argument("--mask_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/mask_dilate", type=str, help="path to the masks")
    # parser.add_argument("--caption_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/caption2", type=str, help="path to the captions")
    parser.add_argument("--file_root",
                        default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DIV2K_edit/img",
                        type=str, help="path to the input images")
    parser.add_argument("--mask_root",
                        default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DIV2K_edit/mask_dilate",
                        type=str, help="path to the masks")
    parser.add_argument("--caption_root",
                        default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DIV2K_edit/caption",
                        type=str, help="path to the captions")
    
    parser.add_argument("--start_idx", type=int, default=2, help="the indx of the image to start")
    parser.add_argument("--end_idx", type=int, default=10000, help="the index of the image to end")
    parser.add_argument('--ckpt', type=str, default=None ,help='the checkpoint path pf the model')

    parser.add_argument('--arch', type=str, default='pipeline_anysize_inpaintv5', help='models architectures (pipeline_anysize_inpaintv5)')
    parser.add_argument('--save_root', type=str, default='./results', help='root for saving the images')
    args = parser.parse_args()

    name_t = str(time.time())
    os.makedirs(args.save_root, exist_ok=True)

    if args.ckpt ==None:
        args.ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # model_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DemoFusionSDXLPipeline.from_pretrained(args.ckpt, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # pipe = StableAnysizeInpaintPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0",
    #     # torch_dtype = torch.float32,
    #     torch_dtype=torch.float16,
    #     variant="fp16",
    #     use_safetensors=True,
    # )
    # pipe.to("cuda")


    img_list = []
    mask_list = []
    caption_list = []
    ut.listdir(args.file_root, img_list)
    ut.listdir(args.mask_root, mask_list)
    ut.listdir(args.caption_root, caption_list)

    # generator = torch.Generator(device='cuda')  # random seed generator
    # generator = generator.manual_seed(5)

    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
    for ii, (img_path,mask_path,caption_path) in enumerate(zip(img_list,mask_list,caption_list)):
        # name_ = f"{ii}"
        name_ = os.path.basename(img_path)
        name_ = str(name_).strip().split(".")[0]
        #
        if ii <args.start_idx or ii>= args.end_idx : continue
        print("img_path:",img_path)
        print("mask_path:",mask_path)
        print("caption_path:",caption_path)

        init_image = load_image(img_path).convert("RGB")
        init_image.save(f"{args.save_root}/{name_}_oriinput.png")

        width, height = init_image.size
        print("the original image size is: ", width, height)
        mask_image = load_image(mask_path).convert("RGB").resize((width, height))
        mask_image.save(f"{args.save_root}/{name_}_orimask.png")

        prompt = read_lines_from_file(caption_path)[0]
        print("the prompt is: ", prompt)


        max_size = max(width, height)
        max_scale = math.ceil(float(max_size)/1024)

        # x1_width, x1_hight  = get_start_size(width,height)

        min_padding = math.lcm(max_scale*8,1024)
        # Pad the image
        init_image, orig_size = pad_image_to_multiple_num(init_image,min_padding,color=(255,255,255))
        mask_image, orig_size = pad_image_to_multiple_num(mask_image,min_padding,color=(0,0,0))
        # init_image = pad_image_to_square(init_image,color=(255,255,255))
        # mask_image = pad_image_to_square(mask_image,color=(0,0,0))

        width, height = init_image.size
        print("the padded image size is: ", width, height)

        start = time.time()
        # init_image.save(f"{args.save_root}/{name_}_input.png")
        # mask_image.save(f"{args.save_root}/{name_}_mask.png")
        # images = pipe(
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80,
        #     ##
        #     height = height, width = width,
        #     # tar_height=height,
        #     # tar_width=width,
        #     view_batch_size=16, stride=64,
        #     cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, sigma=0.8,
        #     multi_decoder=True, show_image=False,
        #     save_image_tag = False,
        #     file_name=name_,
        #     save_root=args.save_root,
        #
        # )

        tar_width,tar_height = width,height
        width_, height_ = get_start_size(width, height, fix_size=1024)
        image_processor = VaeImageProcessor()
        init_image = image_processor.preprocess(init_image, height=height_, width=width_).to("cuda",dtype=torch.float16)
        mask_image = image_processor.preprocess(mask_image, height=height_, width=width_).to("cuda",dtype=torch.float16)
        # inpainted_image = init_image * (1 - mask_image) + init_image * mask_image


        ####
        generator = torch.Generator(device='cuda') #random seed generator
        import random
        # seeds = random.randint(1,100)
        # generator = generator.manual_seed(seeds)
        generator = generator.manual_seed(5)

        images = pipe(prompt, negative_prompt=negative_prompt, generator=generator,
                      height=tar_height, width=tar_width,
                      view_batch_size=16, stride=64,
                      num_inference_steps=50, guidance_scale = 7.5,
                      cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, sigma=0.8,
                      multi_decoder=True, show_image=False,
                      image_lr = init_image
                     )

        # 需要计算的代码块

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

        #
        # Later, if you want to crop it back



def generate_random_image(width, height):
    import numpy as np
    # 生成随机像素数据
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    # 创建 Image 对象
    image = Image.fromarray(data, 'RGB')
    return image


def model_resource_inference():

    """
    demofision comparison
    real image editing or inpainting
    直接对整个图片文件夹进行推理

    """
    from diffusers.utils import load_image
    from diffusers.image_processor import VaeImageProcessor
    from pipeline_demofusion_sdxl_edit import DemoFusionSDXLPipeline

    import time
    name_ = str(time.time())

    from diffusers.utils import load_image
    import time
    import argparse

    parser = argparse.ArgumentParser(description="UltraDiffEdit trainer")
    # parser.add_argument("--file_root", default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/image", type=str, help="path to the input images")
    # parser.add_argument("--mask_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/mask_dilate", type=str, help="path to the masks")
    # parser.add_argument("--caption_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/caption2", type=str, help="path to the captions")
    # parser.add_argument("--file_root",
    #                     default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DIV2K_edit/img",
    #                     type=str, help="path to the input images")
    # parser.add_argument("--mask_root",
    #                     default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DIV2K_edit/mask_dilate",
    #                     type=str, help="path to the masks")
    # parser.add_argument("--caption_root",
    #                     default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DIV2K_edit/caption",
    #                     type=str, help="path to the captions")
    
    # parser.add_argument("--start_idx", type=int, default=2, help="the indx of the image to start")
    # parser.add_argument("--end_idx", type=int, default=10000, help="the index of the image to end")
    parser.add_argument('--ckpt', type=str, default=None ,help='the checkpoint path pf the model')

    parser.add_argument('--arch', type=str, default='pipeline_anysize_inpaintv5', help='models architectures (pipeline_anysize_inpaintv5)')
    parser.add_argument('--save_root', type=str, default='./results', help='root for saving the images')
    args = parser.parse_args()

    name_t = str(time.time())
    os.makedirs(args.save_root, exist_ok=True)

    if args.ckpt ==None:
        args.ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # model_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DemoFusionSDXLPipeline.from_pretrained(args.ckpt, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")



    # img_list = []
    # mask_list = []
    # caption_list = []
    # ut.listdir(args.file_root, img_list)
    # ut.listdir(args.mask_root, mask_list)
    # ut.listdir(args.caption_root, caption_list)

    # generator = torch.Generator(device='cuda')  # random seed generator
    # generator = generator.manual_seed(5)

    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
    # for ii, (img_path,mask_path,caption_path) in enumerate(zip(img_list,mask_list,caption_list)):
    #     # name_ = f"{ii}"
    #     name_ = os.path.basename(img_path)
    #     name_ = str(name_).strip().split(".")[0]
    #     #
    #     if ii <args.start_idx or ii>= args.end_idx : continue
    #     print("img_path:",img_path)
    #     print("mask_path:",mask_path)
    #     print("caption_path:",caption_path)


    # size_list = [512,1024,2048,3072,4096,5120,6144,7168,8192]
    size_list = [8192]

    for size_ in size_list:
        print("for the size:",size_)
        init_image = generate_random_image(size_,size_)
        # init_image = load_image(img_path).convert("RGB")
        # init_image.save(f"{args.save_root}/{name_}_oriinput.png")

        width, height = init_image.size
        print("the original image size is: ", width, height)
        mask_image = generate_random_image(size_,size_)
        # mask_image = load_image(mask_path).convert("RGB").resize((width, height))
        # mask_image.save(f"{args.save_root}/{name_}_orimask.png")

        prompt = "asdasdsadsadasda"
        print("the prompt is: ", prompt)

        # max_size = max(width, height)
        # max_scale = math.ceil(float(max_size)/1024)

        # x1_width, x1_hight  = get_start_size(width,height)

        # min_padding = math.lcm(max_scale*8,1024)
        # # Pad the image
        # init_image, orig_size = pad_image_to_multiple_num(init_image,min_padding,color=(255,255,255))
        # mask_image, orig_size = pad_image_to_multiple_num(mask_image,min_padding,color=(0,0,0))
        # init_image = pad_image_to_square(init_image,color=(255,255,255))
        # mask_image = pad_image_to_square(mask_image,color=(0,0,0))

        width, height = init_image.size
        print("the padded image size is: ", width, height)

        start = time.time()
        # init_image.save(f"{args.save_root}/{name_}_input.png")
        # mask_image.save(f"{args.save_root}/{name_}_mask.png")
        # images = pipe(
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80,
        #     ##
        #     height = height, width = width,
        #     # tar_height=height,
        #     # tar_width=width,
        #     view_batch_size=16, stride=64,
        #     cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, sigma=0.8,
        #     multi_decoder=True, show_image=False,
        #     save_image_tag = False,
        #     file_name=name_,
        #     save_root=args.save_root,
        #
        # )

        tar_width,tar_height = width,height
        width_, height_ = get_start_size(width, height, fix_size=1024)
        image_processor = VaeImageProcessor()
        init_image = image_processor.preprocess(init_image, height=height_, width=width_).to("cuda",dtype=torch.float16)
        mask_image = image_processor.preprocess(mask_image, height=height_, width=width_).to("cuda",dtype=torch.float16)
        # inpainted_image = init_image * (1 - mask_image) + init_image * mask_image


        ####
        generator = torch.Generator(device='cuda') #random seed generator
        import random
        seeds = random.randint(1,100)
        generator = generator.manual_seed(seeds)
        # generator = generator.manual_seed(5)

        images = pipe(prompt, negative_prompt=negative_prompt, generator=generator,
                    height=tar_height, width=tar_width,
                    view_batch_size=16, stride=64,
                    num_inference_steps=50, guidance_scale = 7.5,
                    cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, sigma=0.8,
                    multi_decoder=True, show_image=False,
                    # image_lr = init_image,
                    image_lr = None,

                    )

        # 需要计算的代码块
        # for ii, img in enumerate(images):
        #     # img.save(f"results/{name_}_{ii}_out.png")
        #     w, h = img.size
        #     ratio_w = w / width * orig_size[0]
        #     ratio_h = h / height * orig_size[1]
        #     #
        #     img = crop_image_to_original(img, (ratio_w, ratio_h))
        #     # img.save(f"{args.save_root}/{name_}_{ii}_out.png")

        #     if ii == len(images) - 1:
        #         img.save(f"{args.save_root}/{name_}_finalout.png")

        end = time.time()
        print('time for running is : %s Seconds' % (end - start))

     




if __name__ == '__main__':
    # model_inference_real()
    model_resource_inference()