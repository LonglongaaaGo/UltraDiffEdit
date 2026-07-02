


def get_start_size(width,height,fix_size=1024):
    w, h = width,height
    aspect_ratio = w / h
    size = (min(fix_size, int(fix_size * aspect_ratio)),
            min(fix_size, int(fix_size / aspect_ratio)))

    return size

def main():
    """
    ultraDiffEdit main script
    """
    from Util.img_pad_crop import pad_image_to_multiple_num,crop_image_to_original,pad_image_to_square
    from Util.caption_read import read_lines_from_file
    from diffusers.utils import load_image
    import argparse
    from pipeline_ultradiffedit_sdxl import StableAnysizeInpaintPipeline
    import math
    import time
    import torch
    import os
    import Util.utils_train  as ut
    
    parser = argparse.ArgumentParser(description="UltraDiffEdit inference")
    # parser.add_argument("--file_root", default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/image", type=str, help="path to the input images")
    # parser.add_argument("--mask_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/mask_dilate", type=str, help="path to the masks")
    # parser.add_argument("--caption_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/caption2", type=str, help="path to the captions")
    
    parser.add_argument("--file_root", default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/Sys2K/img", type=str, help="path to the input images")
    parser.add_argument("--mask_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/Sys2K/mask_dilate", type=str, help="path to the masks")
    parser.add_argument("--caption_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/Sys2K/text_prompt", type=str, help="path to the captions")
    
    parser.add_argument("--start_idx", type=int, default=42, help="the indx of the image to start")
    parser.add_argument("--end_idx", type=int, default=999999, help="the index of the image to end")

    parser.add_argument("--beta1", type=float, default=3, help="the beta1 weight")
    parser.add_argument("--beta2", type=float, default=1, help="the beta2 weight")
    parser.add_argument("--ug_weight", type=float, default=0.2, help="the ug_weight weight")

    parser.add_argument("--view_batch_size", type=int, default=16, help="the batch size to use")
    parser.add_argument('--ckpt', type=str, default=None ,help='the checkpoint path pf the model')
    parser.add_argument('--save_img', type=bool, default=False ,help='if or not save the intermediate images')
    parser.add_argument('--save_root', type=str, default='./results',help='root for saving the images')
    parser.add_argument('--stages', type=str, default='two',help='[two, three, S] stages setting for inference')
    parser.add_argument("--stride", type=int, default=64, help="the stride to use for the inference")

    args = parser.parse_args()
    
    os.makedirs(args.save_root, exist_ok=True)

    if args.ckpt ==None:
        args.ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("model load from  ",args.ckpt)

    pipe = StableAnysizeInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path= args.ckpt,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")

    img_list = []
    mask_list = []
    caption_list = []
    ut.listdir(args.file_root, img_list)
    ut.listdir(args.mask_root, mask_list)
    ut.listdir(args.caption_root, caption_list)

    import random 
    generator = torch.Generator(device='cuda')  # random seed generator
    ##
    seed_ = random.randint(1,10000)
    generator = generator.manual_seed(seed_)
    # generator = generator.manual_seed(5)

    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
    for ii, (img_path,mask_path,caption_path) in enumerate(zip(img_list,mask_list,caption_list)):
        name_ = os.path.basename(img_path)
        name_ = str(name_).strip().split(".")[0]
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

        min_padding = math.lcm(max_scale*8,1024)
        # Pad the image
        init_image, orig_size = pad_image_to_multiple_num(init_image,min_padding,color=(255,255,255))
        mask_image, orig_size = pad_image_to_multiple_num(mask_image,min_padding,color=(0,0,0))

        width, height = init_image.size
        print("the padded image size is: ", width, height)

        start = time.time()
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80,
            generator = generator,
            tar_height=height,
            tar_width=width,
            view_batch_size=args.view_batch_size,
            stride=args.stride,
            beta_scale_1=args.beta1, beta_scale_2=args.beta2, 
            cosine_scale_3=1, sigma=0.8,
            multi_decoder=True, show_image=False,
            save_image_tag = False,
            file_name=name_,
            save_root=args.save_root,
            run_stage = args.stages,
            ug_weight = args.ug_weight,
            orig_size = [orig_size[1],orig_size[0]],
        )

        for ii,img in enumerate(images):
            w,h = img.size
            if w != orig_size[0] or h != orig_size[1]:
                ratio_w = w/width * orig_size[0]
                ratio_h = h/height * orig_size[1]
                # crop the padding
                img = crop_image_to_original(img, (ratio_w,ratio_h))

            if args.save_img and ii < len(images)-1:
                img.save(f"{args.save_root}/{name_}_{ii}_out.png")

            if ii == len(images)-1:
                img.save(f"{args.save_root}/{name_}_finalout.png")

        end = time.time()
        print('time for running is : %s Seconds' % (end - start))


def main5_outpainting():
    """
    ultraDiffEdit main script
    """
    from Util.img_pad_crop import pad_image_to_multiple_num,crop_image_to_original,pad_image_to_square
    from Util.caption_read import read_lines_from_file
    from diffusers.utils import load_image
    import argparse
    from pipeline_ultradiffedit_sdxl import StableAnysizeInpaintPipeline
    import math
    import time
    import torch
    import os
    import Util.utils_train  as ut
    from PIL import ImageOps

    
    parser = argparse.ArgumentParser(description="UltraDiffEdit inference")
    parser.add_argument("--file_root", default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/image", type=str, help="path to the input images")
    parser.add_argument("--mask_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/mask_dilate", type=str, help="path to the masks")
    parser.add_argument("--caption_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/caption2", type=str, help="path to the captions")
    
    parser.add_argument("--start_idx", type=int, default=2, help="the indx of the image to start")
    parser.add_argument("--end_idx", type=int, default=999999, help="the index of the image to end")

    parser.add_argument("--beta1", type=float, default=3, help="the beta1 weight")
    parser.add_argument("--beta2", type=float, default=1, help="the beta2 weight")
    parser.add_argument("--ug_weight", type=float, default=0.2, help="the ug_weight weight")

    parser.add_argument("--view_batch_size", type=int, default=16, help="the batch size to use")
    parser.add_argument('--ckpt', type=str, default=None ,help='the checkpoint path pf the model')
    parser.add_argument('--save_img', type=bool, default=False ,help='if or not save the intermediate images')
    parser.add_argument('--save_root', type=str, default='./results',help='root for saving the images')
    parser.add_argument('--stages', type=str, default='two',help='[two, three, S] stages setting for inference')
    parser.add_argument("--stride", type=int, default=64, help="the stride to use for the inference")

    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    if args.ckpt ==None:
        args.ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("model load from  ",args.ckpt)

    pipe = StableAnysizeInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path= args.ckpt,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")

    img_list = []
    mask_list = []
    caption_list = []
    ut.listdir(args.file_root, img_list)
    ut.listdir(args.mask_root, mask_list)
    ut.listdir(args.caption_root, caption_list)

    import random 
    generator = torch.Generator(device='cuda')  # random seed generator
    # seed_ = random.randint(1,100)
    # generator = generator.manual_seed(seed_)
    generator = generator.manual_seed(5)

    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
    for ii, (img_path,mask_path,caption_path) in enumerate(zip(img_list,mask_list,caption_list)):
        name_ = os.path.basename(img_path)
        name_ = str(name_).strip().split(".")[0]
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

        min_padding = math.lcm(max_scale*8,1024)
        # Pad the image
        init_image, orig_size = pad_image_to_multiple_num(init_image,min_padding,color=(255,255,255))
        mask_image, orig_size = pad_image_to_multiple_num(mask_image,min_padding,color=(0,0,0))

        # Reverse the mask: invert black/white
        reverse_mask_image = ImageOps.invert(mask_image.convert("L")).convert("RGB")
        reverse_mask_image.save(f"{args.save_root}/{name_}_reversemask.png")

        width, height = init_image.size
        print("the padded image size is: ", width, height)

        start = time.time()
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image, mask_image=reverse_mask_image, num_inference_steps=50, strength=0.80,
            generator = generator,
            tar_height=height,
            tar_width=width,
            view_batch_size=args.view_batch_size,
            stride=args.stride,
            beta_scale_1=args.beta1, beta_scale_2=args.beta2, 
            cosine_scale_3=1, sigma=0.8,
            multi_decoder=True, show_image=False,
            save_image_tag = False,
            file_name=name_,
            save_root=args.save_root,
            run_stage = args.stages,
            ug_weight = args.ug_weight,
            orig_size = [orig_size[1],orig_size[0]]
        )

        for ii,img in enumerate(images):
            w,h = img.size
            if w != orig_size[0] or h != orig_size[1]:
                ratio_w = w/width * orig_size[0]
                ratio_h = h/height * orig_size[1]
                # crop the padding
                img = crop_image_to_original(img, (ratio_w,ratio_h))

            if args.save_img and ii < len(images)-1:
                img.save(f"{args.save_root}/{name_}_{ii}_out.png")

            if ii == len(images)-1:
                img.save(f"{args.save_root}/{name_}_finalout.png")

        end = time.time()
        print('time for running is : %s Seconds' % (end - start))



def main_showNoiseMap():
    """
    ultraDiffEdit main script
    """
    from Util.img_pad_crop import pad_image_to_multiple_num,crop_image_to_original,pad_image_to_square
    from Util.caption_read import read_lines_from_file
    from diffusers.utils import load_image
    import argparse
    from pipeline_ultradiffedit_sdxl import StableAnysizeInpaintPipeline

    import math
    import time
    import torch
    import os
    import Util.utils_train  as ut
    
    parser = argparse.ArgumentParser(description="UltraDiffEdit inference")
    # parser.add_argument("--file_root", default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/image", type=str, help="path to the input images")
    # parser.add_argument("--mask_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/mask_dilate", type=str, help="path to the masks")
    # parser.add_argument("--caption_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/caption2", type=str, help="path to the captions")
    
    parser.add_argument("--file_root", default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/Sys2K/img", type=str, help="path to the input images")
    parser.add_argument("--mask_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/Sys2K/mask_dilate", type=str, help="path to the masks")
    parser.add_argument("--caption_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/Sys2K/text_prompt", type=str, help="path to the captions")
    
    parser.add_argument("--start_idx", type=int, default=42, help="the indx of the image to start")
    parser.add_argument("--end_idx", type=int, default=999999, help="the index of the image to end")

    parser.add_argument("--beta1", type=float, default=3, help="the beta1 weight")
    parser.add_argument("--beta2", type=float, default=1, help="the beta2 weight")
    parser.add_argument("--ug_weight", type=float, default=0.2, help="the ug_weight weight")

    parser.add_argument("--view_batch_size", type=int, default=16, help="the batch size to use")
    parser.add_argument('--ckpt', type=str, default=None ,help='the checkpoint path pf the model')
    parser.add_argument('--save_img', type=bool, default=False ,help='if or not save the intermediate images')
    parser.add_argument('--save_root', type=str, default='./results',help='root for saving the images')
    parser.add_argument('--stages', type=str, default='two',help='[two, three, S] stages setting for inference')
    parser.add_argument("--stride", type=int, default=64, help="the stride to use for the inference")

    args = parser.parse_args()
    
    os.makedirs(args.save_root, exist_ok=True)

    if args.ckpt ==None:
        args.ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("model load from  ",args.ckpt)

    pipe = StableAnysizeInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path= args.ckpt,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")

    img_list = []
    mask_list = []
    caption_list = []
    ut.listdir(args.file_root, img_list)
    ut.listdir(args.mask_root, mask_list)
    ut.listdir(args.caption_root, caption_list)

    import random 
    generator = torch.Generator(device='cuda')  # random seed generator
    # seed_ = random.randint(1,100)
    # generator = generator.manual_seed(seed_)
    generator = generator.manual_seed(5)

    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
    for ii, (img_path,mask_path,caption_path) in enumerate(zip(img_list,mask_list,caption_list)):
        name_ = os.path.basename(img_path)
        name_ = str(name_).strip().split(".")[0]
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

        min_padding = math.lcm(max_scale*8,1024)
        # Pad the image
        init_image, orig_size = pad_image_to_multiple_num(init_image,min_padding,color=(255,255,255))
        mask_image, orig_size = pad_image_to_multiple_num(mask_image,min_padding,color=(0,0,0))

        width, height = init_image.size
        print("the padded image size is: ", width, height)

        start = time.time()
        images,out_vis_analysisList = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80,
            generator = generator,
            tar_height=height,
            tar_width=width,
            view_batch_size=args.view_batch_size,
            stride=args.stride,
            beta_scale_1=args.beta1, beta_scale_2=args.beta2, 
            cosine_scale_3=1, sigma=0.8,
            multi_decoder=True, show_image=False,
            save_image_tag = False,
            file_name=name_,
            save_root=args.save_root,
            run_stage = args.stages,
            ug_weight = args.ug_weight,
            orig_size = [orig_size[1],orig_size[0]],
            vis_analysis = True
        )

        for ii,img in enumerate(images):
            w,h = img.size
            if w != orig_size[0] or h != orig_size[1]:
                ratio_w = w/width * orig_size[0]
                ratio_h = h/height * orig_size[1]
                # crop the padding
                img = crop_image_to_original(img, (ratio_w,ratio_h))

            if args.save_img and ii < len(images)-1:
                img.save(f"{args.save_root}/{name_}_{ii}_out.png")

            if ii == len(images)-1:
                img.save(f"{args.save_root}/{name_}_finalout.png")

        end = time.time()
        print('time for running is : %s Seconds' % (end - start))

        for (img,ii) in out_vis_analysisList:
            if isinstance(ii, torch.Tensor):
                ii = int(ii.item())
            elif isinstance(ii, (torch.IntTensor, torch.LongTensor)):
                ii = int(ii.cpu().item())
            elif hasattr(ii, "item") and callable(ii.item):
                ii = int(ii.item())
            w,h = img.size
            if w != orig_size[0] or h != orig_size[1]:
                ratio_w = w/width * orig_size[0]
                ratio_h = h/height * orig_size[1]
                # crop the padding
                img = crop_image_to_original(img, (ratio_w,ratio_h))

            # if args.save_img and ii < len(images)-1:
            img.save(f"{args.save_root}/{name_}_denoise_step_{ii}_out.png")



        


def main5_outpainting():
    """
    ultraDiffEdit main script
    """
    from Util.img_pad_crop import pad_image_to_multiple_num,crop_image_to_original,pad_image_to_square
    from Util.caption_read import read_lines_from_file
    from diffusers.utils import load_image
    import argparse
    from pipeline_ultradiffedit_sdxl import StableAnysizeInpaintPipeline
    import math
    import time
    import torch
    import os
    import Util.utils_train  as ut
    from PIL import ImageOps

    
    parser = argparse.ArgumentParser(description="UltraDiffEdit inference")
    parser.add_argument("--file_root", default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/image", type=str, help="path to the input images")
    parser.add_argument("--mask_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/mask_dilate", type=str, help="path to the masks")
    parser.add_argument("--caption_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/caption2", type=str, help="path to the captions")
    
    parser.add_argument("--start_idx", type=int, default=2, help="the indx of the image to start")
    parser.add_argument("--end_idx", type=int, default=999999, help="the index of the image to end")

    parser.add_argument("--beta1", type=float, default=3, help="the beta1 weight")
    parser.add_argument("--beta2", type=float, default=1, help="the beta2 weight")
    parser.add_argument("--ug_weight", type=float, default=0.2, help="the ug_weight weight")

    parser.add_argument("--view_batch_size", type=int, default=16, help="the batch size to use")
    parser.add_argument('--ckpt', type=str, default=None ,help='the checkpoint path pf the model')
    parser.add_argument('--save_img', type=bool, default=False ,help='if or not save the intermediate images')
    parser.add_argument('--save_root', type=str, default='./results',help='root for saving the images')
    parser.add_argument('--stages', type=str, default='two',help='[two, three, S] stages setting for inference')
    parser.add_argument("--stride", type=int, default=64, help="the stride to use for the inference")

    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    if args.ckpt ==None:
        args.ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("model load from  ",args.ckpt)

    pipe = StableAnysizeInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path= args.ckpt,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")

    img_list = []
    mask_list = []
    caption_list = []
    ut.listdir(args.file_root, img_list)
    ut.listdir(args.mask_root, mask_list)
    ut.listdir(args.caption_root, caption_list)

    import random 
    generator = torch.Generator(device='cuda')  # random seed generator
    # seed_ = random.randint(1,100)
    # generator = generator.manual_seed(seed_)
    generator = generator.manual_seed(5)

    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
    for ii, (img_path,mask_path,caption_path) in enumerate(zip(img_list,mask_list,caption_list)):
        name_ = os.path.basename(img_path)
        name_ = str(name_).strip().split(".")[0]
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

        min_padding = math.lcm(max_scale*8,1024)
        # Pad the image
        init_image, orig_size = pad_image_to_multiple_num(init_image,min_padding,color=(255,255,255))
        mask_image, orig_size = pad_image_to_multiple_num(mask_image,min_padding,color=(0,0,0))

        # Reverse the mask: invert black/white
        reverse_mask_image = ImageOps.invert(mask_image.convert("L")).convert("RGB")
        reverse_mask_image.save(f"{args.save_root}/{name_}_reversemask.png")

        width, height = init_image.size
        print("the padded image size is: ", width, height)

        start = time.time()
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image, mask_image=reverse_mask_image, num_inference_steps=50, strength=0.80,
            generator = generator,
            tar_height=height,
            tar_width=width,
            view_batch_size=args.view_batch_size,
            stride=args.stride,
            beta_scale_1=args.beta1, beta_scale_2=args.beta2, 
            cosine_scale_3=1, sigma=0.8,
            multi_decoder=True, show_image=False,
            save_image_tag = False,
            file_name=name_,
            save_root=args.save_root,
            run_stage = args.stages,
            ug_weight = args.ug_weight,
            orig_size = [orig_size[1],orig_size[0]]
        )

        for ii,img in enumerate(images):
            w,h = img.size
            if w != orig_size[0] or h != orig_size[1]:
                ratio_w = w/width * orig_size[0]
                ratio_h = h/height * orig_size[1]
                # crop the padding
                img = crop_image_to_original(img, (ratio_w,ratio_h))

            if args.save_img and ii < len(images)-1:
                img.save(f"{args.save_root}/{name_}_{ii}_out.png")

            if ii == len(images)-1:
                img.save(f"{args.save_root}/{name_}_finalout.png")

        end = time.time()
        print('time for running is : %s Seconds' % (end - start))



def main4_reconstraction():
    """
    ultraDiffEdit main script for reconstraction
    """
    from Util.img_pad_crop import pad_image_to_multiple_num,crop_image_to_original,pad_image_to_square
    from Util.caption_read import read_lines_from_file
    from diffusers.utils import load_image
    import argparse
    from pipeline_ultradiffedit_sdxl import StableAnysizeInpaintPipeline
    import math
    import time
    import torch
    import os
    import Util.utils_train  as ut
    
    parser = argparse.ArgumentParser(description="UltraDiffEdit inference")
    parser.add_argument("--file_root", default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/image", type=str, help="path to the input images")
    parser.add_argument("--mask_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/mask_dilate", type=str, help="path to the masks")
    parser.add_argument("--caption_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/caption2", type=str, help="path to the captions")
    
    parser.add_argument("--start_idx", type=int, default=2, help="the indx of the image to start")
    parser.add_argument("--end_idx", type=int, default=999999, help="the index of the image to end")

    parser.add_argument("--beta1", type=float, default=3, help="the beta1 weight")
    parser.add_argument("--beta2", type=float, default=1, help="the beta2 weight")
    parser.add_argument("--ug_weight", type=float, default=0.2, help="the ug_weight weight")

    parser.add_argument("--view_batch_size", type=int, default=16, help="the batch size to use")
    parser.add_argument('--ckpt', type=str, default=None ,help='the checkpoint path pf the model')
    parser.add_argument('--save_img', type=bool, default=False ,help='if or not save the intermediate images')
    parser.add_argument('--save_root', type=str, default='./results',help='root for saving the images')
    parser.add_argument('--stages', type=str, default='two',help='[two, three, S] stages setting for inference')
    parser.add_argument("--stride", type=int, default=64, help="the stride to use for the inference")

    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    if args.ckpt ==None:
        args.ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("model load from  ",args.ckpt)

    pipe = StableAnysizeInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path= args.ckpt,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")

    img_list = []
    mask_list = []
    caption_list = []
    ut.listdir(args.file_root, img_list)
    ut.listdir(args.mask_root, mask_list)
    ut.listdir(args.caption_root, caption_list)

    import random 
    generator = torch.Generator(device='cuda')  # random seed generator
    # seed_ = random.randint(1,100)
    # generator = generator.manual_seed(seed_)
    generator = generator.manual_seed(5)

    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
    for ii, (img_path,mask_path,caption_path) in enumerate(zip(img_list,mask_list,caption_list)):
        name_ = os.path.basename(img_path)
        name_ = str(name_).strip().split(".")[0]
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

        min_padding = math.lcm(max_scale*8,1024)
        # Pad the image
        init_image, orig_size = pad_image_to_multiple_num(init_image,min_padding,color=(255,255,255))
        mask_image, orig_size = pad_image_to_multiple_num(mask_image,min_padding,color=(0,0,0))

        width, height = init_image.size
        print("the padded image size is: ", width, height)

        start = time.time()
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80,
            generator = generator,
            tar_height=height,
            tar_width=width,
            view_batch_size=args.view_batch_size,
            stride=args.stride,
            beta_scale_1=args.beta1, beta_scale_2=args.beta2, 
            cosine_scale_3=1, sigma=0.8,
            multi_decoder=True, show_image=False,
            save_image_tag = False,
            file_name=name_,
            save_root=args.save_root,
            run_stage = args.stages,
            ug_weight = args.ug_weight,
            orig_size = [orig_size[1],orig_size[0]]
        )

        for ii,img in enumerate(images):
            w,h = img.size
            if w != orig_size[0] or h != orig_size[1]:
                ratio_w = w/width * orig_size[0]
                ratio_h = h/height * orig_size[1]
                # crop the padding
                img = crop_image_to_original(img, (ratio_w,ratio_h))

            if args.save_img and ii < len(images)-1:
                img.save(f"{args.save_root}/{name_}_{ii}_out.png")

            if ii == len(images)-1:
                img.save(f"{args.save_root}/{name_}_finalout.png")

        end = time.time()
        print('time for running is : %s Seconds' % (end - start))


def main3_diverse_seeds():
    """
    based on the main function, it is with diverse seeds 
    """
    from Util.img_pad_crop import pad_image_to_multiple_num, crop_image_to_original, pad_image_to_square
    from Util.caption_read import read_lines_from_file
    from diffusers.utils import load_image
    import argparse
    from pipeline_ultradiffedit_sdxl import StableAnysizeInpaintPipeline  #
    import math
    import time
    import torch
    import os
    import Util.utils_train as ut

    parser = argparse.ArgumentParser(description="UltraDiffEdit inference")
    parser.add_argument("--file_root",
                        default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/image",
                        type=str, help="path to the input images")
    parser.add_argument("--mask_root",
                        default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/mask_dilate",
                        type=str, help="path to the masks")
    parser.add_argument("--caption_root",
                        default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/DemoFusion_data/UHRSD_TE_edit/caption2",
                        type=str, help="path to the captions")

    parser.add_argument("--start_idx", type=int, default=2, help="the indx of the image to start")
    parser.add_argument("--end_idx", type=int, default=3, help="the index of the image to end")

    parser.add_argument("--beta1", type=float, default=3, help="the beta1 weight")
    parser.add_argument("--beta2", type=float, default=1, help="the beta2 weight")
    parser.add_argument("--ug_weight", type=float, default=0.2, help="the ug_weight weight")

    parser.add_argument("--view_batch_size", type=int, default=16, help="the batch size to use")
    parser.add_argument('--ckpt', type=str, default=None, help='the checkpoint path pf the model')
    parser.add_argument('--save_img', type=bool, default=False, help='if or not save the intermediate images')
    parser.add_argument('--save_root', type=str, default='./results', help='root for saving the images')
    parser.add_argument('--stages', type=str, default='two', help='[two, three, S] stages setting for inference')
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    if args.ckpt == None:
        args.ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("model load from  ", args.ckpt)

    pipe = StableAnysizeInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path=args.ckpt,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")

    img_list = []
    mask_list = []
    caption_list = []
    ut.listdir(args.file_root, img_list)
    ut.listdir(args.mask_root, mask_list)
    ut.listdir(args.caption_root, caption_list)


    while True:

        import random
        generator = torch.Generator(device='cuda')  # random seed generator
        seed_ = random.randint(1,10000)
        generator = generator.manual_seed(seed_)
        # generator = generator.manual_seed(5)

        negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
        for ii, (img_path, mask_path, caption_path) in enumerate(zip(img_list, mask_list, caption_list)):
            name_ = os.path.basename(img_path)
            name_ = str(name_).strip().split(".")[0]
            if ii < args.start_idx or ii >= args.end_idx: continue

            print("img_path:", img_path)
            print("mask_path:", mask_path)
            print("caption_path:", caption_path)

            init_image = load_image(img_path).convert("RGB")
            init_image.save(f"{args.save_root}/{name_}_oriinput.png")

            width, height = init_image.size
            print("the original image size is: ", width, height)
            mask_image = load_image(mask_path).convert("RGB").resize((width, height))
            mask_image.save(f"{args.save_root}/{name_}_orimask.png")

            prompt = read_lines_from_file(caption_path)[0]
            print("the prompt is: ", prompt)

            max_size = max(width, height)
            max_scale = math.ceil(float(max_size) / 1024)

            min_padding = math.lcm(max_scale * 8, 1024)
            # Pad the image
            init_image, orig_size = pad_image_to_multiple_num(init_image, min_padding, color=(255, 255, 255))
            mask_image, orig_size = pad_image_to_multiple_num(mask_image, min_padding, color=(0, 0, 0))

            width, height = init_image.size
            print("the padded image size is: ", width, height)

            start = time.time()
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80,
                generator=generator,
                tar_height=height,
                tar_width=width,
                view_batch_size=args.view_batch_size,
                stride=64,
                beta_scale_1=args.beta1, beta_scale_2=args.beta2,
                cosine_scale_3=1, sigma=0.8,
                multi_decoder=True, show_image=False,
                save_image_tag=False,
                file_name=name_,
                save_root=args.save_root,
                run_stage=args.stages,
                ug_weight=args.ug_weight,
                orig_size=[orig_size[1], orig_size[0]]
            )

            for ii, img in enumerate(images):
                w, h = img.size
                if w != orig_size[0] or h != orig_size[1]:
                    ratio_w = w / width * orig_size[0]
                    ratio_h = h / height * orig_size[1]
                    # crop the padding
                    img = crop_image_to_original(img, (ratio_w, ratio_h))

                if args.save_img and ii < len(images) - 1:
                    img.save(f"{args.save_root}/{name_}_{ii}_{seed_}_out.png")

                if ii == len(images) - 1:
                    img.save(f"{args.save_root}/{name_}_{seed_}_finalout.png")

            end = time.time()
            print('time for running is : %s Seconds' % (end - start))


def generate_multi_object_prompt(chosen):
    import random
    templates = [
        "A scene featuring " + ", ".join(chosen[:-1]) + ", and " + chosen[-1],
        "An illustration of " + ", ".join(chosen[:-1]) + ", and " + chosen[-1],
        "A photo containing " + ", ".join(chosen[:-1]) + ", and " + chosen[-1],
        "A detailed rendering of " + ", ".join(chosen[:-1]) + ", and " + chosen[-1],
        "An artistic composition of " + ", ".join(chosen[:-1]) + ", and " + chosen[-1],
        "An outdoor scene with " + ", ".join(chosen[:-1]) + ", and " + chosen[-1],
        "A realistic depiction of " + ", ".join(chosen[:-1]) + ", and " + chosen[-1],
        "A fantasy world including " + ", ".join(chosen[:-1]) + ", and " + chosen[-1],
        "A still life arrangement of " + ", ".join(chosen[:-1]) + ", and " + chosen[-1],
        "A surreal image with " + ", ".join(chosen[:-1]) + ", and " + chosen[-1]
    ]
    return random.choice(templates)

def prompt_generation():
    # Re-import necessary modules after code execution state reset
    import random
    import pandas as pd

    # Define object list
    objects = [
        "rabbit", "cat", "dog", "fox", "deer", "lion", "zebra", "giraffe", "elephant", "bear",
        "penguin", "dolphin", "shark", "whale", "octopus", "horse", "cow", "sheep", "goat", "duck",
        "chicken", "mouse", "squirrel", "bat", "kangaroo", "panda", "raccoon", "wolf", "tiger", "leopard",
        "robot", "drone", "laptop", "camera", "bicycle", "motorcycle", "car", "bus", "truck", "train",
        "balloon", "spaceship", "satellite", "telescope", "spacesuit", "sword", "shield", "treasure chest", "tent", "campfire",
        "scuba diver", "astronaut", "wizard", "chef", "pirate", "knight", "witch", "goblin", "alien", "monster",
        "painting", "canvas", "mug", "book", "guitar", "piano", "microphone", "lantern", "bench", "bridge",
        "vending machine", "streetlight", "castle", "cabin", "house", "treehouse", "tree", "mushroom", "flower", "rock",
        "river", "lake", "cloud", "mountain", "volcano", "island", "beach", "desert", "forest", "jungle",
        "kite", "toy", "snowman", "snowmobile", "skateboard", "surfboard", "camera drone", "binoculars", "telescope", "flashlight"
    ]

    # Generate 100 multi-object prompts
    prompts = []
    for _ in range(100):
        chosen = random.sample(objects, k=random.randint(2, 3))
        prompt = generate_multi_object_prompt(chosen)
        # prompt = "A scene featuring " + ", ".join(chosen[:-1]) + ", and " + chosen[-1]
        prompts.append(prompt)

    return prompts
    # Convert to DataFrame
    # df_prompts = pd.DataFrame(prompts, columns=["Multi-Object Prompt"])

    # # Display DataFrame to user
    # import ace_tools as tools; tools.display_dataframe_to_user(name="100 Multi-Object Prompts", dataframe=df_prompts)


def main2_prompt_editing():
    """
    based on the main function, this function is used to edit the prompt

    """
    from Util.img_pad_crop import pad_image_to_multiple_num,crop_image_to_original,pad_image_to_square
    from Util.caption_read import read_lines_from_file
    from diffusers.utils import load_image
    import argparse
    from pipeline_ultradiffedit_sdxl import StableAnysizeInpaintPipeline

    import math
    import time
    import torch
    import os
    import Util.utils_train  as ut
    import random
    
    parser = argparse.ArgumentParser(description="UltraDiffEdit inference")
    parser.add_argument("--file_root", default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/Sys2K/img", type=str, help="path to the input images")
    parser.add_argument("--mask_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/Sys2K/mask_dilate", type=str, help="path to the masks")
    parser.add_argument("--caption_root",default="/media/onelong/Longlongaaago_memo/Longlongaaago3/work_data/Sys2K/text_prompt", type=str, help="path to the captions")
    
    parser.add_argument("--start_idx", type=int, default=822, help="the indx of the image to start")
    parser.add_argument("--end_idx", type=int, default=999999, help="the index of the image to end")

    parser.add_argument("--beta1", type=float, default=3, help="the beta1 weight")
    parser.add_argument("--beta2", type=float, default=1, help="the beta2 weight")
    parser.add_argument("--ug_weight", type=float, default=0.2, help="the ug_weight weight")

    parser.add_argument("--view_batch_size", type=int, default=16, help="the batch size to use")
    parser.add_argument('--ckpt', type=str, default=None ,help='the checkpoint path pf the model')
    parser.add_argument('--save_img', type=bool, default=False ,help='if or not save the intermediate images')
    parser.add_argument('--save_root', type=str, default='./results',help='root for saving the images')
    parser.add_argument('--stages', type=str, default='two',help='[two, three, S] stages setting for inference')
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    if args.ckpt ==None:
        args.ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("model load from  ",args.ckpt)

    pipe = StableAnysizeInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path= args.ckpt,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")

    img_list = []
    mask_list = []
    caption_list = []
    ut.listdir(args.file_root, img_list)
    ut.listdir(args.mask_root, mask_list)
    ut.listdir(args.caption_root, caption_list)

    generator = torch.Generator(device='cuda')  # random seed generator
    seed_ = random.randint(1,10000)
    generator = generator.manual_seed(seed_)
    # generator = generator.manual_seed(5)

    my_caption_list = prompt_generation()  # generate multi-object prompts
    # Randomly sample 5 captions from the generated list
    my_caption_list = random.sample(my_caption_list, 5)
    # my_caption_list = [
    #     "A futuristic glass dome in the woods",
    #     "A cozy log cabin with smoke coming out of the chimney in the woods",
    #     "An abandoned medieval castle in the woods",
    #     "A giant mushroom house in the woods",
    #     "A spaceship landing site in the woods",
    #     "A traditional Japanese tea house in the woods",
    #     "A mystical glowing crystal in the woods",
    #     "A towering wooden watchtower in the woods",
    #     "A rustic treehouse nestled high in the woods",
    #     "A bustling elf village in the woods",
    #     "A stone circle with glowing runes in the woods",
    #     "A modern minimalist cube-shaped house in the woods",
    #     "A colorful hot air balloon tied down in the woods",
    #     "A mysterious glowing portal in the woods",
    #     "A superhero flying through the woods",
    #     "A small tent with a campfire in the woods",
    #     "A Viking longhouse with a thatched roof in the woods",
    #     "A giant clock tower standing alone in the woods",
    #     "A fantasy dragon’s nest in the woods",
    #     "A ruined ancient temple reclaimed by nature in the woods",
    #     "A massive library hidden in the woods",
    #     "A magical fairy circle with glowing lights in the woods",
    #     "A robotic outpost blending into the woods",
    #     "A tropical bamboo hut surrounded by greenery in the woods",
    #     "A luxurious mansion with a pool in the woods",
    #     "A glowing alien artifact hovering in the woods",
    #     "A quaint bakery with a wooden sign in the woods",
    #     "A Victorian greenhouse filled with exotic plants in the woods",
    #     "A massive bear cave with glowing eyes in the woods",
    #     "A whimsical candy house inspired by a fairytale in the woods",
    #     "A charming thatched cottage in the woods",
    #     "A secluded wooden cabin in the woods",
    #     "A small stone house with ivy growing on it in the woods",
    #     "A cozy A-frame cabin in the woods",
    #     "A picturesque chalet with a balcony in the woods",
    #     "A log house surrounded by colorful flowers in the woods",
    #     "A whimsical gingerbread house in the woods",
    #     "An old farmhouse with broken windows in the woods",
    #     "A tiny hobbit hole tucked into a hill in the woods",
    #     "A wooden windmill spinning gently in the woods",
    #     "A rustic barn covered in moss in the woods",
    #     "An abandoned hunting lodge in the woods",
    #     "A secret bunker hidden in the woods",
    #     "A charming brick cottage with a chimney in the woods",
    #     "A small chapel with stained glass windows in the woods",
    #     "A colorful gypsy wagon parked in the woods",
    #     "A snow-covered winter cabin in the woods",
    #     "A luxurious treehouse overlooking the woods",
    #     "A fairytale castle with towers peeking through the woods",
    #     "A miner’s hut with tools scattered around in the woods",
    #     "A spooky, rundown shack in the woods",
    #     "A classic hunting cabin with mounted antlers in the woods",
    #     "A modern tiny house with large glass windows in the woods",
    #     "A mysterious cave entrance with torches in the woods",
    #     "A rustic boathouse near a pond in the woods",
    #     "A small chapel made of wood and stone in the woods",
    #     "A mountain lodge with a large balcony in the woods",
    #     "A hidden artist’s studio surrounded by easels in the woods",
    #     "A charming Victorian-style gazebo in the woods",
    #     "An enchanted witch’s hut with a crooked roof in the woods"
    # ]

    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
    for ii, (img_path,mask_path,_) in enumerate(zip(img_list,mask_list,caption_list)):
        
        name_ = os.path.basename(img_path)
        name_ = str(name_).strip().split(".")[0]
        if ii <args.start_idx or ii>= args.end_idx : continue

        print("img_path:",img_path)
        print("mask_path:",mask_path)
        # print("caption_path:",caption_path)

        # prompt = read_lines_from_file(caption_path)[0]

        my_caption_list = my_caption_list[::-1]
        for prompt in my_caption_list:
            #
            init_image = load_image(img_path).convert("RGB")
            init_image.save(f"{args.save_root}/{name_}_oriinput.png")

            width, height = init_image.size
            print("the original image size is: ", width, height)
            mask_image = load_image(mask_path).convert("RGB").resize((width, height))
            mask_image.save(f"{args.save_root}/{name_}_orimask.png")
            print("the prompt is: ", prompt)

            max_size = max(width, height)
            max_scale = math.ceil(float(max_size)/1024)

            min_padding = math.lcm(max_scale*8,1024)
            # Pad the image
            init_image, orig_size = pad_image_to_multiple_num(init_image,min_padding,color=(255,255,255))
            mask_image, orig_size = pad_image_to_multiple_num(mask_image,min_padding,color=(0,0,0))

            width, height = init_image.size
            print("the padded image size is: ", width, height)

            start = time.time()
            print(f"start time:_{str(start)}")
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80,
                generator = generator,
                tar_height=height,
                tar_width=width,
                view_batch_size=args.view_batch_size,
                stride=64,
                beta_scale_1=args.beta1, beta_scale_2=args.beta2, 
                cosine_scale_3=1, sigma=0.8,
                multi_decoder=True, show_image=False,
                save_image_tag = False,
                file_name=name_,
                save_root=args.save_root,
                run_stage = args.stages,
                ug_weight = args.ug_weight,
                orig_size = [orig_size[1],orig_size[0]]
            )

            for ii,img in enumerate(images):
                w,h = img.size
                if w != orig_size[0] or h != orig_size[1]:
                    ratio_w = w/width * orig_size[0]
                    ratio_h = h/height * orig_size[1]
                    # crop the padding
                    img = crop_image_to_original(img, (ratio_w,ratio_h))

                if args.save_img and ii < len(images)-1:
                    img.save(f"{args.save_root}/{name_}_{str(start)}_{ii}_out.png")

                if ii == len(images)-1:
                    img.save(f"{args.save_root}/{name_}_{str(start)}_{prompt}_finalout.png")

            end = time.time()
            print('time for running is : %s Seconds' % (end - start))

if __name__ == '__main__':

    main()
    # main_showNoiseMap()
    # main5_outpainting()
    # main4_reconstraction()
    # main2_prompt_editing()
    # main3_diverse_seeds()

    
