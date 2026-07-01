

import math
from time import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import functional as TF

import pietorch



if __name__ == '__main__':
    img_1_png = Image.open('example_images/mug.png')
    img_1 = Image.new("RGB", img_1_png.size, (255, 255, 255))
    img_1.paste(img_1_png, mask=img_1_png.getchannel('A'))

    img_2 = Image.open('example_images/brick_texture.jpg')

    # fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    # ax[0].imshow(img_1)
    # ax[1].imshow(img_2)
    target = TF.to_tensor(img_1)

    source = TF.resize(TF.to_tensor(img_2), [512, 512])

    corner_coord = torch.tensor([144, 100])

    # mask = torch.zeros(source.shape[1:])
    # for i in range(mask.shape[0]):
    #     for j in range(mask.shape[1]):
    #         if ((i - 256) ** 2 + (j - 256) ** 2) ** 0.5 < 200:
    #             mask[i, j] = 1
    mask = torch.ones(source.shape[1:])

    # fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    # ax[0].imshow(torch.movedim(target, 0, -1))
    # ax[1].imshow(torch.movedim(source, 0, -1))
    # ax[2].imshow(mask)
    # plt.axis('off')  # Turn off axis numbers and ticks
    # plt.show()


    start = time()
    res = pietorch.blend(target, source, torch.zeros(source.shape[1:]), corner_coord, True, channels_dim=0)
    print('Took ', time() - start)

    # plt.imshow(torch.movedim(res, 0, -1)) 
    # # plt.imshow(res)

    # plt.axis('off')  # Turn off axis numbers and ticks
    # plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(torch.movedim(res, 0, -1))
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

    # recon_diff = torch.abs(res - target)
    # print(torch.min(recon_diff), '-', torch.mean(recon_diff), '/', torch.median(recon_diff), '-', torch.max(recon_diff))
    #
    # plt.imshow(torch.movedim(recon_diff, 0, -1))
    # # plt.show()
    #
    # start = time()
    # res = pietorch.blend(target, source, mask, corner_coord, True, channels_dim=0)
    # print('Took ', time() - start)
    #
    # plt.figure(figsize=(10, 8))
    # plt.imshow(torch.movedim(res, 0, -1))
    # # plt.show()
    #
    # diff = torch.abs(res - target)
    # print(torch.max(diff))
    # plt.imshow(torch.movedim(diff, 0, -1))
    #
    # start = time()
    # res_wide = pietorch.blend_wide(target, source, mask, corner_coord, True, channels_dim=0)
    # print('Took ', time() - start)
    #
    # plt.figure(figsize=(10, 8))
    # plt.imshow(torch.movedim(res_wide, 0, -1))
    #
    # diff_wide = torch.abs(res_wide - target)
    # print(torch.max(diff_wide))
    # plt.imshow(torch.movedim(diff_wide, 0, -1))
    # plt.show()
    #
    # from pietorch import CachedPoissonBlender
    #
    # # Add green function to cache ahead of time
    # cached_blender = CachedPoissonBlender([(source.shape, 0)])
    #
    # start = time()
    # res = cached_blender.blend(target, source, mask, corner_coord, True, channels_dim=0)
    # print('Took ', time() - start)
    # plt.show()
