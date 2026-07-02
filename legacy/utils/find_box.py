


import torch

def find_bounding_box(mask):
    """
    Find the bounding box of white pixels (value 1) in a binary mask.

    Args:
    mask (torch.Tensor): A binary mask tensor of shape [1, 1, H, W].

    Returns:
    tuple: (xmin, ymin, xmax, ymax) coordinates of the bounding box.
    """
    # 确保mask是二维的
    if mask.dim() != 4 or mask.shape[0] != 1 or mask.shape[1] != 1:
        raise ValueError("Mask must be of shape [1, 1, H, W]")

    # 去掉前两个维度，因为它们都是1
    mask = mask.squeeze()

    # 检查是否存在1
    if not torch.any(mask == 1):
        return None  # 如果没有1，返回None

    # 获取含有1的行和列
    rows = torch.any(mask == 1, dim=1)
    cols = torch.any(mask == 1, dim=0)

    # 确定最小和最大的行列索引
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]

    return  ymin.item(), xmin.item(),  ymax.item(), xmax.item()



if __name__ == '__main__':


    # 示例用法
    # 创建一个示例mask
    h, w = 10, 15
    mask = torch.zeros((1, 1, h, w), dtype=torch.float32)
    mask[0, 0, 2:5, 3:8] = 1  # 创建一个随机的白色矩形

    # 调用函数
    bbox = find_bounding_box(mask)
    print("Bounding Box:", bbox)
