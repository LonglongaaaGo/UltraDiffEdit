
from PIL import Image
import math

def read_lines_from_file(file_path):
    """
    Read all lines from a text file and return them as a list of strings.

    Args:
    file_path (str): The path to the text file.

    Returns:
    list: A list of strings, each representing a line from the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 读取所有行到一个列表
            lines = file.readlines()
            # 去掉每行末尾的换行符
            lines = [line.strip() for line in lines]
        return lines
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []



def pad_image_to_square(image,color=(255, 255, 255)):
    """
    Pad the image so that its dimensions are multiples of 8.

    Args:
    image (PIL.Image): The input image.

    Returns:
    PIL.Image: The padded image.
    tuple: Original dimensions of the image.
    """
    width,height = image.size
    max_size = max(width,height)

    # Create a new image with white background
    new_image = Image.new('RGB', (max_size, max_size), color=color)
    new_image.paste(image, (0, 0))

    return new_image





def pad_image_to_multiple_num(image,num=8,color=(255, 255, 255),aspect_ratio_threshold=2.0):
    """
    Pad the image so that its dimensions are multiples of 8.

    Args:
    image (PIL.Image): The input image.

    Returns:
    PIL.Image: The padded image.
    tuple: Original dimensions of the image.
    """
    original_size = image.size
    new_width = ((original_size[0] - 1) // num + 1) * num
    new_height = ((original_size[1] - 1) // num + 1) * num

    # Check if padding is necessary
    if new_width == original_size[0] and new_height == original_size[1]:
        return image, original_size

    # Adjust padding based on aspect ratio
    aspect_ratio = max(new_width, new_height) / min(new_width, new_height)
    if aspect_ratio > aspect_ratio_threshold:
        if new_width > new_height:
            new_height = ((int(new_width / aspect_ratio_threshold) - 1) // num + 1) * num
        else:
            new_width = ((int(new_height / aspect_ratio_threshold) - 1) // num + 1) * num

    # Create a new image with white background
    new_image = Image.new('RGB', (new_width, new_height), color=color)
    new_image.paste(image, (0, 0))

    return new_image, original_size





def pad_image_to_multiple_factors(image,factors=1024,color=(255, 255, 255)):
    """
    Pad the image so that its dimensions are multiples of 8.

    Args:
    image (PIL.Image): The input image.

    Returns:
    PIL.Image: The padded image.
    tuple: Original dimensions of the image.
    """
    original_size = image.size
    width, height = image.size
    max_width_scale = math.ceil(float(width) / factors)
    max_height_scale = math.ceil(float(height) / factors)

    min_width_padding = math.lcm(max_width_scale * 8, factors)
    min_height_padding = math.lcm(max_height_scale * 8, factors)

    new_width = ((original_size[0] - 1) // min_width_padding + 1) * min_width_padding
    new_height = ((original_size[1] - 1) // min_height_padding + 1) * min_height_padding

    # Check if padding is necessary
    if new_width == original_size[0] and new_height == original_size[1]:
        return image, original_size

    # Create a new image with white background
    new_image = Image.new('RGB', (new_width, new_height), color=color)
    new_image.paste(image, (0, 0))

    return new_image, original_size



def crop_image_to_original(image, original_size):
    """
    Crop the padded image back to its original size.

    Args:
    image (PIL.Image): The padded image.
    original_size (tuple): The original dimensions of the image (width, height).

    Returns:
    PIL.Image: The cropped image.
    """
    return image.crop((0, 0, original_size[0], original_size[1]))



def get_start_size(width,height,fix_size=1024):
    w, h = width,height
    aspect_ratio = w / h
    size = (min(fix_size, int(fix_size * aspect_ratio)),
            min(fix_size, int(fix_size / aspect_ratio)))

    return size

if __name__ == '__main__':
    # Load your image
    img_path = 'path_to_your_image.jpg'
    image = Image.open(img_path)

    # Pad the image
    padded_image, orig_size = pad_image_to_multiple_num(image,num=16,)

    # Later, if you want to crop it back
    cropped_image = crop_image_to_original(padded_image, orig_size)

    # You can now show or save your images
    padded_image.show()
    cropped_image.show()
