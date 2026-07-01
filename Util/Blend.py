
import numpy as np
import cv2

def poisson_blend(
    orig_img: np.ndarray,
    fake_img: np.ndarray,
    mask: np.ndarray,
    pad_width: int = 32,
    dilation: int = 48
) -> np.ndarray:
    """Does poisson blending with some tricks.

    Args:
        orig_img (np.ndarray): Original image.
        fake_img (np.ndarray): Generated fake image to blend.
        mask (np.ndarray): Binary 0-1 mask to use for blending.
        pad_width (np.ndarray): Amount of padding to add before blending (useful to avoid some issues).
        dilation (np.ndarray): Amount of dilation to add to the mask before blending (useful to avoid some issues).

    Returns:
        np.ndarray: Blended image.
    """
    # mask = mask[:, :, 0]
    padding_config = ((pad_width, pad_width), (pad_width, pad_width), (0, 0))
    padded_fake_img = np.pad(fake_img, pad_width=padding_config, mode="reflect")
    padded_orig_img = np.pad(orig_img, pad_width=padding_config, mode="reflect")
    padded_orig_img[:pad_width, :, :] = padded_fake_img[:pad_width, :, :]
    padded_orig_img[:, :pad_width, :] = padded_fake_img[:, :pad_width, :]
    padded_orig_img[-pad_width:, :, :] = padded_fake_img[-pad_width:, :, :]
    padded_orig_img[:, -pad_width:, :] = padded_fake_img[:, -pad_width:, :]
    padded_mask = np.pad(mask, pad_width=padding_config[:2], mode="constant")
    padded_dmask = cv2.dilate(padded_mask, np.ones((dilation, dilation), np.uint8), iterations=1)
    x_min, y_min, rect_w, rect_h = cv2.boundingRect(padded_dmask)
    center = (x_min + rect_w // 2, y_min + rect_h // 2)
    output = cv2.seamlessClone(padded_fake_img, padded_orig_img, padded_dmask, center, cv2.NORMAL_CLONE)
    output = output[pad_width:-pad_width, pad_width:-pad_width]
    return output



