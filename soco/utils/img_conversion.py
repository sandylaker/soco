from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from numpy.typing import NDArray


def paste_pred_to_img(
        img: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        one_is_remove: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Paste predicted pixels to original image(s).

    Args:
        img: Original image(s) with shape (num_samples, 3, height, width) or
            (3, height, width).
        pred: Predicted image(s) with the same shape as ``img``.
        mask: Binary mask indicating which pixels are predicted ones.
        one_is_remove: If True, the 1s in ``mask`` denote the removed (i.e. imputed)
            pixels, e.g. the masks for MAE. Otherwise, the 0s in ``mask``
            denote the removed (i.e. imputed) pixels .

    Returns:
        A tuple of:
            - masked_img: The masked (original) image, for which the pixels to be
                imputed are masked.
            - pasted_img: The image with masked pixels being replaced by the
                predictions.
    """
    if not one_is_remove:
        mask = 1 - mask
    # by default: 0 means keep, 1 means remove
    masked_img: torch.Tensor = img * (1 - mask)
    pasted_img: torch.Tensor = img * (1 - mask) + pred * mask
    return masked_img, pasted_img


def img_tensor_to_ndarray(
        img: torch.Tensor,
        img_norm_cfg: Optional[Dict[str, Sequence]] = None) -> NDArray[np.uint8]:
    """Convert images from ``torch.Tensor`` to ``numpy.ndarray``.

    This function performs 3 steps:

    1. (Optional) Denormalize the image(s) if ``img_norm_cfg`` is not None.
    2. Permute image dimensions such that the last dimension is the channel dimension.
    3. Convert ``torch.Tensor`` to ``numpy.ndarray`` and conver the data type to
        ``numpy.uint8``.

    Args:
        img: Batched images with shape (num_samples, 3, height, width).
        img_norm_cfg: Image normalization config. If the dict is not None, it should
            contain two fields:
            - "mean": A list of channel mean values. Ihe length is equal to the number
                of channels.
            - "std": A list of channel standard deviation. The length is equal to the
                number of channels.

    Returns:
        Converted image with shape (num_samples, 3, height, width).
    """
    if img_norm_cfg is not None:
        img_mean = img.new_tensor(img_norm_cfg['mean']).reshape(1, -1, 1, 1)
        img_std = img.new_tensor(img_norm_cfg['std']).reshape(1, -1, 1, 1)
        # denormalize
        img = img * img_std + img_mean

    # reshape from (N, C, H, W) to (N, H, W, C), clamp to (0, 255) and
    # convert to ndarray
    img = torch.permute(img, [0, 2, 3, 1]).clamp(min=0.0, max=1.0)
    img: NDArray[np.uint8] = (img * 255.0).to(torch.uint8).cpu().numpy()
    return img
