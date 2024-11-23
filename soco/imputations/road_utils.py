import torch


def get_road_mask_by_area(
        smap: torch.Tensor,
        mask_ratio_by_area: float,
        smap_mask_mode: str = 'morf') -> torch.Tensor:
    height, width = smap.shape
    mask = torch.ones_like(smap, dtype=torch.bool).flatten()
    num_masked_pixels = int(height * width * mask_ratio_by_area)

    if smap_mask_mode == 'morf':
        # larger saliency first
        descending = True
    else:
        # smaller saliency first
        descending = False

    # permute first and then sort
    perm_inds = torch.randperm(height * width, device=smap.device)
    sorted_perm_inds = torch.sort(smap.flatten()[perm_inds], descending=descending)[1]
    sorted_ori_inds = torch.gather(perm_inds, dim=0, index=sorted_perm_inds)
    mask[sorted_ori_inds[:num_masked_pixels]] = 0
    mask = mask.reshape(height, width)

    return mask


def get_road_mask_by_value(
        smap: torch.Tensor,
        mask_ratio_by_value: float,
        smap_mask_mode: str = 'morf') -> torch.Tensor:
    if smap.dim() != 2:
        raise ValueError(f'smap should be a 2-D tensor, but got {smap.dim()}')
    if smap_mask_mode == 'morf':
        # mask_ratio of 0.2 means [0.8 - 1.0] are masked
        threshold = torch.max(smap) * (1 - mask_ratio_by_value)
        # masked pixels are indicated by 0s
        mask = (smap <= threshold)
    else:
        # mask_ratio of 0.2 means [0.0 - 0.2] are masked
        threshold = torch.max(smap) * mask_ratio_by_value
        # masked pixels are indicated by 0s
        mask = (smap >= threshold)

    return mask  # type: ignore
