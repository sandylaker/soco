from typing import Dict, Optional

import torch
import torch.nn as nn

from .builder import IMPUTATIONS
from .road_utils import get_road_mask_by_area, get_road_mask_by_value


@IMPUTATIONS.register_module()
class IdentityImputation(nn.Module):

    def __init__(
            self,
            mask_ratio: Optional[float] = None,
            smap_mask_mode: str = 'morf',
            by_area: bool = True,
            fill_value: float = 0.0) -> None:
        super(IdentityImputation, self).__init__()

        self._mask_ratio = mask_ratio
        if smap_mask_mode not in ('morf', 'lerf'):
            raise ValueError('Invalid smap_mask_mode')
        self._smap_mask_mode = smap_mask_mode
        self._by_area = by_area
        self.fill_value = fill_value

    def set_mask_ratio(self, mask_ratio: float) -> None:
        self._mask_ratio = mask_ratio

    @property
    def mask_ratio(self) -> Optional[float]:
        return self._mask_ratio

    def set_smap_mask_mode(self, smap_mask_mode: str) -> None:
        if smap_mask_mode not in ('morf', 'lerf'):
            raise ValueError('Invalid smap_mask_mode')
        self._smap_mask_mode = smap_mask_mode

    @property
    def smap_mask_mode(self) -> str:
        return self._smap_mask_mode

    def set_by_area(self, by_area: bool) -> None:
        self._by_area = by_area

    @property
    def by_area(self) -> bool:
        return self._by_area

    def forward(self, img: torch.Tensor, smap: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.mask_ratio is None:
            raise ValueError(
                'ROAD mask_ratio is not set yet, call set_mask_ratio '
                'method first.')

        # mask: 0 indicates being masked, 1 indicates being kept
        if self._by_area:
            mask = [
                get_road_mask_by_area(
                    s, self.mask_ratio, smap_mask_mode=self.smap_mask_mode)
                for s in smap
            ]
        else:
            mask = [
                get_road_mask_by_value(
                    s, self.mask_ratio, smap_mask_mode=self.smap_mask_mode)
                for s in smap
            ]

        mask = torch.stack(mask, 0)
        # mask_broad_casted shape: (num_samples, num_channels, height, width)
        mask_broad_casted = mask.unsqueeze(1).broadcast_to(img.shape)

        imputed_img = img.clone()
        imputed_img[~mask_broad_casted] = self.fill_value
        result = {'imputed_img': imputed_img, 'mask': mask}
        return result
