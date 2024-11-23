from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from .builder import IMPUTATIONS
from .road_utils import get_road_mask_by_area, get_road_mask_by_value


@IMPUTATIONS.register_module()
class ROAD(nn.Module):
    """`ROAD <https://arxiv.org/abs/2202.00449>`_. The code is adopted from the official
    `codebase`_.

    .. _codebase: https://github.com/tleemann/road_evaluation

    Args:
        noise: Gaussian noise level.
        mask_ratio: Mask ratio (by value or by area).
        smap_mask_mode: mode for computing the binary masks. If 'morf', masking under
            MoRF order; if 'lerf', masking under LeRF order.
        by_area: If True, the mask ratio 0.2 means masking 20% of the pixels. Otherwise,
            mask ratio 0.2 means masking all pixels larger than threshold 0.8 * max_val
            (or smaller than 0.2 * max_val if 'lerf').
    """

    neighbors_weights = [
        ((1, 1), 1 / 12), ((0, 1), 1 / 6), ((-1, 1), 1 / 12), ((1, -1), 1 / 12),
        ((0, -1), 1 / 6), ((-1, -1), 1 / 12), ((1, 0), 1 / 6), ((-1, 0), 1 / 6)
    ]

    def __init__(
        self,
        noise: float = 0.01,
        mask_ratio: Optional[float] = None,
        smap_mask_mode: str = 'morf',
        by_area: bool = True,
    ) -> None:
        super().__init__()

        self.noise = noise
        self._mask_ratio = mask_ratio
        if smap_mask_mode not in ('morf', 'lerf'):
            raise ValueError('Invalid smap_mask_mode')
        self._smap_mask_mode = smap_mask_mode
        self._by_area = by_area

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

    @staticmethod
    def _add_offset_to_indices(
            indices: np.ndarray, offset: Tuple[int, int],
            mask_shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        """Add the corresponding offset to the indices.

        Return new indices plus a valid bit-vector.
        """
        cord1: np.ndarray = indices % mask_shape[1]
        cord0: np.ndarray = indices // mask_shape[1]

        cord0 += offset[0]
        cord1 += offset[1]

        valid = (
            (cord0 < 0) | (cord1 < 0) | (cord0 >= mask_shape[0]) |
            (cord1 >= mask_shape[1]))
        return ~valid, indices + offset[0] * mask_shape[1] + offset[1]

    @staticmethod
    def _setup_sparse_system(
        mask: np.ndarray,
        img: np.ndarray,
        neighbors_weights: List[Tuple[Tuple[int, int], float]]
    ) -> Tuple[lil_matrix, NDArray[np.float64]]:
        maskflt = mask.flatten()
        imgflat = img.reshape((img.shape[0], -1))
        indices = np.argwhere(maskflt == 0).flatten()
        coords_to_vidx = np.zeros(len(maskflt), dtype=int)
        coords_to_vidx[indices] = np.arange(len(indices))

        numEquations = len(indices)
        # System matrix
        A = lil_matrix((numEquations, numEquations))
        b = np.zeros((numEquations, img.shape[0]))
        # Sum of weights assigned
        sum_neighbors = np.ones(numEquations)

        for n in neighbors_weights:
            offset, weight = n[0], n[1]
            # Sum of the neighbors.
            # Take out outliers
            valid, new_coords = ROAD._add_offset_to_indices(indices, offset, mask.shape)

            valid_coords = new_coords[valid]
            valid_ids = np.argwhere(valid == 1).flatten()

            # Add values to the right hand-side
            has_values_coords = valid_coords[maskflt[valid_coords] > 0.5]
            has_values_ids = valid_ids[maskflt[valid_coords] > 0.5]
            b[has_values_ids, :] -= weight * imgflat[:, has_values_coords].T

            # Add weights to the system (left hand side)
            # Find coordinates in the system.
            has_no_values = valid_coords[maskflt[valid_coords] < 0.5]
            variable_ids = coords_to_vidx[has_no_values]
            has_no_values_ids = valid_ids[maskflt[valid_coords] < 0.5]

            A[has_no_values_ids, variable_ids] = weight

            # Reduce weight for invalid
            invalid_ids = np.argwhere(valid == 0).flatten()
            sum_neighbors[invalid_ids] = sum_neighbors[invalid_ids] - weight

        A[np.arange(numEquations), np.arange(numEquations)] = -sum_neighbors
        return A, b

    def _impute_single(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """This is the function to do the linear infilling.

        Args:
            img: single image tensor with shape (num_channels, height, width).
            mask: single binary mask with shape (height, width). Zeros indicate the
                pixels to be imputed.

        Returns:
            The imputed image with shape (C, H, W).
        """
        imgflt = img.reshape(img.shape[0], -1)
        mask_ndarray = mask.cpu().numpy()

        # Indices that need to be imputed.
        indices_linear = np.argwhere(mask_ndarray.reshape(-1) == 0).flatten()
        # Set up sparse equation system, solve system.
        A, b = ROAD._setup_sparse_system(
            mask_ndarray,
            img.cpu().numpy(), self.neighbors_weights)
        res = img.new_tensor(spsolve(csc_matrix(A), b))

        # Fill the values with the solution of the system.
        img_infill = imgflt.clone()
        img_infill[:, indices_linear] = res.t() + self.noise * torch.randn_like(res.t())

        return img_infill.reshape_as(img)

    def forward(
        self,
        img: torch.Tensor,
        smap: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Impute a batched of images.

        .. note::

            This function returns the imputed images. It means that there is no need to
            call ``paste_pred_to_img`` to paste the predicted pixels into original
            images before visualizing them or feeding them into a classifier.

        Args:
            img: Batched images with shape (num_samples, num_channels, height, width).
            smap: Saliency maps with shape (num_samples, height ,width). The data type
                is ``torch.uint8``.

        Returns:
            A dictionary with following fields:
                "imputed_img": The imputed image with shape (num_samples, num_channels,
                    height, width).

                "mask": The binary mask with shape (num_samples, height, width)
                    where 0s are pixels to be imputed and 1s are original pixels.
        """
        if self.mask_ratio is None:
            raise ValueError(
                'ROAD mask_ratio is not set yet, call set_mask_ratio '
                'method first.')

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

        imputed_img = [
            self._impute_single(img_single, mask_single)
            for (img_single, mask_single) in zip(img, mask)
        ]

        imputed_img = torch.stack(imputed_img, 0)
        mask = torch.stack(mask, 0)

        result = {'imputed_img': imputed_img, 'mask': mask}
        return result
