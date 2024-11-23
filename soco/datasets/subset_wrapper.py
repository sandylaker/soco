import os.path as osp
from typing import List, Optional

import numpy as np
from mmengine import Registry
from torch.utils.data import Subset

from .image_folder import ImageFolder

SUBSET_WRAPPER = Registry('subset wrapper')


@SUBSET_WRAPPER.register_module()
class RandomSubset:

    def __init__(self, num_samples: int, seed: Optional[int] = None) -> None:
        self.num_samples = num_samples
        self.seed = seed

    def extract_subset(self, dataset: ImageFolder) -> Subset:
        random_sate = np.random.RandomState(seed=self.seed)
        sample_inds = random_sate.randint(0, len(dataset), self.num_samples)
        subset = Subset(dataset, indices=sample_inds)
        return subset


@SUBSET_WRAPPER.register_module()
class RangeSubset:

    def __init__(self, start: int, end: int, step: Optional[int] = None) -> None:
        self.start = start
        self.end = end
        self.step = step

    def extract_subset(self, dataset: ImageFolder) -> Subset:
        sample_inds = np.arange(self.start, self.end, self.step)
        subset = Subset(dataset, indices=sample_inds)
        return subset


@SUBSET_WRAPPER.register_module()
class KFoldSubset:

    def __init__(self, num_folds: int = 3, fold_ind: int = 0) -> None:
        assert num_folds > 0 and fold_ind < num_folds
        self.num_folds = num_folds
        self.fold_ind = fold_ind

    def extract_subset(self, dataset: ImageFolder) -> Subset:
        img_paths = dataset.img_paths
        cls_to_ind = dataset.get_cls_to_ind()
        sample_ind_to_label = {
            i: cls_to_ind[osp.basename(osp.split(p)[0])]
            for i, p in enumerate(img_paths)
        }

        num_classes = len(cls_to_ind.values())

        inds_per_class: List[List[int]] = [[] for _ in range(num_classes)]
        for i, label in sample_ind_to_label.items():
            inds_per_class[label].append(i)

        selected_sample_inds = []
        for inds in inds_per_class:
            num_samples_per_class = len(inds)
            fold_size = num_samples_per_class // self.num_folds
            start = fold_size * self.fold_ind
            if self.fold_ind == self.num_folds - 1:
                end = num_samples_per_class
            else:
                end = fold_size * (self.fold_ind + 1)

            selected_sample_inds.extend(inds[start:end])

        subset = Subset(dataset, selected_sample_inds)
        return subset
