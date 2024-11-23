from typing import Dict, Optional

from mmengine import Registry
from torch.utils.data import Dataset

DATASETS = Registry('datasets')


def build_dataset(cfg: Dict, default_args: Optional[Dict] = None) -> Dataset:
    return DATASETS.build(cfg=cfg, default_args=default_args)
