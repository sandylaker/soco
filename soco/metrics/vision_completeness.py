import os.path as osp
import warnings
from copy import deepcopy
from typing import Union

import numpy as np
import torch
from mmengine import Config, ProgressBar, dump
from torch.utils.data import DataLoader

from ..classifiers import init_classifier
from ..datasets import (
    SUBSET_WRAPPER,
    ImageFolder,
    build_dataset,
    image_folder_collate_fn,
)
from ..imputations import build_imputation
from ..utils import setup_logger
from .accuracy import Accuracy


class VisionCompleteness:

    def __init__(
            self, cfg: Config, device: Union[str, torch.device] = 'cuda:0') -> None:
        self.cfg = cfg
        self.device = device

        self._validate_config(self.cfg)
        self.logger = setup_logger('soco')

        infer_dataset = build_dataset(cfg.data['test'])
        assert isinstance(
            infer_dataset, ImageFolder), 'Currently only supports ImageFolder dataset.'
        if self.cfg.data.get('subset_wrapper', None) is not None:
            wrapper_cfg = deepcopy(self.cfg.data['subset_wrapper'])
            subset_wrapper = SUBSET_WRAPPER.build(wrapper_cfg)
            infer_dataset = subset_wrapper.extract_subset(infer_dataset)

        data_loader_cfg = deepcopy(self.cfg.data['data_loader'])
        data_loader_cfg.update({'collate_fn': image_folder_collate_fn})
        self.data_loader = DataLoader(infer_dataset, **data_loader_cfg)\

        # initialize classifier
        self.classifier = init_classifier(deepcopy(self.cfg.classifier), device=device)

        mask_ratios_start, mask_ratios_end, mask_ratios_step = (
            self.cfg.mask_ratios['start'],
            self.cfg.mask_ratios['end'],
            self.cfg.mask_ratios['step'],
        )
        # round the ratios to make them prettier in log messages and dumped JSON file
        self.mask_ratios = np.arange(
            mask_ratios_start, mask_ratios_end, mask_ratios_step).round(5)

        self.imputation = build_imputation(cfg.imputation)
        self.imputation.eval()
        self.imputation.to(device)

        self.result_dict = {'mask_ratios': self.mask_ratios.tolist(), 'acc_diffs': []}

    def run(self) -> None:
        for i, mask_ratio in enumerate(self.mask_ratios):
            acc_diff = self._run_vision_completeness_single(mask_ratio=mask_ratio)

            self.result_dict['acc_diffs'].append(acc_diff)
            self.logger.info(
                f'mode: {self.imputation.smap_mask_mode}; '
                f'by_area: {self.imputation.by_area}; '
                f'mask_ratio: {mask_ratio:.2f}; '
                f'acc_diff: {acc_diff:.4f}')

        # dump the main result
        out_file = osp.join(self.cfg.work_dir, 'vision_completeness_metric.json')
        dump(self.result_dict, file=out_file)
        self.logger.info(f'Result is dumped to: {out_file}')

    @torch.no_grad()
    def _run_vision_completeness_single(self, mask_ratio: float) -> float:
        self.imputation.set_mask_ratio(mask_ratio)

        prog_bar = ProgressBar(
            len(self.data_loader.dataset), bar_width=20)  # type: ignore
        ori_acc_metric = Accuracy()
        imputed_acc_metric = Accuracy()

        for i, data in enumerate(self.data_loader):
            smap = data.get('smap', None)
            if smap is not None:
                smap = smap.to(self.device)
            img = data['img'].to(self.device)
            label = data['target'].to(self.device)

            # compute the reconstruction result
            result = self.imputation(img=img, smap=smap)
            imputed_img = result['imputed_img']

            ori_score = self.classifier(img)
            ori_acc_metric.update(ori_score, label)
            imputed_score = self.classifier(imputed_img)
            imputed_acc_metric.update(imputed_score, label)

            batch_size = data['img'].size(0)
            for _ in range(batch_size):
                prog_bar.update()

        prog_bar.file.write('\n')
        prog_bar.file.flush()

        acc_diff = ori_acc_metric.finalize() - imputed_acc_metric.finalize()
        return acc_diff

    @staticmethod
    def _validate_config(cfg: Config) -> None:

        # validate mask ratios
        if 'mask_ratio' in cfg.imputation:
            raise ValueError(
                "'mask_ratio' of imputation should not be specified in the "
                'cfg.imputation entry. It will be dynamically set by the '
                "'mask_ratios' entry in the config.")

        if cfg.imputation.get('by_area', True):
            warnings.warn(
                'When running completeness metric, imputation.by_area should '
                'be false. Otherwise, it is equivalent to ROAD or Insertion/Deletion.')

        mask_ratios_start, mask_ratios_end, mask_ratios_step = (
            cfg.mask_ratios['start'],
            cfg.mask_ratios['end'],
            cfg.mask_ratios['step'],
        )
        if min(mask_ratios_start, mask_ratios_end) < 0 or max(mask_ratios_start,
                                                              mask_ratios_end) > 1:
            raise ValueError(
                f'Mask ratios config should have start, end in [0.0, 1.0], '
                f'but got {cfg.mask_ratios}')
