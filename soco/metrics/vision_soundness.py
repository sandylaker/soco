import os.path as osp
import warnings
from copy import deepcopy
from typing import List, Optional, Tuple, TypeVar, Union

import cv2
import mmengine
import numpy as np
import torch
from mmengine import Config, ProgressBar, dump
from torch.utils.data import DataLoader, TensorDataset

from ..classifiers import init_classifier
from ..datasets import (
    SUBSET_WRAPPER,
    ImageFolder,
    build_dataset,
    image_folder_collate_fn,
)
from ..imputations import build_imputation
from ..utils import setup_logger
from .builder import CLS_METRICS

T = TypeVar('T', float, torch.Tensor)


class VisionSoundness:

    def __init__(
            self, cfg: Config, device: Union[str, torch.device] = 'cuda:0') -> None:
        self.cfg = cfg
        self.device = device

        self._validate_config(self.cfg)
        self.logger = setup_logger('soco')

        infer_dataset = build_dataset(self.cfg.data['test'])
        assert isinstance(
            infer_dataset, ImageFolder), 'Currently only supports ImageFolder dataset.'
        if self.cfg.data.get('subset_wrapper', None) is not None:
            wrapper_cfg = deepcopy(self.cfg.data['subset_wrapper'])
            subset_wrapper = SUBSET_WRAPPER.build(wrapper_cfg)
            infer_dataset = subset_wrapper.extract_subset(infer_dataset)

        data_loader_cfg = deepcopy(self.cfg.data['data_loader'])
        data_loader_cfg.update({'collate_fn': image_folder_collate_fn})
        self.data_loader = DataLoader(infer_dataset, **data_loader_cfg)

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

        self.cls_threshold = cfg.soundness['cls_threshold']
        self.cls_result_start = cfg.soundness['cls_result_start']
        self.result_dict = {
            'mask_ratios': self.mask_ratios.tolist(),
            'cls_results': [],
            'attr_ratios': []
        }

    def run(self) -> None:
        prev_mask_ratio: float = 1.0
        accum_false_attr: torch.Tensor = torch.zeros(
            (len(self.data_loader.dataset), ),  # type: ignore
            dtype=torch.float32,
            device=self.device)

        # if True, save boolean tensors indicating false attribution
        if self.cfg.soundness.get('save_is_false_attr', False):
            # initialize a bool tensor for each sample. Query the first sample to
            # get the transformed image size.
            # sample_img shape: (num_channels, height, width)
            sample_img = self.data_loader.dataset[0]['img']
            height, width = sample_img.shape[1:]
            num_samples = len(self.data_loader.dataset)  # type: ignore
            is_false_attr: torch.Tensor = torch.zeros(
                (num_samples, height, width), dtype=torch.bool, device=self.device)
        else:
            is_false_attr: Optional[torch.Tensor] = None
        prev_cls_result: Union[float, torch.Tensor] = 0

        for i, curr_mask_ratio in enumerate(self.mask_ratios):
            single_result = self._run_vision_soundness_single(
                prev_mask_ratio=prev_mask_ratio,
                curr_mask_ratio=curr_mask_ratio,
                accum_false_attr=accum_false_attr,
            )
            attr_ratio, curr_cls_result, new_added_attr, is_new_added = single_result

            self._update_false_attr(
                curr_cls_result,
                prev_cls_result,
                self.cls_threshold,
                accum_false_attr,
                new_added_attr,
                is_false_attr,
                is_new_added,
                cls_result_start=self.cls_result_start)
            prev_cls_result = curr_cls_result
            prev_mask_ratio = curr_mask_ratio

            if isinstance(curr_cls_result, torch.Tensor):
                self.result_dict['cls_results'].append(curr_cls_result.mean().item())
            else:
                self.result_dict['cls_results'].append(curr_cls_result)
            self.result_dict['attr_ratios'].append(attr_ratio)

            self._log_single_step(curr_mask_ratio, attr_ratio, curr_cls_result)

        self.dump_result(is_false_attr)

    @staticmethod
    def _update_false_attr(
        curr_cls_result: T,
        prev_cls_result: T,
        cls_threshold: float,
        accum_false_attr: torch.Tensor,
        new_added_attr: torch.Tensor,
        is_false_attr: Optional[torch.Tensor],
        is_new_added: torch.Tensor,
        cls_result_start: Optional[float] = None,
    ) -> None:
        if cls_result_start is not None:
            # if using prediction score as metric, compare the mean score with the
            # cls_result_start, otherwise use accuracy
            if isinstance(curr_cls_result, torch.Tensor):
                mean_cls_result = curr_cls_result.mean().item()
            else:
                mean_cls_result = curr_cls_result

            # if the mean predictions score or accuracy is too low,
            # then do not update the false attribution
            if mean_cls_result <= cls_result_start:
                return

        if isinstance(curr_cls_result, torch.Tensor):
            # use prediction score as classification metric
            to_be_updated = (curr_cls_result - prev_cls_result) <= cls_threshold
            accum_false_attr[to_be_updated] += new_added_attr[to_be_updated]
            if is_false_attr is not None:
                is_false_attr[to_be_updated, ...] += is_new_added[to_be_updated, ...]
        elif isinstance(curr_cls_result, float):
            # use accuracy as classification metric
            if (curr_cls_result - prev_cls_result) <= cls_threshold:
                accum_false_attr += new_added_attr
                if is_false_attr is not None:
                    is_false_attr += is_new_added
        else:
            raise TypeError(
                f'Invalid type of curr_cls_result: '
                f'{curr_cls_result.__class__.__name__}')

    def _log_single_step(
        self,
        curr_mask_ratio: float,
        mean_attr_ratio: float,
        cls_result: Union[torch.Tensor, float],
    ):
        if isinstance(cls_result, torch.Tensor):
            # use pred score as classification metric
            cls_result = cls_result.mean().item()

        self.logger.info(
            f'mode: {self.imputation.smap_mask_mode}; '
            f'by_area: {self.imputation.by_area}; '
            f'mask_ratio: {curr_mask_ratio:.2f}; '
            f'attr_ratio: {mean_attr_ratio:.4f}; '
            f'cls_result: {cls_result:.4f}')

    @torch.no_grad()
    def _run_vision_soundness_single(
        self,
        prev_mask_ratio: float,
        curr_mask_ratio: float,
        accum_false_attr: torch.Tensor,
    ) -> Tuple[float, Union[float, torch.Tensor], torch.Tensor, torch.Tensor]:
        self.imputation.set_mask_ratio(curr_mask_ratio)
        prog_bar = ProgressBar(
            len(self.data_loader.dataset), bar_width=20)  # type: ignore

        cls_metric = CLS_METRICS.build(self.cfg['soundness']['cls_metric'])
        attr_ratio_list: List[torch.Tensor] = []
        new_added_attr_list: List[torch.Tensor] = []
        is_new_added_list: List[torch.Tensor] = []
        false_attr_loader = DataLoader(
            TensorDataset(accum_false_attr),
            batch_size=self.data_loader.batch_size,
            shuffle=False,
            num_workers=0,
        )

        for i, (data, (batch_accum_false_attr, )) in enumerate(zip(self.data_loader,
                                                                   false_attr_loader)):
            smap = data.get('smap', None)
            if smap is not None:
                smap = smap.to(self.device)
            img = data['img'].to(self.device)
            label = data['target'].to(self.device)

            attr_ratio, batch_new_added_attr, batch_is_new_added = \
                self.compute_attr_ratio(
                    smap, prev_mask_ratio, curr_mask_ratio, batch_accum_false_attr)
            attr_ratio_list.append(attr_ratio)
            new_added_attr_list.append(batch_new_added_attr)
            is_new_added_list.append(batch_is_new_added)

            # compute the reconstruction result
            imputed_img = self.imputation(img=img, smap=smap)['imputed_img']

            cls_score = self.classifier(imputed_img)
            cls_metric.update(cls_score, label)

            for _ in range(len(label)):
                prog_bar.update()

        prog_bar.file.write('\n')
        prog_bar.file.flush()

        mean_attr_ratio = torch.concat(attr_ratio_list).mean().item()
        cls_result: Union[float, torch.Tensor] = cls_metric.finalize()
        new_added_attr = torch.cat(new_added_attr_list)
        is_new_added = torch.cat(is_new_added_list)

        return mean_attr_ratio, cls_result, new_added_attr, is_new_added

    @staticmethod
    def compute_attr_ratio(
        smap: torch.Tensor,
        prev_mask_ratio: float,
        curr_mask_ratio: float,
        accum_false_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the ratio between remaining attribution and total attribution.

        Args:
            smap: Batched saliency map with size ``(num_samples, height, width)``.
            prev_mask_ratio: Previous mask ratio (LeRF).
            curr_mask_ratio: Current mask ratio (LeRF).
            accum_false_attr: Accumulative false attribution with shape
                ``(num_samples,)``.

        Returns:
            A tuple of:
                Attribution ratio with shape ``(num_samples,)``.
                New added attribution with shape ``(num_samples,)``.
                A boolean tensor indicating whether a pixel is new inserted at the
                    current step, shape ``(num_samples, height, width)``.
        """
        if smap.dim() != 3:
            raise ValueError(f'smap should have shape (N, H, W), but got {smap.shape}')
        # first convert the smap from uint8 to float32
        # and then normalize it to range (0, 1)
        smap = smap.to(torch.float32) / 255.0
        smap_shape = smap.shape
        smap = smap.view(smap.shape[0], -1)

        # LeRF mode
        # first permute and then sort
        perm_inds = torch.randperm(smap.shape[1], device=smap.device)
        sorted_perm_inds = torch.sort(smap[:, perm_inds], dim=1, descending=False)[1]
        sorted_ori_inds = torch.gather(
            perm_inds.unsqueeze(0).repeat(smap.shape[0], 1),
            dim=1,
            index=sorted_perm_inds)
        sorted_smap = torch.gather(smap, dim=1, index=sorted_ori_inds)

        curr_num_masked = int(smap.shape[1] * curr_mask_ratio)
        prev_num_masked = int(smap.shape[1] * prev_mask_ratio)
        assert curr_num_masked <= prev_num_masked

        # compute the new added attribution at the current iteration
        # shape: (num_samples,)
        new_added_attr = sorted_smap[:, curr_num_masked:prev_num_masked].sum(1)
        # shape: (num_samples, num_new_added_pixels)
        new_added_inds = sorted_ori_inds[:, curr_num_masked:prev_num_masked]
        is_new_added = torch.zeros_like(smap)
        is_new_added.scatter_(dim=1, index=new_added_inds, src=torch.ones_like(smap))
        # reshape the indicator tensor to original shape (N, H, W)
        # and convert it to bool
        is_new_added = is_new_added.view(smap_shape).to(torch.bool)

        # first subtract sum of remaining attribution from accumulative
        # false attribution, then divide it by total remaining attribution
        remaining_attr = sorted_smap[:, curr_num_masked:].sum(1)
        remaining_attr_wo_false = remaining_attr - accum_false_attr
        attr_ratio = remaining_attr_wo_false / (remaining_attr + 1e-6)

        return attr_ratio, new_added_attr, is_new_added

    def dump_result(self, is_false_attr: Optional[torch.Tensor] = None) -> None:
        if is_false_attr is not None:
            is_false_attr = is_false_attr.cpu()
            self._save_is_false_attr(is_false_attr, self.data_loader, self.cfg.work_dir)
            self.logger.info(
                f'is_false_attr have been save as png images under '
                f'{osp.join(self.cfg.work_dir, "is_false_attr")}')

        # dump the result
        out_file = osp.join(self.cfg.work_dir, 'vision_soundness_metric.json')
        dump(self.result_dict, file=out_file)
        self.logger.info(f'Result is dumped to: {out_file}')

    @staticmethod
    def _save_is_false_attr(
            is_false_attr: torch.Tensor, data_loader: DataLoader,
            work_dir: str) -> None:
        save_dir = osp.join(work_dir, 'is_false_attr')
        mmengine.mkdir_or_exist(save_dir)

        is_false_attr_loader = DataLoader(
            TensorDataset(is_false_attr),
            batch_size=data_loader.batch_size,
            shuffle=False,
            num_workers=0)

        for i, (data, (batch_is_false_attr, )) in enumerate(zip(data_loader,
                                                                is_false_attr_loader)):

            img_path = data['meta']['img_path']
            batch_is_false_attr = batch_is_false_attr.numpy().astype(np.uint8)

            for single_is_false_attr, single_img_path in zip(batch_is_false_attr,
                                                             img_path):
                # img_file: e.g. ILSVRC2012_xxx.JPEG
                img_file = osp.basename(single_img_path)
                synset_id = osp.basename(osp.dirname(single_img_path))

                dst_dir = osp.join(save_dir, synset_id)
                mmengine.mkdir_or_exist(dst_dir)

                dst_path = osp.join(dst_dir, osp.splitext(img_file)[0] + '.png')
                cv2.imwrite(dst_path, single_is_false_attr)

    @staticmethod
    def _validate_config(cfg: Config) -> None:
        # validate mask ratios
        if 'mask_ratio' in cfg.imputation:
            raise ValueError(
                "'mask_ratio' of the model should not be specified in the "
                'cfg.model entry. It will be dynamically set by the '
                "'mask_ratios' entry in the config.")

        if not cfg.imputation.get('by_area', True):
            raise ValueError('The cfg.mask_ratios.by_area should be True.')

        if cfg.imputation.get('smap_mask_mode', 'morf') != 'lerf':
            warnings.warn("Soundness metric should have smap_mask_mode='lerf'.")

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
        if mask_ratios_start <= mask_ratios_end or mask_ratios_step >= 0:
            raise ValueError(
                'Mask ratio should start from a larger number to a smaller number with '
                'a negative step.')
