import os.path as osp
from argparse import ArgumentParser
from typing import Callable

import cv2
import mmengine
import numpy as np
import torch
import torch.nn as nn
from captum.attr import LayerGradCam
from torch.utils.data import DataLoader
from torchvision.models import vgg16

from soco.datasets import build_dataset, image_folder_collate_fn
from soco.utils import setup_logger


def parse_args():
    parser = ArgumentParser('Get GradCAM or Extremal Perturbation saliency maps.')
    parser.add_argument('config', help='Config file.')
    parser.add_argument(
        'method', choices=('gradcam', 'ex_perturb'), help='Attribution method.')
    parser.add_argument(
        '--area',
        type=float,
        default=0.3,
        help='Perturbed area in Extremal Perturbations. This argument is only useful '
        'when method is ex_perturb.')
    parser.add_argument('--work-dir', help='Directory to store the output files.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=mmengine.DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.',
    )

    args = parser.parse_args()
    return args


def get_gc_ep_smaps(
        cfg: mmengine.Config,
        method: str,
        area: float = 0.3,
        device: str = 'cuda:0') -> None:
    if method not in ('gradcam', 'ex_perturb'):
        raise ValueError(f'Invalid method name: {method}')
    work_dir = cfg.work_dir
    mmengine.mkdir_or_exist(work_dir)
    device = torch.device(device)

    classifier = vgg16(**cfg.classifier)
    if cfg.classifier_ckpt is not None:
        state_dict = torch.load(cfg.classifier_ckpt, map_location='cpu')
        classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()

    attr_set = build_dataset(cfg.data['test'])
    attr_loader = DataLoader(
        attr_set, **cfg.data['data_loader'], collate_fn=image_folder_collate_fn)

    if method == 'gradcam':
        gradcam_layer = classifier.get_submodule('features.28')
        attr_method = LayerGradCam(classifier, gradcam_layer)
    else:
        try:
            from torchray.attribution.extremal_perturbation import extremal_perturbation
        except ImportError:
            raise ImportError(
                'Extremal perturbation requires torchray. '
                "Please install it via 'pip install torchray'.")

        attr_method = extremal_perturbation

    prog_bar = mmengine.ProgressBar(task_num=len(attr_set))
    for data in attr_loader:
        img = data['img'].to(device)
        target = data['target'].to(device)
        img_path = data['meta']['img_path']
        ori_size = data['meta']['ori_size']

        for single_img, single_target, single_img_path, single_ori_size in zip(
                img, target, img_path, ori_size):
            # single_img shape: (1, C, H, W)
            single_img = single_img.unsqueeze(0)
            single_target = single_target.item()
            ori_height, ori_width = single_ori_size

            if method == 'gradcam':
                smap = produce_gradcam_smap(attr_method, img, target)
            else:
                smap = produce_ex_perturb_smap(
                    attr_method, classifier, area, single_img, single_target)

            # smap squueze from (1, C, H, W) to (H, W)
            smap = smap.mean((0, 1))
            smap = np.clip(smap, a_min=0.0, a_max=1.0)
            smap = (smap * 255.0).astype(np.uint8)
            smap = cv2.resize(
                smap, dsize=(ori_width, ori_height), interpolation=cv2.INTER_LINEAR)

            synset_id = osp.basename(osp.dirname(single_img_path))
            # single_img_name does not have a file extension
            single_img_name = osp.splitext(osp.basename(single_img_path))[0]
            out_dir = osp.join(work_dir, synset_id)
            mmengine.mkdir_or_exist(out_dir)

            cv2.imwrite(osp.join(out_dir, f'{single_img_name}.png'), smap)
            prog_bar.update(1)


def produce_gradcam_smap(
        attr_method: LayerGradCam, img: torch.Tensor,
        target: torch.Tensor) -> np.ndarray:
    img.requires_grad = True
    smap = attr_method.attribute(img, target, relu_attributions=True)
    smap = (smap - smap.min()) / (smap.max() - smap.min())
    smap = smap.detach().cpu().numpy()

    img.requires_grad = False
    return smap


def produce_ex_perturb_smap(
        attr_method: Callable,
        classifier: nn.Module,
        area: float,
        img: torch.Tensor,
        target: torch.Tensor) -> np.ndarray:
    smap, _ = attr_method(classifier, img, target, areas=[area])
    smap = smap.detach().cpu().numpy()
    return smap


def main():
    args = parse_args()
    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = args.work_dir
    method = args.method
    device = f'cuda:{args.gpu_id}'

    logger = setup_logger('soco')
    logger.info(f'Config:\n{cfg.pretty_text}\n' + '-' * 60 + '\n')

    get_gc_ep_smaps(cfg, method, args.area, device)


if __name__ == '__main__':
    main()
