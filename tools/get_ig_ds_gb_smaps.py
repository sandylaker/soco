import os.path as osp
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional, Tuple, Union, cast

import cv2
import mmengine
import numpy as np
import torch
import torch.nn as nn
from captum.attr import DeepLiftShap, GuidedBackprop, IntegratedGradients, NoiseTunnel
from mmengine import Config
from torch.utils.data import DataLoader
from torchvision.models import vgg16

from soco.classifiers import init_classifier
from soco.datasets import build_dataset, image_folder_collate_fn
from soco.utils import setup_logger


def parse_args() -> Namespace:
    parser = ArgumentParser('Get Saliency maps of Integrated Gradients or DeepSHAP.')
    parser.add_argument('config', help='Path to the data config file.')
    parser.add_argument(
        'method', choices=('ig', 'ds', 'gb'), help='Attribution method name.')
    parser.add_argument(
        'work_dir', help='Working directory for storing the output files.')
    parser.add_argument('--ensemble', help='Ensemble method name.')
    parser.add_argument('--ensemble-bs', type=int, help='Ensemble forward batch size.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=mmengine.DictAction,  # type: ignore
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.',
    )

    args = parser.parse_args()
    return args


class VGG(nn.Module):
    """VGG with out-of-place ReLU."""

    def __init__(
            self,
            features: nn.Module,
            num_classes: int = 1000,
            init_weights: bool = True,
            dropout: float = 0.5) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(False),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(False),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


_vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64,
        64,
        'M',
        128,
        128,
        'M',
        256,
        256,
        256,
        'M',
        512,
        512,
        512,
        'M',
        512,
        512,
        512,
        'M'
    ],
    'E': [
        64,
        64,
        'M',
        128,
        128,
        'M',
        256,
        256,
        256,
        256,
        'M',
        512,
        512,
        512,
        512,
        'M',
        512,
        512,
        512,
        512,
        'M'
    ],
}


class CustomVGG16(nn.Module):
    """ImageNet pre-trained VGG16 with out-of-place ReLU."""

    def __init__(
            self,
            num_classes: int = 1000,
            pretrained: bool = True,
            ckpt: Optional[str] = None) -> None:
        super().__init__()
        self._vgg = VGG(make_layers(_vgg_cfgs['D'], False), num_classes=num_classes)

        if not (pretrained ^ (ckpt is not None)):
            raise ValueError(
                'only one of pretrained and (ckpt is not None) is allowed '
                f'to be True, but got pretrained: {pretrained} and '
                f'ckpt: {ckpt}.')

        if pretrained:
            state_dict = vgg16(pretrained=pretrained).state_dict()
        else:
            state_dict = torch.load(ckpt, map_location='cpu')

        self._vgg.load_state_dict(state_dict)

    def forward(self, x):
        return self._vgg(x)


def get_ig_ds_gb_smaps(
        cfg: Config,
        method: str,
        ensemble: Optional[str] = None,
        ensemble_bs: Optional[int] = None,
        device: Union[str, torch.device] = 'cuda:0') -> None:
    if method not in ('ig', 'ds', 'gb'):
        raise ValueError(f'Invalid method: {method}')
    if cfg.data['test']['type'] != 'ImageFolder':
        raise TypeError('the type of test set should be ImageFolder')
    if cfg.data['test'].get('smap_root', None) is not None:
        raise ValueError('the smap_root in test set should be None.')

    work_dir = cfg.work_dir
    mmengine.mkdir_or_exist(work_dir)

    logger = setup_logger('soco')
    logger.info(f'Saliency maps will be saved to {work_dir}')

    test_set = build_dataset(cfg.data['test'])
    cfg.data['data_loader'].update({'shuffle': False})
    test_loader = DataLoader(
        test_set, **cfg.data['data_loader'], collate_fn=image_folder_collate_fn)

    if 'model_name' in cfg.classifier:
        model = init_classifier(cfg.classifier, device=device)
    else:
        model = CustomVGG16(**cfg.classifier)
        model.eval()
        model.to(device)
    logger.info('VGG16 is built.')

    if method == 'ig':
        attr_method = IntegratedGradients(model, multiply_by_inputs=True)
    elif method == 'ds':
        attr_method = DeepLiftShap(model, multiply_by_inputs=True)
    else:
        attr_method = GuidedBackprop(model)

    if ensemble is not None:
        attr_method = NoiseTunnel(attr_method)

    prog_bar = mmengine.ProgressBar(len(test_set))

    for batch in test_loader:
        img: torch.Tensor = batch['img'].to(device)
        target: torch.Tensor = batch['target'].to(device)
        meta = batch['meta']

        img_path: List[str] = meta['img_path']
        ori_size: List[Tuple[int, int]] = meta['ori_size']

        if method == 'ig' or method == 'ds':
            baseline = torch.zeros_like(img)
            if ensemble is not None:
                smap = attr_method.attribute(
                    img,
                    target=target,
                    nt_type=ensemble,
                    nt_samples=20,
                    stdevs=0.3,
                    nt_samples_batch_size=ensemble_bs)
            else:
                smap = attr_method.attribute(img, baseline, target)
        else:
            if ensemble is not None:
                smap = attr_method.attribute(
                    img,
                    target=target,
                    nt_type=ensemble,
                    nt_samples=20,
                    stdevs=0.3,
                    nt_samples_batch_size=ensemble_bs)
            else:
                smap = attr_method.attribute(img, target)
        # smap shape: (num_samples, height, width)
        smap = torch.mean(torch.abs(smap.detach()), dim=1)

        for _smap, _img_path, _ori_size in zip(smap, img_path, ori_size):
            _smap: torch.Tensor = (_smap - _smap.min()) / (_smap.max() - _smap.min())
            _smap.clamp_(0.0, 1.0)
            _smap: np.ndarray = (_smap * 255.0).to(torch.uint8).cpu().numpy()

            ori_height, ori_width = _ori_size
            _smap = cv2.resize(
                _smap, dsize=(ori_width, ori_height), interpolation=cv2.INTER_LINEAR)

            synset_id = osp.basename(osp.dirname(_img_path))
            # _img_name does not have a file extension
            _img_name = osp.splitext(osp.basename(_img_path))[0]
            out_dir = osp.join(work_dir, synset_id)
            mmengine.mkdir_or_exist(out_dir)

            cv2.imwrite(osp.join(out_dir, f'{_img_name}.png'), _smap)
            prog_bar.update(1)

    prog_bar.file.write('\n')
    logger.info('Finished generating saliency maps.')


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.work_dir = args.work_dir
    device = torch.device(f'cuda:{args.gpu_id}')

    logger = setup_logger('soco')
    logger.info(f'Config:\n{cfg.pretty_text}\n' + '-' * 60 + '\n')

    get_ig_ds_gb_smaps(
        cfg,
        args.method,
        ensemble=args.ensemble,
        ensemble_bs=args.ensemble_bs,
        device=device)
