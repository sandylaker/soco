import os.path as osp
import time
from argparse import ArgumentParser

import mmengine
import torch

from soco.metrics import VisionCompleteness
from soco.utils import setup_logger


def parse_args():
    parser = ArgumentParser('Run the experiment for computing completeness.')
    parser.add_argument('config', help='Path to the config file.')
    parser.add_argument('work_dir', help='Working directory to store the output files.')
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


def main():
    args = parse_args()
    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    device = torch.device(f'cuda:{args.gpu_id}')

    cfg.work_dir = args.work_dir
    mmengine.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = setup_logger('soco', filepath=log_file)

    logger.info(f'Config:\n{cfg.pretty_text}\n' + '-' * 60)

    vision_completeness = VisionCompleteness(cfg=cfg, device=device)
    vision_completeness.run()


if __name__ == '__main__':
    main()
