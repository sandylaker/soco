import os
import os.path as osp
from argparse import ArgumentParser
from typing import Tuple

import cv2
import mmengine
import numpy as np

from soco.utils import setup_logger


def parse_args():
    parser = ArgumentParser(
        'Generate the saliency maps with rectangular salient regions.')

    parser.add_argument('base_smaps_root', help='Root of the baseline saliency maps.')
    parser.add_argument('dst_root', help='Destination root to save the new smaps.')
    parser.add_argument(
        '--rect-size',
        type=int,
        nargs=2,
        default=[30, 45],
        help='Size (in format of (height, width) and in pixels) '
        'of the rectangular region.')

    return parser.parse_args()


def get_rectangular_smaps(
        base_smaps_root: str, dst_root: str, rect_size: Tuple[int, int]) -> None:
    logger = setup_logger('soco')
    mmengine.mkdir_or_exist(dst_root)

    synset_dirs = os.listdir(base_smaps_root)
    pbar = mmengine.ProgressBar(len(synset_dirs))

    for synset in synset_dirs:
        dst_synset_dir = osp.join(dst_root, synset)
        mmengine.mkdir_or_exist(dst_synset_dir)

        smap_files = os.listdir(osp.join(base_smaps_root, synset))
        for smap_file in smap_files:
            smap = cv2.imread(
                osp.join(base_smaps_root, synset, smap_file), cv2.IMREAD_UNCHANGED)
            smap = smap.astype(float) / 255.0

            height, width = smap.shape
            ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
            # set the weight of less salient pixels to 0

            y_center = int(np.clip((smap * ys).sum() / (smap.sum() + 1e-6), 0, height))
            x_center = int(np.clip((smap * xs).sum() / (smap.sum() + 1e-6), 0, width))
            rect_y_range = (y_center - rect_size[0] // 2, y_center + rect_size[0] // 2)
            rect_x_range = (x_center - rect_size[1] // 2, x_center + rect_size[1] // 2)
            rect_y_range = np.clip(rect_y_range, a_min=0, a_max=height)
            rect_x_range = np.clip(rect_x_range, a_min=0, a_max=width)

            rect_smaps = np.zeros((height, width), dtype=np.uint8)
            rect_smaps[rect_y_range[0]:rect_y_range[1],
                       rect_x_range[0]:rect_x_range[1]] = 255

            out_path = osp.join(dst_synset_dir, smap_file)
            cv2.imwrite(out_path, rect_smaps)

        pbar.update(1)

    pbar.file.write('\n')
    pbar.file.flush()
    logger.info(f'Saliency maps are saved under {dst_root}')


if __name__ == '__main__':
    args = vars(parse_args())
    get_rectangular_smaps(**args)
