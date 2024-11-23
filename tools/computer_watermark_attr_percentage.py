import os
import os.path as osp
from argparse import ArgumentParser

import cv2
import numpy as np
from mmengine import ProgressBar

from soco.utils import setup_logger


def parse_args():
    parser = ArgumentParser('Compute the Attr% of the watermark semi-natural dataset')

    parser.add_argument('smap_root', help='Saliency map root.')
    parser.add_argument('watermark_root', help='Watermark root.')

    return parser.parse_args()


def compute_watermark_attr_percentage(smap_root: str, watermark_root: str) -> None:
    logger = setup_logger('soco')

    synset_dirs = os.listdir(smap_root)
    pbar = ProgressBar(len(synset_dirs))
    all_attr_perc = []

    for synset in synset_dirs:
        smap_files = os.listdir(osp.join(smap_root, synset))

        for smap_file in smap_files:
            gt_path = osp.join(watermark_root, synset, smap_file)
            gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2GRAY)
            if gt is None:
                raise ValueError('gt watermark map is not correctly loaded.')
            gt = (gt != 0)

            smap_path = osp.join(smap_root, synset, smap_file)
            smap = cv2.imread(smap_path, cv2.IMREAD_UNCHANGED)
            smap = smap.astype(float) / 255.0

            attr_perc = (smap * gt).sum() / (smap.sum() + 1e-8)
            all_attr_perc.append(attr_perc)

        pbar.update(1)

    pbar.file.write('\n')
    pbar.file.flush()

    mean_attr_perc = np.mean(all_attr_perc)
    logger.info(f'Average Attr% is: {mean_attr_perc:.4f}')


if __name__ == '__main__':
    args = vars(parse_args())
    compute_watermark_attr_percentage(**args)
