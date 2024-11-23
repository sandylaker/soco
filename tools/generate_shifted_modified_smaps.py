import os
import os.path as osp
from argparse import ArgumentParser
from typing import List

import cv2
import mmengine
import numpy as np


def parse_args():
    parser = ArgumentParser('Generate shifted modified saliency maps.')
    parser.add_argument('src_smap_root', help='Root path of source saliency maps.')
    parser.add_argument(
        'dst_smap_root', help='Destination path to save the modified saliency maps.')
    parser.add_argument(
        'shift_type', choices=['constant', 'random'], help='Type of shift')
    parser.add_argument(
        '--scale',
        type=float,
        default=0.6,
        help='Scale of shift. This value will be used as the shift in the constant '
        'shift type, and as the scale of uniform distribution in the random '
        'shift type.')
    parser.add_argument(
        '--mode',
        choices=('remove', 'introduce'),
        default='remove',
        help='Remove saliency or introduce saliency.')

    args = parser.parse_args()
    return args


def generate_shifted_smaps(
        src_smap_root: str,
        dst_smap_root: str,
        shift_type: str,
        scale: float,
        mode: str = 'remove') -> None:
    if shift_type not in ('random', 'constant'):
        raise ValueError(
            f"shift_type must be one of 'random' or 'constant', but got {shift_type}")
    if mode not in ('remove', 'introduce'):
        raise ValueError(f"mode must be one of 'more' or 'less', but got {mode}")

    mmengine.mkdir_or_exist(dst_smap_root)

    synset_id_list: List[str] = os.listdir(src_smap_root)
    prog_bar = mmengine.ProgressBar(len(synset_id_list))

    for synset_id in synset_id_list:
        mmengine.mkdir_or_exist(osp.join(dst_smap_root, synset_id))

        smap_files = os.listdir(osp.join(src_smap_root, synset_id))
        for smap_file in smap_files:
            smap: np.ndarray = cv2.imread(
                osp.join(src_smap_root, synset_id, smap_file), cv2.IMREAD_UNCHANGED)
            modified_smap = shift_smap(smap, shift_type, scale, mode)
            dst_path = osp.join(dst_smap_root, synset_id, smap_file)
            cv2.imwrite(dst_path, modified_smap)

        prog_bar.update(1)

    prog_bar.file.write('\n')
    prog_bar.file.flush()


def shift_smap(
        smap: np.ndarray, shift_type: str, scale: float, mode: str) -> np.ndarray:
    # first change smap from uint8 to int32 to prevent under-/overflow when
    # subtracting/adding the shift
    smap = smap.astype(np.int32)

    if shift_type == 'random':
        shift = np.random.randint(0, int(scale * 255), smap.shape, dtype=np.int32)
    else:
        shift = np.int32(scale * 255)

    if mode == 'introduce':
        smap = np.clip(smap + shift, a_min=0, a_max=255)
    else:
        smap = np.clip(smap - shift, a_min=0, a_max=255)

    smap = smap.astype(np.uint8)
    return smap


def main():
    args = vars(parse_args())
    generate_shifted_smaps(**args)


if __name__ == '__main__':
    main()
