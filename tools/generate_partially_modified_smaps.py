import os
import os.path as osp
from argparse import ArgumentParser
from typing import List, Tuple

import cv2
import mmengine
import numpy as np


def parse_args():
    parser = ArgumentParser('Generate partially modified saliency maps. ')
    parser.add_argument('src_smap_root', help='Root path of source saliency maps.')
    parser.add_argument(
        'dst_smap_root', help='Destination path to save the modified saliency maps.')
    parser.add_argument(
        '--range-r',
        type=float,
        default=[0.8, 1.0],
        nargs=2,
        help='Quantile range (in ascending order) of saliency map pixels to be '
        'modified.')
    parser.add_argument(
        '--range-i',
        type=float,
        default=[0.0, 0.4],
        nargs=2,
        help='Quantile range (in ascending order) of saliency map pixels to be '
        'modified.')
    parser.add_argument(
        '--quantile-r',
        type=float,
        default=0.0,
        help='Quantile for computing the replace value in the remove mode.')
    parser.add_argument(
        '--quantile-i',
        type=float,
        default=0.8,
        help='Quantile for computing the replace value in the introduce mode.')
    parser.add_argument(
        '--mode',
        choices=('remove', 'introduce'),
        default='remove',
        help='Remove saliency or introduce saliency.')

    args = parser.parse_args()
    return args


def generate_modified_smaps(
        src_smap_root: str,
        dst_smap_root: str,
        range_r: Tuple[float] = (0.8, 1.0),
        range_i: Tuple[float] = (0.0, 0.4),
        quantile_r: float = 0.0,
        quantile_i: float = 0.8,
        mode: str = 'remove') -> None:
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
            if mode == 'remove':
                # range_r is the range of quantile in ascending order
                # but in Remove, the pixels are sorted in descending order
                start_ind = int((1 - range_r[1]) * smap.size)
                end_ind = int((1 - range_r[0]) * smap.size)
                assert start_ind < end_ind
                modified_smap = remove_saliency(
                    smap, start_ind=start_ind, end_ind=end_ind, quantile=quantile_r)
            else:
                start_ind = int(range_i[0] * smap.size)
                end_ind = int(range_i[1] * smap.size)
                assert start_ind < end_ind
                modified_smap = introduce_saliency(
                    smap, start_ind=start_ind, end_ind=end_ind, quantile=quantile_i)

            dst_path = osp.join(dst_smap_root, synset_id, smap_file)
            cv2.imwrite(dst_path, modified_smap)

        prog_bar.update(1)

    prog_bar.file.write('\n')
    prog_bar.file.flush()


def remove_saliency(
        smap: np.ndarray,
        start_ind: int,
        end_ind: int,
        quantile: float = 0.0) -> np.ndarray:
    modified_smap = np.copy(smap)
    replace_val = np.quantile(smap, q=quantile)

    sorted_y_inds, sorted_x_inds = np.unravel_index(
        np.argsort(-smap, axis=None), shape=smap.shape)

    # locations of pixels that will be modified
    y_inds = sorted_y_inds[start_ind:end_ind]
    x_inds = sorted_x_inds[start_ind:end_ind]
    modified_smap[y_inds, x_inds] = replace_val
    return modified_smap


def introduce_saliency(
        smap: np.ndarray,
        start_ind: int,
        end_ind: int,
        quantile: float = 0.8) -> np.ndarray:
    modified_smap = np.copy(smap)
    if quantile >= 1.0:
        # make sure that the introduced saliency will be ranked higher
        # than the maximal saliency in the sorting process
        max_saliency = np.max(smap)
        replace_val = max_saliency
        smap[smap == max_saliency] -= 1
    else:
        replace_val = np.quantile(smap, q=quantile)

    sorted_y_inds, sorted_x_inds = np.unravel_index(
        np.argsort(smap, axis=None), shape=smap.shape)

    # locations of pixels that will be modified
    y_inds = sorted_y_inds[start_ind:end_ind]
    x_inds = sorted_x_inds[start_ind:end_ind]
    modified_smap[y_inds, x_inds] = replace_val
    return modified_smap


def main():
    args = vars(parse_args())
    args.update({'range_r': tuple(args['range_r']), 'range_i': tuple(args['range_i'])})
    generate_modified_smaps(**args)


if __name__ == '__main__':
    main()
