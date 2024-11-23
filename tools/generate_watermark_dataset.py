import os
import os.path as osp
from argparse import ArgumentParser
from typing import Optional, Tuple

import cv2
import mmengine
import numpy as np

from soco.utils import setup_logger


def parse_args():
    parser = ArgumentParser('Generate watermark dataset.')
    parser.add_argument('img_root', help='Source image root.')
    parser.add_argument('dst_root', help='Destination root.')
    parser.add_argument(
        '--gt-root',
        '-g',
        help='Root for storing the ground-truth watermark. If None, then ground-truth '
        'watermark will not be saved.')
    parser.add_argument(
        '--watermark',
        '-w',
        default='number',
        choices=['stripe', 'number', 'mosaic', 'square'])
    parser.add_argument('--with-bg', action='store_true', help='With background image.')

    args = parser.parse_args()
    return args


def generate_watermark_dataset(
    img_root: str,
    dst_root: str,
    gt_root: Optional[str],
    watermark: str,
    with_bg: bool,
) -> None:
    assert watermark in ('number', 'stripe', 'mosaic', 'square')
    logger = setup_logger('soco')
    mmengine.mkdir_or_exist(dst_root)
    save_gt = gt_root is not None
    if save_gt:
        mmengine.mkdir_or_exist(gt_root)

    synset_ids = sorted(os.listdir(img_root))
    logger.info(f'Found {len(synset_ids)} classes.')

    reassign_probs = get_label_reassign_probs(len(synset_ids))

    # make all the subdirectories
    for label, synset in enumerate(synset_ids):
        dst_sub_dir = osp.join(dst_root, synset)
        mmengine.mkdir_or_exist(dst_sub_dir)
        if save_gt:
            gt_sub_dir = osp.join(gt_root, synset)
            mmengine.mkdir_or_exist(gt_sub_dir)

    # re-assign labels
    for old_label, old_synset in enumerate(synset_ids):
        img_files = os.listdir(osp.join(img_root, old_synset))
        pbar = mmengine.ProgressBar(len(img_files), bar_width=20)

        for img_file in img_files:
            img = cv2.imread(osp.join(img_root, old_synset, img_file))

            probs = reassign_probs[old_label]
            rng = np.random.default_rng()
            new_label = rng.choice(
                len(synset_ids), replace=False, p=probs, shuffle=False)
            new_synset = synset_ids[new_label]

            if not with_bg:
                img = np.zeros_like(img)
            if watermark == 'number':
                img, gt = inject_number_watermark(img, new_label)
            elif watermark == 'stripe':
                img, gt = inject_stripe_watermark(img, new_label)
            elif watermark == 'mosaic':
                img, gt = inject_mosaic_watermark(img, new_label)
            else:
                img, gt = inject_square_watermark(img, new_label)

            dst_path = osp.join(dst_root, new_synset, img_file)
            cv2.imwrite(dst_path, img)
            if save_gt:
                gt_path = osp.join(gt_root, new_synset, img_file)
                cv2.imwrite(gt_path, gt)
            pbar.update(1)

        pbar.file.write('\n')
        pbar.file.flush()

    logger.info(f'Images are saved to {dst_root}')


def inject_number_watermark(img: np.ndarray,
                            label: int) -> Tuple[np.ndarray, np.ndarray]:
    height, width = img.shape[:2]
    gt = np.zeros_like(img)

    org = (int(width * 0.1), int(height * 0.9))
    # draw black region first
    x0, y0 = org[0], org[1] - 50
    x1, y1 = org[0] + 88, org[1] + 10

    img[y0:y1, x0:x1] = 0
    # the black region is modified, so set the corresponding region to 1
    gt[y0:y1, x0:x1] = 255

    text = str(label)
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 10, cv2.LINE_AA)

    cv2.putText(
        gt, text, org, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 10, cv2.LINE_AA)

    return img, gt


def inject_stripe_watermark(img: np.ndarray,
                            label: int) -> Tuple[np.ndarray, np.ndarray]:
    height, width = img.shape[:2]
    # why seven? Because the number of classes is 100, which needs 7 bits
    stripe_height = int(np.ceil(height / 7))
    bin_label = np.asarray([int(x) for x in np.binary_repr(label, width=7)])
    bin_label = np.repeat(bin_label, stripe_height)[:height]
    # convert to a 2-D boolean mask
    bin_label = np.repeat(bin_label[:, np.newaxis], width, axis=1).astype(bool)

    gt = np.zeros_like(img)

    img[bin_label] = 255
    gt[bin_label] = 255

    return img, gt


def inject_square_watermark(img: np.ndarray,
                            label: int) -> Tuple[np.ndarray, np.ndarray]:
    size = 50
    stride = 15
    x0 = (label % 10) * stride
    y0 = (label // 10) * stride
    x1 = min(x0 + size, img.shape[1])
    y1 = min(y0 + size, img.shape[0])

    gt = np.zeros_like(img)

    # BGR
    img[y0:y1, x0:x1] = np.array([255, 0, 0], dtype=np.uint8)
    gt[y0:y1, x0:x1] = 255

    return img, gt


def inject_mosaic_watermark(img: np.ndarray,
                            label: int) -> Tuple[np.ndarray, np.ndarray]:
    block_size = 20

    # 9 digit to make a 3x3 mosaic watermark
    bin_label = np.asarray([int(x) for x in np.binary_repr(label, width=9)])
    # shape: (9, 20)
    bin_label = np.repeat(bin_label, block_size).reshape(-1, block_size)
    # shape: (3, 3, 20, 20)
    bin_label = np.repeat(
        bin_label, block_size, axis=0).reshape((3, 3, block_size, block_size))
    # shape: (60, 60)
    bin_label = np.concatenate(
        [np.concatenate([col for col in row], axis=1) for row in bin_label],
        axis=0).astype(bool)
    whole_mosaic_size = bin_label.shape[0]

    gt = np.zeros_like(img)

    img[:whole_mosaic_size, :whole_mosaic_size][bin_label] = 255
    gt[:whole_mosaic_size, :whole_mosaic_size][bin_label] = 255

    return img, gt


def get_label_reassign_probs(num_classes: int) -> np.ndarray:
    """Get the probability matrix for label re-assignment.

    Args:
        num_classes: Number of classes.

    Returns:
        Probability matrix, where the `[i, j]`-th element indicates the probability of
            re-assign the label `i` to label `j`.
    """
    rng = np.random.default_rng(2022)
    probs = rng.random((num_classes, num_classes))
    probs /= probs.sum(1, keepdims=True)

    if not np.allclose(probs.sum(1), np.ones(num_classes)):
        raise ValueError('Probabilities are not normalized.')

    return probs


if __name__ == '__main__':
    args = vars(parse_args())
    generate_watermark_dataset(**args)
