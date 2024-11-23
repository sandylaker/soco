import os
import os.path as osp
from argparse import ArgumentParser

import cv2
import mmengine
import torch
import torch.nn.functional as F

from soco.utils import setup_logger


def parse_args():
    parser = ArgumentParser('Generate the saliency maps by pooling base saliency maps.')

    parser.add_argument('base_smaps_root', help='Root of the baseline saliency maps.')
    parser.add_argument('dst_root', help='Destination root to save the new smaps.')
    parser.add_argument(
        '--stripe-height', '-s', type=int, default=32, help='Height of the stripes.')

    return parser.parse_args()


def get_pooling_smaps(base_smaps_root: str, dst_root: str, stripe_height: int) -> None:
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
            # avg_pool2d requires a 4-D tensor
            smap = torch.from_numpy(smap).to(torch.float32).unsqueeze(0).unsqueeze(0)
            # first apply pooling and then convert the 4-D tensor to 2-D
            # the shape after the operations should be (pooled_height, 1)
            pooling_smap = F.avg_pool2d(smap,
                                        (stripe_height, width)).squeeze(0).squeeze(0)
            # first apply repeat_interleave to expand the height of the smap
            # then apply repeat (equivalent to np.tile) to expand the width of the samp
            pooling_smap = torch.repeat_interleave(
                pooling_smap, stripe_height, dim=0)[:height, :]
            pooling_smap = pooling_smap.repeat(1, width)
            assert pooling_smap.shape == (height, width)

            # convert the values from [0., 1.] to [0, 255] and dtype to uint8
            pooling_smap = (pooling_smap * 255.).clamp(
                min=0, max=255).to(torch.uint8).numpy()

            out_path = osp.join(dst_synset_dir, smap_file)
            cv2.imwrite(out_path, pooling_smap)

        pbar.update(1)

    pbar.file.write('\n')
    pbar.file.flush()
    logger.info(f'Saliency maps are saved under {dst_root}')


if __name__ == '__main__':
    args = vars(parse_args())
    get_pooling_smaps(**args)
