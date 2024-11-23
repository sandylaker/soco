from argparse import ArgumentParser
from typing import Dict

import torch

# map the indices in the features module of mmcls VGG16 to that of torchvision VGG16
FEATURES_INDEX_MAPPING = {
    '0': '0',
    '1': '2',
    '3': '5',
    '4': '7',
    '6': '10',
    '7': '12',
    '8': '14',
    '10': '17',
    '11': '19',
    '12': '21',
    '14': '24',
    '15': '26',
    '16': '28',
}


def parse_args():
    parser = ArgumentParser(
        'Map mmcls VGG16 weights to torchvision VGG16 weights by '
        'changing the keys.')
    parser.add_argument('mmcls_ckpt', help='Path to the mmcls VGG16 checkpoint.')
    parser.add_argument('output_path', help='Output path to store the new weights.')

    args = parser.parse_args()
    return args


def map_key(old_key: str) -> str:
    if old_key.startswith('features'):
        splits = old_key.split('.')
        # ['features', '<index>', 'weight' or 'bias']
        splits = [splits[0], FEATURES_INDEX_MAPPING[splits[1]], splits[3]]
        return '.'.join(splits)
    else:
        return old_key


def map_weights(mmcls_state_dict: Dict) -> Dict:
    # remove the 'backbone' prefix
    mmcls_state_dict = {k[9:]: v for k, v in mmcls_state_dict.items()}
    mmcls_state_dict = {map_key(k): v for k, v in mmcls_state_dict.items()}
    return mmcls_state_dict


def main():
    args = parse_args()
    mmcls_state_dict = torch.load(args.mmcls_ckpt, map_location='cpu')['state_dict']
    vision_state_dict = map_weights(mmcls_state_dict)
    torch.save(vision_state_dict, args.output_path)


if __name__ == '__main__':
    main()
