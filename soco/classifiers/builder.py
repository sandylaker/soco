from typing import Dict, Union

import timm
import torch
import torch.nn as nn


def init_classifier(
        classifier: Dict, device: Union[str, torch.device] = None) -> nn.Module:
    """Initialize a classifier.

    Args:
        classifier: Config of the classifier.
        logger: Logger.
        device: device to which the classifier will be moved.

    Returns:
        The classifier.
    """
    classifier = timm.create_model(**classifier)

    classifier.to(device)
    classifier.eval()
    return classifier
