import torch

from .builder import CLS_METRICS


@CLS_METRICS.register_module()
class Accuracy:

    def __init__(self) -> None:
        self._num_samples: int = 0
        self._num_correct: int = 0

    def reset(self) -> None:
        self._num_samples = 0
        self._num_correct = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_class = torch.argmax(pred, dim=-1)
        self._num_correct += (pred_class == target).sum().item()
        self._num_samples += target.shape[0]

    def finalize(self) -> float:
        return self._num_correct / (self._num_samples + 1e-6)
