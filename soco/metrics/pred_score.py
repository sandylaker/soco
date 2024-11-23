from typing import List

import torch

from .builder import CLS_METRICS


@CLS_METRICS.register_module()
class PredScore:

    def __init__(self) -> None:
        self._pred_score: List[torch.Tensor] = []

    def reset(self) -> None:
        self._pred_score = []

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        # score shape: (num_samples,)
        score = torch.gather(pred, dim=1, index=target.unsqueeze(1)).squeeze(1)
        self._pred_score.append(score)

    def finalize(self) -> torch.Tensor:
        return torch.cat(self._pred_score, 0)
