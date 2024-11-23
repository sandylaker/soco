# A Dual-Perspective Approach to Evaluating Feature Attribution Methods (Accepted by TMLR 2024)

## Installation
1. `pip install -e .` (Do not miss the dot `.`). Developers please run `pip install -e ".[dev]"`.
2. (Optional, for developers only ) Install pre-commit hooks via `pip install pre-commit && pre-commit install`.


## Code structure

### Scripts for running experiments
The scripts are under `tools/` directory.
* `run_vision_completeness.py` is the script for completeness evaluation.
* `run_vision_soundness.py` is the script for soundness evaluation.

### Configuration files
The configuration files are under `configs` directory.

### Library
The library is under `soco/` directory.
* `soco/classifiers/` contains the function for building `mmcls` classifiers.
* `soco/imputations/` contains the imputation methods.
* `soco/datasets/` contains the datasets and image transformation pipelines.
* `soco/metrics/` contains the classes for evaluating soundness and completeness.
* `soco/utils/` contains some utility functions.

### Citation

```bibtex
@article{li2024dual,
  title={A Dual-Perspective Approach to Evaluating Feature Attribution Methods},
  author={Li, Yawei and Zhang, Yang and Kawaguchi, Kenji and Khakzar, Ashkan and Bischl, Bernd and Rezaei, Mina},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2024}
}
```
