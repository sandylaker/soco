# A Dual-Perspective Approach to Evaluating Feature Attribution Methods (Accepted by TMLR 2024)

This repository contains code and scripts accompanying the paper
*["A Dual-Perspective Approach to Evaluating Feature Attribution Methods"](https://openreview.net/forum?id=znlTP5RLur)*
accepted by **Transactions on Machine Learning Research (TMLR) 11/2024**.

---

## Installation

To set up the environment, follow these steps:

1. Install the package:
   ```bash
   pip install -e .
   ```
   *(Do not miss the dot `.`)*

2. For developers, install additional dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. (Optional, for developers) Set up pre-commit hooks:
   ```bash
   pip install pre-commit && pre-commit install
   ```

---

## Data Preparation

### Image Folder Structure

Images should follow the `torchvision.datasets.ImageFolder` format.
Additionally, it is recommended to prepare a JSON file mapping class names to class indices.
This mapping ensures consistency between the class indices determined by the `ImageFolder` alphabetical sorting and
those used by the classifier model.

For example: The model's output logits might correspond to `["B", "A", "C"]`, while the `ImageFolder` format may
organize classes as `["A", "B", "C"]`. For more details, refer to the `soco.datasets.ImageFolder` class.

### Attribution Maps

Attribution maps (referred to as `smap` in this repository) should mirror the structure of the image folder. Key guidelines:
- Attribution map filenames should match the corresponding image filenames.
- Use a consistent format for all attribution maps (e.g., with file extension `.png` or `.jpeg`).

---

## Running Experiments

The main scripts for running experiments are located in the `tools/` directory.

### Scripts
- **Completeness Evaluation**: Use `tools/run_vision_completeness.py`.
- **Soundness Evaluation**: Use `tools/run_vision_soundness.py`.

### Configuration
Before running the scripts, adapt the configuration files located in the `configs/` directory. Configuration can be adjusted in two ways:

1. **Directly Modify Config Files**:
   Update the configuration entries directly in the YAML files. Note that some configurations may inherit settings from other files. For details, see the [mmengine.Config documentation](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html).

2. **Override via Command Line**:
   Use the `-o` option to override specific configurations directly in the command line.

### Performance Note
The completeness and soundness evaluations involve a linear imputation step (based on the ROAD paper). When perturbing a large number of pixels, this step can be computationally expensive as it requires solving a sparse linear system on the CPU.

### Example Usage
Run a completeness evaluation with the following command:

```bash
python tools/run_vision_completeness.py \
  configs/imagenet_completeness/gradcam_imagenet_completeness.yaml \
  workdirs/gradcam/completeness/ \
  -o data.test.img_root=PATH_TO_IMAGE_FOLDER \
     data.test.smap_root=PATH_TO_SMAP_FOLDER \
     data.test.smap_extension=png \
     data.test.cls_to_ind_file=PATH_TO_JSON_FILE \
     classifier.model_name=vgg16
```

---

## Citation


```bibtex
@article{li2024dual,
  title={A Dual-Perspective Approach to Evaluating Feature Attribution Methods},
  author={Li, Yawei and Zhang, Yang and Kawaguchi, Kenji and Khakzar, Ashkan and Bischl, Bernd and Rezaei, Mina},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2024}
}
```
