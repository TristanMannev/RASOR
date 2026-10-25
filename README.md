# 🪒 RASOR 
=============================

### Image Registration via Auxiliary Segmentation and Optical Flow Regularization

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.x-purple)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Code-orange)

RASOR is a Python library for deep-learning-based optical-flow registration
and cell segmentation in microscopy time series. It provides:

- **RASOR-CNN**, a convolutional U-Net with registration and segmentation
  decoders.
- **RASOR**, a Swin Transformer-based model with the same joint outputs.
- **VoxelMorph** and **TransMorph2D** backbones for registration-only
  experiments.

The models accept a source/target image pair and return a displacement field.
RASOR-CNN and RASOR additionally return segmentation logits.

## 📦 Installation

Install the published package: (Not yet available)

```bash
pip install RASOR
```

Install the repository in editable mode for development:

```bash
git clone https://github.com/bia-pasteur/RASOR.git
cd RASOR
pip install -e .
```

## Python API

```python
import torch
from rasor import WarperVxm

model = WarperVxm(
    ndim=2,
    source_channels=1,
    target_channels=1,
    model_type="rasor-cnn",  # or "rasor", "voxelmorph", "transmorph2d"
)

source = torch.randn(1, 1, 256, 256)
target = torch.randn(1, 1, 256, 256)
outputs = model(source, target, return_warped_source=True)
displacement = outputs["displacement"]
segmentation = outputs["source_segmentation"]
```

The public model classes are `rasor.RASORCNNUNet`, `rasor.RASORUNet`,
`rasor.TransMorph2D`, and `rasor.WarperVxm`. The former model identifiers
`flowseg` and `transflowseg` remain accepted as compatibility aliases, but
new experiments should use `rasor-cnn` and `rasor`.

## 🔬 Reproducing the microscopy pipeline

The example pipeline was developed with Cell Tracking Challenge data. Input
videos and masks should follow this structure:

```text
data/
└── data1/
    ├── train_data/
    ├── seg_train_data/
    ├── test_data/
    └── seg_test_data/
```

To create temporally consistent labels from CTC tracking data:

```bash
python scripts/datapreprocessing.py --relabel --dataset data1
```

To generate training masks with Cellpose:

```bash
python scripts/datapreprocessing.py --cellpose --dataset data1
```

Train a model with:

```bash
python scripts/train_cells.py \
  --video-path data/data1/train_data \
  --seg-path data/data1/seg_train_data \
  --model_type rasor-cnn
```

Evaluate a trained model with:

```bash
python scripts/tests.py \
  --video-path data/data1/test_data \
  --seg-path data/data1/seg_test_data \
  --output_dir outputs/rasor-cnn/checkpoint \
  --model_type rasor-cnn
```

## 📜 Citation

When using RASOR in published work, please cite the accompanying paper and
the upstream VoxelMorph and TransMorph papers relevant to the selected
backbone.
