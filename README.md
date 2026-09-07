# LeJEPA Pretraining and Evaluation Package

A modular PyTorch toolkit for self-supervised backbone pretraining with **LeJEPA**
(prediction loss + Sketched Isotropic Gaussian Regularization), plus leak-free
downstream evaluation (linear-probe classification and regression) and
dataset/result visualization — built around forage/feed image datasets but
usable on any image folder.

---

## Repository Structure

```text
lejepa_workspace/
├── lejepa_Core/
│   ├── __init__.py                # Package exports
│   ├── Backbone_pretrain.py       # Tiling dataset, SIGReg, 4 backbones, pretraining loop
│   ├── Evaluate_classification.py # Linear-probe classification (few-shot + full, 3-way splits)
│   ├── Evaluate_moisture.py       # Linear-probe moisture regression (group-aware splits)
│   └── visualizations.py          # Class-grid overview, PCA plots, confusion matrix, residuals
├── notebooks/
│   ├── showcase.ipynb             # End-to-end demo of the package
│   └── checkpoints/               # Example pretrained weights
├── pyproject.toml / setup.py      # Package configuration (lejepa_Core v0.1.0)
└── README.md
```

## What is LeJEPA?

LeJEPA pretrains an image encoder without labels by:

1. **Tiling** each image into four quadrants and drawing augmented views from
   them (`ForageTileDataset` / `tile_image`).
2. Encoding views with a shared backbone + projection head and minimizing a
   **prediction loss** — the variance of a sample's projected views around
   their mean (`lejepa_prediction_loss`).
3. Regularizing the projected feature space toward an **isotropic Gaussian**
   via **SIGReg** (`SIGReg`), which prevents representation collapse without
   needing negative pairs, a momentum encoder, or a stop-gradient.

The two losses are combined as `(1 - λ) * prediction_loss + λ * sigreg_loss`.

## Installation

```bash
cd lejepa_workspace
pip install -e .
```

This installs the core dependencies declared in `pyproject.toml`:
`torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

A few modules import additional packages that are **not** currently listed as
package dependencies, so install them yourself depending on what you use:

| Module | Extra dependency | Needed for |
| --- | --- | --- |
| `visualizations.py` | `pillow-heif` | HEIC/HEIF image support in `plot_class_grid` |
| `Backbone_pretrain.py` | `Pillow` | Image loading during pretraining |
| `Evaluate_classification.py`, `Evaluate_moisture.py` | `tqdm`, `openpyxl` | Progress bars; writing `.xlsx` summaries |
| `Evaluate_classification.py`, `Evaluate_moisture.py` | `torch-directml` | Only if you keep the hardcoded DirectML device (see caveat below) |

```bash
pip install pillow-heif pillow tqdm openpyxl
```

> **Windows/DirectML caveat:** `Evaluate_classification.py` and
> `Evaluate_moisture.py` hardcode `DEVICE = torch_directml.device()` and
> `import torch_directml`, a Windows-only GPU backend. On macOS/Linux, or if
> you want CUDA/MPS/CPU instead, edit that line before running the module,
> e.g. `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`.
> `Backbone_pretrain.start_pretraining` does **not** have this issue — it
> already auto-selects CUDA/CPU.

## Quickstart: Pretraining a Backbone

```python
from lejepa_Core.Backbone_pretrain import start_pretraining

start_pretraining(
    data_dir="/path/to/unlabeled_images",   # folder of .jpg/.jpeg/.png images
    model_name="resnet18",                  # "convnet" | "efficientnetb0" | "mobilenetv2" | "resnet18"
    proj_dim=128,
    epochs=100,
    batch_size=16,
    lr=1e-3,
    lambda_reg=0.5,                         # weight on the SIGReg term
    save_dir="checkpoints",
)
```

This saves weights to `checkpoints/lejepa_<model_name>_pretrained.pth` and
auto-selects CUDA if available, otherwise CPU.

### Loading pretrained weights into a fresh backbone

```python
from lejepa_Core.Backbone_pretrain import ResNet18Encoder, load_lejepa

encoder = ResNet18Encoder(proj_dim=128)
encoder = load_lejepa(encoder, "checkpoints/lejepa_resnet18_pretrained.pth")
```

`load_lejepa` remaps checkpoint keys (stripping `backbone.` prefixes, etc.) so
weights saved during pretraining line up with the target module's state dict.
It's implemented independently (with a few extra prefix cases) in
`Evaluate_classification.py` and `Evaluate_moisture.py` for their own
`*Backbone` feature-extractor classes.

## Downstream Evaluation

`Evaluate_classification.py` and `Evaluate_moisture.py` are **editable
scripts**, not general-purpose library functions: each defines module-level
constants (dataset paths, checkpoint paths, output directory, hyperparameters)
using `PATH_TO_...` placeholders near the top of the file. Edit those
constants for your dataset layout, then run the module directly or import
`run_classification()` / `run_regression()`.

### Classification linear probing

```python
# In Evaluate_classification.py, set:
#   CLASSIFICATION_ROOT, OUTPUT_DIR, CKPT_RESNET, CKPT_EFFICIENTNET,
#   CKPT_MOBILENET, CKPT_CUSTOMCNN

from lejepa_Core.Evaluate_classification import run_classification
run_classification()
```

Expects `CLASSIFICATION_ROOT` to be a folder of per-class subfolders. The
pipeline:

- Builds a stratified 80/20 train/test split (`make_80_20_splits`).
- Freezes each of the 4 backbones and caches embeddings.
- Trains a `LinearClassHead` per backbone across few-shot regimes
  (`one_shot`, `ten_shot`, `full`) over 5 seeds, reporting mean ± std accuracy.
- Writes `classification_summary.xlsx` to `OUTPUT_DIR`.

### Moisture regression

```python
# In Evaluate_moisture.py, set:
#   BASE_DIR (containing metadata.parquet), OUTPUT_DIR,
#   CKPT_RESNET, CKPT_EFFICIENTNET, CKPT_MOBILENET, CKPT_CUSTOMCNN

from lejepa_Core.Evaluate_moisture import run_regression
run_regression()
```

Expects a `metadata.parquet` with an image-path column (`jpeg_file` or
`image_path`) and a `moisture_content_percent` target column. Trains a
`LinearRegHead` per frozen backbone on an 80/20 split and reports R², MAE,
and RMSE, exported to `moisture_summary.xlsx`.

## Visualization

```python
from lejepa_Core.visualizations import (
    plot_class_grid,
    plot_pca_2d,
    plot_classification_confusion_matrix,
    plot_regression_residuals,
)

# Overview collage of N sample images per class
plot_class_grid("path/to/dataset", "figures/class_overview.png", samples_per_class=4)

# 2D PCA of embeddings (categorical labels -> legend, float labels -> colorbar)
plot_pca_2d(embeddings, labels, class_names, "figures/pca.png")

# Normalized confusion matrix with count + percentage overlays
plot_classification_confusion_matrix(y_true, y_pred, class_names, "figures/confusion.png")

# True-vs-predicted scatter with an ideal-fit line
plot_regression_residuals(y_true, y_pred, "figures/residuals.png")
```

All four functions save directly to `out_path` and close their figure; none
return a value or keep a figure open. `plot_class_grid` requires `pillow-heif`
to be installed (it registers the HEIC/HEIF opener at import time) even if
your dataset only contains JPG/PNG.

## Notebooks

`notebooks/showcase.ipynb` walks through pretraining a ResNet18 backbone and
generating a class-overview grid end to end. `notebooks/checkpoints/` holds
an example pretrained ResNet18 checkpoint you can load with `load_lejepa`.

## Known Limitations

- `Evaluate_classification.py` / `Evaluate_moisture.py` ship with placeholder
  paths and a Windows-only DirectML device — treat them as templates to copy
  and adapt rather than call as-is.
- `pyproject.toml` does not declare `pillow-heif`, `tqdm`, or `openpyxl` as
  dependencies; install them manually per the table above.
- No `LICENSE` file is currently included in this repository.
