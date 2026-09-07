# LeJEPA Pretraining and Evaluation Package

A modular PyTorch toolkit for self-supervised backbone pretraining with **LeJEPA**
(prediction loss + Sketched Isotropic Gaussian Regularization), plus
downstream linear-probe evaluation (classification and regression) and
dataset/result visualization — built around forage/feed image datasets but
usable on any image folder.

---

## Repository Structure

```text
lejepa_workspace/
├── lejepa_Core/
│   ├── __init__.py                # Package exports
│   ├── Backbone_pretrain.py       # Tiling dataset, SIGReg, 4 backbones, pretraining loop
│   ├── Evaluate_classification.py # Linear-probe classification (few-shot + full, 80/20 split)
│   ├── Evaluate_moisture.py       # Linear-probe moisture regression (80/20 split)
│   └── visualizations.py          # Class-grid overview, PCA plots, confusion matrix, residuals
├── notebooks/
│   ├── showcase.ipynb             # End-to-end demo of the package
│   └── checkpoints/               # Example pretrained weights
├── tests/
│   └── test_load_lejepa.py        # Round-trip checks that saved weights load into every backbone
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

This installs the dependencies declared in `pyproject.toml`: `torch`,
`torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `Pillow`,
`tqdm`, `openpyxl`.

Optional extras:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `pip install -e ".[heic]"` | `pillow-heif` | HEIC/HEIF images in `plot_class_grid` (JPG/PNG work without it) |
| `pip install -e ".[directml]"` | `torch-directml` | DirectML GPU backend on Windows |
| `pip install -e ".[test]"` | `pytest` | Running the test suite |

**Device selection.** Every module uses `select_device()`, which picks
DirectML if `torch_directml` is installed, else CUDA, else Apple MPS, else CPU.
No editing is needed to run on macOS/Linux.

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

This saves weights to `checkpoints/lejepa_<model_name>_pretrained.pth` on the
device chosen by `select_device()`.

### Loading pretrained weights into a fresh backbone

```python
from lejepa_Core.Backbone_pretrain import ResNet18Encoder, load_lejepa

encoder = ResNet18Encoder(proj_dim=128)
encoder = load_lejepa(encoder, "checkpoints/lejepa_resnet18_pretrained.pth")
```

`load_lejepa` remaps checkpoint keys so weights saved during pretraining line
up with the target module, whether that is one of the pretraining encoders
above or one of the `*Backbone` feature extractors in the evaluation modules
(which wrap the same layers in an `nn.Sequential` under `features.`). It
**raises** if the checkpoint does not exist, if nothing matches the target
architecture, or (with the default `strict=True`) if any backbone parameter is
left uninitialised, so a silent random-init fallback is no longer possible.
Projection-head weights are ignored when loading into a feature extractor.

`tests/test_load_lejepa.py` round-trips every encoder through save/load into
every consumer and asserts the features are identical:

```bash
cd lejepa_workspace && python -m pytest tests
```

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

- Builds a stratified 80/20 train/test split (`make_80_20_splits`). There is
  no validation split; tune hyperparameters elsewhere, not on the test set.
- Optionally center-crops ("zooms") images before the transforms. `ZOOM_CLASSES`
  defaults to `[]` (off). Setting it to `None` zooms every class. Zooming only
  some classes leaks the label into the input and inflates accuracy, so avoid
  a per-class list unless you have a reason.
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
`LinearRegHead` per frozen backbone on a random 80/20 split and reports R²,
MAE, and RMSE, exported to `moisture_summary.xlsx`. The split is **not**
group-aware: if several images come from the same physical sample, put them on
the same side of the split yourself (e.g. `GroupShuffleSplit`) to avoid leakage.

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
return a value or keep a figure open. `plot_class_grid` reads HEIC/HEIF files
only when `pillow-heif` is installed; otherwise those files are skipped with a
message and JPG/PNG still work.

## Notebooks

`notebooks/showcase.ipynb` walks through pretraining a ResNet18 backbone and
generating a class-overview grid end to end. `notebooks/checkpoints/` holds
an example ResNet18 checkpoint you can load with `load_lejepa`. It was
produced by the notebook (2 epochs on 64 images) and is only a format example,
not a usable pretrained model.

## Known Limitations

- `Evaluate_classification.py` / `Evaluate_moisture.py` ship with placeholder
  paths — treat them as templates to copy and adapt rather than call as-is.
- Both evaluation scripts use a single random 80/20 split with no validation
  set and no grouping (see the notes above).
- No `LICENSE` file is currently included in this repository.
