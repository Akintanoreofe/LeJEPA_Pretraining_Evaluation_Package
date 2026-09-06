"""
Visualization utilities for dataset inspection and model evaluation.

Includes helpers for:
    - Building a class-overview image grid from a labeled image dataset.
    - Projecting high-dimensional embeddings to 2D via PCA.
    - Rendering a normalized confusion matrix for classification tasks.
    - Plotting true-vs-predicted residuals for regression tasks.

All plotting functions save their output directly to disk (`out_path`) and
close their figure afterward; none return a value or keep a figure open.
"""

import os
import math
import random
from pathlib import Path
from typing import Tuple, List, Union

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    raise ImportError(
        "HEIC support requires 'pillow-heif'. Install it with: pip install pillow-heif"
    )


def plot_class_grid(
    dataset_dir: Union[str, Path],
    out_path: Union[str, Path],
    samples_per_class: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    title: str = "Dataset Class Overview",
    seed: int = 42) -> None:
    """
    Scans a directory of class subfolders and generates a grid overview image
    where each class cell displays a multi-image grouped collage.

    Parameters
    ----------
    dataset_dir : str or Path
        Root directory containing subfolders named after each class.
    out_path : str or Path
        Destination file path to save the generated grid overview image.
    samples_per_class : int, default=4
        Number of images to randomly sample and arrange into a collage per class.
    image_size : tuple of int, default=(224, 224)
        Target resolution (width, height) for individual sample images.
    title : str, default="Dataset Class Overview"
        Global title displayed above the entire image grid.
    seed : int, default=42
        Random seed for reproducible image selection.

    Returns
    -------
    None
        Saves the resulting plot directly to `out_path` and closes the figure.

    Raises
    ------
    FileNotFoundError
        If `dataset_dir` does not exist or contains no valid class subfolders.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"The dataset path '{dataset_dir}' does not exist.")

    samples_per_class = max(1, samples_per_class)
    class_samples = []
    rng = random.Random(seed)
    # Added .heic and .heif to the accepted extensions
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif")

    # Gather all valid class subfolders and their images
    for class_folder in sorted(dataset_path.iterdir()):
        if class_folder.is_dir() and not class_folder.name.startswith('.'):
            image_files = [
                f for f in class_folder.iterdir()
                if f.is_file() and f.suffix.lower() in valid_exts
            ]
            if image_files:
                n_samples = min(len(image_files), samples_per_class)
                selected_imgs = rng.sample(image_files, n_samples)
                class_samples.append((class_folder.name, selected_imgs))

    if not class_samples:
        raise FileNotFoundError(f"No valid image files found in '{dataset_dir}'.")

    n_classes = len(class_samples)

    # Dynamically compute grid rows and columns based on total class count
    ncols = math.ceil(math.sqrt(n_classes))
    nrows = math.ceil(n_classes / ncols)

    plt.rcParams.update({'font.family': 'sans-serif', 'axes.edgecolor': '#CCCCCC', 'axes.linewidth': 0.8})

    # Force squeeze=False so plt.subplots ALWAYS returns a 2D numpy array of axes, preventing single-element collapses
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), squeeze=False)
    axes_flat = axes.flatten()

    # Calculate inner collage dimensions per class cell
    inner_cols = math.ceil(math.sqrt(samples_per_class))
    inner_rows = math.ceil(samples_per_class / inner_cols)
    single_w = image_size[0] // inner_cols
    single_h = image_size[1] // inner_rows

    for rank, (class_name, img_paths) in enumerate(class_samples):
        ax = axes_flat[rank]
        collage = Image.new("RGB", (inner_cols * single_w, inner_rows * single_h), (255, 255, 255))

        for idx, img_path in enumerate(img_paths):
            try:
                # pillow_heif's register_heif_opener() lets Image.open handle
                # .heic/.heif transparently, same as any other format
                with Image.open(img_path) as img:
                    img_resized = img.convert("RGB").resize((single_w, single_h), Image.Resampling.LANCZOS)
                    x_pos = (idx % inner_cols) * single_w
                    y_pos = (idx // inner_cols) * single_h
                    collage.paste(img_resized, (x_pos, y_pos))
            except Exception as err:
                print(f"Error loading {img_path}: {err}")

        ax.imshow(collage)
        ax.set_title(class_name, fontsize=12, fontweight="bold")
        ax.axis("off")

    # Hide any unused subplots if the grid slots exceed the number of classes
    for rank in range(n_classes, len(axes_flat)):
        axes_flat[rank].axis("off")

    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_pca_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    out_path: Union[str, Path],
    title: str = "PCA Embeddings"
) -> None:
    """
    Generates and saves a 2D PCA projection plot of feature embeddings.

    Supports both categorical classification targets and continuous
    regression labels: integer-valued labels are drawn as discrete, colored
    classes with a legend, while float-valued labels are drawn as a
    continuous scatter with a colorbar.

    Parameters
    ----------
    embeddings : numpy.ndarray
        High-dimensional feature embeddings of shape `(N, D)`.
    labels : numpy.ndarray
        Class index array (int) or continuous values (float) of shape `(N,)`.
    class_names : list of str
        Human-readable labels corresponding to categorical class indices.
        Only used when `labels` is integer-valued.
    out_path : str or Path
        Destination path where the output image will be saved.
    title : str, default="PCA Embeddings"
        Title header for the projection plot.

    Returns
    -------
    None
        Saves the output figure to disk and closes the figure handle.
    """
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(embeddings)

    plt.figure(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")

    if np.issubdtype(np.array(labels).dtype, np.floating):
        scatter = plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="viridis", s=30, alpha=0.8)
        plt.colorbar(scatter, label="Target Value")
    else:
        for c in sorted(np.unique(labels)):
            mask = np.array(labels) == c
            plt.scatter(Z[mask, 0], Z[mask, 1], label=class_names[c], color=cmap(c), s=35, alpha=0.8)
        plt.legend(fontsize=9, loc="best")

    plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=11)
    plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_classification_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_path: Union[str, Path],
    title: str = "Confusion Matrix"
) -> None:
    """
    Renders and saves a normalized classification confusion matrix with
    percentage overlays.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth class targets of shape `(N,)`.
    y_pred : numpy.ndarray
        Model-predicted class targets of shape `(N,)`.
    class_names : list of str
        List of class label strings, ordered to match class indices.
    out_path : str or Path
        Destination path to write the output image.
    title : str, default="Confusion Matrix"
        Title header for the matrix plot.

    Returns
    -------
    None
        Saves the output figure directly to disk and closes the figure handle.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(max(7, len(class_names) * 1.2), max(5, len(class_names) * 1.0)))
    plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(label="Normalized Accuracy")

    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.title(title, fontweight="bold")

    thresh = cm_norm.max() / 2.0 if cm_norm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)"
            plt.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_regression_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Union[str, Path],
    title: str = "Residual Plot"
) -> None:
    """
    Plots true vs. predicted regression targets alongside an ideal identity
    line.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth continuous target values.
    y_pred : numpy.ndarray
        Model-predicted continuous values.
    out_path : str or Path
        Output path where the residual plot will be saved.
    title : str, default="Residual Plot"
        Title header for the plot.

    Returns
    -------
    None
        Saves the output plot to disk and releases memory resources.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, color="teal", edgecolor="k")

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Fit")

    plt.xlabel("True Values", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted Values", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()