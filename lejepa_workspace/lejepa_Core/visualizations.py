
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import pandas as pd

def plot_pca_2d(embeddings, labels, class_names, out_path, title="PCA Embeddings"):
    """
    Generates and saves a 2D PCA projection of the latent feature embeddings.

    Automatically detects whether the labels are continuous (for regression) or
    categorical (for classification) and adjusts the colormap and legend accordingly.

    Parameters
    ----------
    embeddings : array-like of shape (n_samples, n_features)
        The high-dimensional feature embeddings to be projected.
    labels : array-like of shape (n_samples,)
        The target values or class indices corresponding to each embedding.
    class_names : list of str or dict
        A mapping from class index to class name. Used for the legend when 
        labels are categorical.
    out_path : str or pathlib.Path
        The destination file path where the generated plot will be saved.
    title : str, optional
        The title of the plot. Default is "PCA Embeddings".
    """
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(embeddings)

    plt.figure(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")

    # If labels are continuous (Regression), use a scatter colorbar
    if np.issubdtype(np.array(labels).dtype, np.floating):
        scatter = plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="viridis", s=30, alpha=0.8)
        plt.colorbar(scatter, label="Target Value")
    else:
        # Categorical labels (Classification)
        for c in sorted(np.unique(labels)):
            mask = np.array(labels) == c
            plt.scatter(Z[mask, 0], Z[mask, 1], label=class_names[c], color=cmap(c), s=35, alpha=0.8)
        plt.legend(fontsize=9, loc="best")

    plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_classification_confusion_matrix(y_true, y_pred, class_names, out_path, title="Confusion Matrix"):
    """
    Renders and saves a normalized confusion matrix with percentage overlaps.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The ground truth (correct) target labels.
    y_pred : array-like of shape (n_samples,)
        The estimated targets as returned by a classifier.
    class_names : list of str
        An ordered list of class names corresponding to the label indices.
    out_path : str or pathlib.Path
        The destination file path where the generated plot will be saved.
    title : str, optional
        The title of the plot. Default is "Confusion Matrix".
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
            plt.text(j, i, txt, ha="center", va="center", 
                     color="white" if cm_norm[i, j] > thresh else "black", fontsize=9)
            
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_regression_residuals(y_true, y_pred, out_path, title="Residual Plot"):
    """
    Plots true vs. predicted values alongside an ideal fit line for regression evaluation.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The ground truth (correct) continuous target values.
    y_pred : array-like of shape (n_samples,)
        The estimated continuous target values returned by a regression model.
    out_path : str or pathlib.Path
        The destination file path where the generated plot will be saved.
    title : str, optional
        The title of the plot. Default is "Residual Plot".
    """
    plt.figure(figsize=(8, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.6, color="teal", edgecolor="k")
    
    # Ideal line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal Fit")
    
    plt.xlabel("True Values", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted Values", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()