
import os
import copy
import random
import logging
from pathlib import Path
import tqdm

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import torch_directml
from torchvision import models
from torchvision.transforms import v2

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

ImageFile.LOAD_TRUNCATED_IMAGES = True


CLASSIFICATION_ROOT = "PATH_TO_CLASSIFICATION_DATASET"
OUTPUT_DIR = "PATH_TO_OUTPUT_DIR"

CKPT_RESNET = "PATH_TO_LEJEPA_RESNET_CKPT"
CKPT_EFFICIENTNET = "PATH_TO_LEJEPA_EFFNET_CKPT"
CKPT_MOBILENET = "PATH_TO_LEJEPA_MOBILENET_CKPT"
CKPT_CUSTOMCNN = "PATH_TO_LEJEPA_CUSTOMCNN_CKPT"

DEVICE = torch_directml.device()
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.00499971
WEIGHT_DECAY = 1.82e-06
BASE_SEED = 0
MULTIPLE_SEEDS = [0, 1, 2, 3, 4]
NUM_WORKERS = 0

CLASSIFICATION_SHOTS = {"one_shot": 1, "ten_shot": 10, "full": None}
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
ZOOM_CLASSES = ["alfalfa", "haylage", "tmr"]
CHOSEN_ZOOM = 0.15

def set_seed(seed: int):
    """
    Set the global seed for random number generators to ensure reproducibility.

    Parameters
    ----------
    seed : int
        The integer seed value to use for the random, numpy, and torch modules.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


train_tf = v2.Compose([
    v2.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(10),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_tf = v2.Compose([
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def safe_open(path: str):
    """
    Safely open an image file and convert it to RGB.

    If the image fails to open or is corrupted, a blank black RGB image
    matching the configured IMAGE_SIZE is returned instead.

    Parameters
    ----------
    path : str
        The file path to the image.

    Returns
    -------
    PIL.Image.Image
        The loaded RGB image or a blank RGB placeholder image.
    """
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))

class ImageFolderDataset(Dataset):
    """
    A PyTorch Dataset that loads images from a directory structure.

    Assumes a directory structure where each subfolder corresponds to a classification class.

    Parameters
    ----------
    root : str or pathlib.Path
        The root directory containing the class subfolders.

    Attributes
    ----------
    samples : list of tuple
        A list containing tuples of (file_path, class_index).
    classes : list of str
        A list of class names found in the root directory.
    class_to_idx : dict
        A mapping of class names to their corresponding integer index.
    """
    def __init__(self, root):
        self.samples = []
        root = Path(root)
        class_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith((".", "_"))]
        self.classes = [p.name for p in class_dirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for c in self.classes:
            idx = self.class_to_idx[c]
            for f in (root / c).rglob("*"):
                if f.is_file() and f.suffix.lower() in VALID_EXTS:
                    self.samples.append((str(f), idx))
                    
    def __len__(self): 
        """
        Get the total number of samples in the dataset.

        Returns
        -------
        int
            The number of valid images found.
        """
        return len(self.samples)

class ZoomPreprocessingDataset(Dataset):
    """
    A wrapper dataset that applies a center crop zoom to specific classes before standard transforms.

    Parameters
    ----------
    base_dataset : torch.utils.data.Dataset
        The underlying dataset providing samples and classes.
    tf : callable
        The torchvision transform pipeline to apply after the zoom operation.
    zoom_classes : list of str
        A list of class names that should have the zoom applied.
    zoom_factor : float
        The fraction of the image dimensions to keep during the crop.

    Attributes
    ----------
    classes : list of str
        The classes from the base dataset.
    samples : list of tuple
        The sample references from the base dataset.
    """
    def __init__(self, base_dataset, tf, zoom_classes, zoom_factor):
        self.base_dataset = base_dataset
        self.tf = tf
        self.zoom_classes = zoom_classes
        self.zoom_factor = zoom_factor
        self.classes = base_dataset.classes
        self.samples = base_dataset.samples

    def __len__(self): 
        """
        Get the total number of samples in the dataset.

        Returns
        -------
        int
            The number of samples.
        """
        return len(self.base_dataset)
        
    def __getitem__(self, i):
        """
        Retrieve and process a single sample from the dataset.

        Parameters
        ----------
        i : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing the transformed image tensor, the label tensor,
            and the original image path.
        """
        path, label_idx = self.base_dataset.samples[i]
        class_name = self.base_dataset.classes[label_idx].lower()
        img = safe_open(path)
        if any(target in class_name for target in self.zoom_classes):
            w, h = img.size
            left, top = w * (0.5 - self.zoom_factor / 2), h * (0.5 - self.zoom_factor / 2)
            right, bottom = w * (0.5 + self.zoom_factor / 2), h * (0.5 + self.zoom_factor / 2)
            img = img.crop((int(left), int(top), int(right), int(bottom)))
        return self.tf(img), torch.tensor(int(label_idx), dtype=torch.long), path


class CustomCNNBackbone(nn.Module):
    """
    A custom Convolutional Neural Network backbone for feature extraction.

    Attributes
    ----------
    features : torch.nn.Sequential
        The sequential convolutional and pooling layers.
    out_dim : int
        The dimension of the output feature vector.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = 512
        
    def forward(self, x): 
        """
        Perform a forward pass through the custom CNN.

        Parameters
        ----------
        x : torch.Tensor
            The input image tensor.

        Returns
        -------
        torch.Tensor
            The flattened feature embeddings.
        """
        return self.features(x).flatten(1)

class ResNet18Backbone(nn.Module):
    """
    A ResNet18 backbone modified for feature extraction.

    Parameters
    ----------
    weights : torchvision.models.ResNet18_Weights or None, optional
        Pretrained weights to initialize the backbone.

    Attributes
    ----------
    features : torch.nn.Sequential
        The convolutional blocks of the ResNet18 model.
    pool : torch.nn.AdaptiveAvgPool2d
        The final pooling layer.
    out_dim : int
        The dimension of the output feature vector.
    """
    def __init__(self, weights=None):
        super().__init__()
        m = models.resnet18(weights=weights)
        self.features = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512
        
    def forward(self, x): 
        """
        Perform a forward pass through the ResNet18 backbone.

        Parameters
        ----------
        x : torch.Tensor
            The input image tensor.

        Returns
        -------
        torch.Tensor
            The flattened feature embeddings.
        """
        return self.pool(self.features(x)).flatten(1)

class EfficientNetBackbone(nn.Module):
    """
    An EfficientNet-B0 backbone modified for feature extraction.

    Parameters
    ----------
    weights : torchvision.models.EfficientNet_B0_Weights or None, optional
        Pretrained weights to initialize the backbone.

    Attributes
    ----------
    features : torch.nn.Sequential
        The core feature layers of the EfficientNet model.
    pool : torch.nn.AdaptiveAvgPool2d
        The final pooling layer.
    out_dim : int
        The dimension of the output feature vector.
    """
    def __init__(self, weights=None):
        super().__init__()
        m = models.efficientnet_b0(weights=weights)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1280
        
    def forward(self, x): 
        """
        Perform a forward pass through the EfficientNet backbone.

        Parameters
        ----------
        x : torch.Tensor
            The input image tensor.

        Returns
        -------
        torch.Tensor
            The flattened feature embeddings.
        """
        return self.pool(self.features(x)).flatten(1)

class MobileNetBackbone(nn.Module):
    """
    A MobileNetV2 backbone modified for feature extraction.

    Parameters
    ----------
    weights : torchvision.models.MobileNet_V2_Weights or None, optional
        Pretrained weights to initialize the backbone.

    Attributes
    ----------
    features : torch.nn.Sequential
        The core feature layers of the MobileNet model.
    pool : torch.nn.AdaptiveAvgPool2d
        The final pooling layer.
    out_dim : int
        The dimension of the output feature vector.
    """
    def __init__(self, weights=None):
        super().__init__()
        m = models.mobilenet_v2(weights=weights)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1280
        
    def forward(self, x): 
        """
        Perform a forward pass through the MobileNet backbone.

        Parameters
        ----------
        x : torch.Tensor
            The input image tensor.

        Returns
        -------
        torch.Tensor
            The flattened feature embeddings.
        """
        return self.pool(self.features(x)).flatten(1)

def load_lejepa(backbone, ckpt_path, is_custom=False):
    """
    Load custom LeJEPA pretrained weights into the given backbone model.

    Handles structural mismatches and prefixes that commonly arise when loading 
    custom pretraining state dictionaries into standard torchvision architectures.

    Parameters
    ----------
    backbone : torch.nn.Module
        The neural network backbone to populate with weights.
    ckpt_path : str
        The file path to the saved PyTorch checkpoint.
    is_custom : bool, optional
        A flag indicating whether the backbone is the custom CNN (default is False).

    Returns
    -------
    torch.nn.Module
        The backbone loaded with the remapped LeJEPA weights. If the checkpoint 
        is not found, the original randomly initialized backbone is returned.
    """
    if not os.path.exists(ckpt_path):
        print(f"[WARNING] Checkpoint not found: {ckpt_path}. Returning randomly initialized backbone.")
        return backbone

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for wrapper in ("state_dict", "model_state_dict", "model", "backbone_state_dict"):
            if wrapper in sd and isinstance(sd[wrapper], dict):
                sd = sd[wrapper]
                break

    target_keys = set(backbone.state_dict().keys())
    remapped = {}
    
    for k, v in sd.items():
        k_clean = k.replace("backbone.", "").replace("encoder.", "").replace("features.", "")
        
        if is_custom:
            if k_clean in target_keys:
                remapped[k_clean] = v
            elif f"features.{k_clean}" in target_keys:
                remapped[f"features.{k_clean}"] = v
        else:
            if k_clean in target_keys:
                remapped[k_clean] = v
            elif f"features.{k_clean}" in target_keys:
                remapped[f"features.{k_clean}"] = v
            else:
                remapped[k_clean] = v

    backbone.load_state_dict(remapped, strict=False)
    return backbone


def make_80_20_splits(dataset):
    """
    Create a stratified 80/20 train/test split based on class labels.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to split. Must contain a `base_dataset.samples` attribute 
        containing paths and labels.

    Returns
    -------
    tuple of list
        A tuple containing two lists: (train_indices, test_indices).
    """
    y = np.array([dataset.base_dataset.samples[i][1] for i in range(len(dataset))], dtype=int)
    tr, te = train_test_split(np.arange(len(dataset)), test_size=0.20, random_state=BASE_SEED, stratify=y)
    return tr.tolist(), te.tolist()

def extract_and_cache_embeddings(backbone, loader):
    """
    Extract and cache feature embeddings from a data loader using a frozen backbone.

    Parameters
    ----------
    backbone : torch.nn.Module
        The frozen neural network model used to extract features.
    loader : torch.utils.data.DataLoader
        The data loader providing the images.

    Returns
    -------
    tuple of torch.Tensor
        A tuple containing (embeddings_tensor, labels_tensor), both mapped to CPU.
    """
    backbone.eval()
    backbone.to(DEVICE)
    all_z, all_y = [], []
    with torch.no_grad():
        for x, y, _ in tqdm.tqdm(loader, desc="Caching", leave=False):
            all_z.append(backbone(x.to(DEVICE)).cpu())
            all_y.append(y.cpu())
    return torch.cat(all_z), torch.cat(all_y)

def select_few_shot_indices(y_tr, shots, seed):
    """
    Select a balanced few-shot subset of indices from the training set.

    Parameters
    ----------
    y_tr : torch.Tensor
        The tensor containing all training labels.
    shots : int or None
        The exact number of samples to select per class. If None, returns 
        all available training indices.
    seed : int
        The random seed used to sample the instances for reproducibility.

    Returns
    -------
    list of int
        A randomly shuffled list of selected integer indices for the few-shot split.
    """
    if shots is None: return np.arange(len(y_tr))
    rng = np.random.RandomState(seed)
    selected = []
    for c in torch.unique(y_tr):
        c_idx = torch.where(y_tr == c)[0].numpy()
        chosen = rng.choice(c_idx, shots, replace=False) if len(c_idx) > shots else c_idx
        selected.extend(chosen)
    rng.shuffle(selected)
    return selected

class LinearClassHead(nn.Module):
    """
    A single-layer linear classification head for evaluating frozen features.

    Parameters
    ----------
    dim : int
        The dimensionality of the input features.
    n_classes : int
        The number of target output classes.

    Attributes
    ----------
    fc : torch.nn.Linear
        The fully connected linear classification layer.
    """
    def __init__(self, dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(dim, n_classes)
        
    def forward(self, x): 
        """
        Perform a forward pass through the classification head.

        Parameters
        ----------
        x : torch.Tensor
            The input feature embeddings.

        Returns
        -------
        torch.Tensor
            The raw classification logits.
        """
        return self.fc(x)

def run_classification():
    """
    Execute the full linear probing classification pipeline.

    This function coordinates dataset initialization, stratified splitting, feature 
    extraction across 4 backbones, and probe training across multiple few-shot and 
    full dataset regimes. Results are exported to an Excel summary file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cls_ds = ZoomPreprocessingDataset(ImageFolderDataset(CLASSIFICATION_ROOT), eval_tf, ZOOM_CLASSES, CHOSEN_ZOOM)
    c_tr, c_te = make_80_20_splits(cls_ds)
    n_classes = len(cls_ds.classes)
    
    train_subset = Subset(cls_ds, c_tr); train_subset.dataset = copy.copy(cls_ds); train_subset.dataset.tf = train_tf
    test_subset = Subset(cls_ds, c_te); test_subset.dataset = copy.copy(cls_ds); test_subset.dataset.tf = eval_tf
    
    tr_ldr = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False)
    te_ldr = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    models_dict = {
        "lejepa_resnet18": load_lejepa(ResNet18Backbone(), CKPT_RESNET),
        "lejepa_efficientnetb0": load_lejepa(EfficientNetBackbone(), CKPT_EFFICIENTNET),
        "lejepa_mobilenetv2": load_lejepa(MobileNetBackbone(), CKPT_MOBILENET),
        "lejepa_customcnn": load_lejepa(CustomCNNBackbone(), CKPT_CUSTOMCNN, is_custom=True),
    }

    results = []
    for model_name, backbone in models_dict.items():
        print(f"\nEvaluating: {model_name}")
        for p in backbone.parameters(): p.requires_grad = False
        
        X_tr_full, y_tr_full = extract_and_cache_embeddings(backbone, tr_ldr)
        X_te, y_te = extract_and_cache_embeddings(backbone, te_ldr)
        
        for regime, shots in CLASSIFICATION_SHOTS.items():
            seed_accs = []
            for s in MULTIPLE_SEEDS:
                idx = select_few_shot_indices(y_tr_full, shots, s)
                X_tr, y_tr = X_tr_full[idx], y_tr_full[idx]
                
                head = LinearClassHead(backbone.out_dim, n_classes).to(DEVICE)
                optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
                loss_fn = nn.CrossEntropyLoss()
                
                tds = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
                for _ in range(EPOCHS):
                    head.train()
                    for z, y in tds:
                        optimizer.zero_grad()
                        loss_fn(head(z.to(DEVICE)), y.to(DEVICE).long()).backward()
                        optimizer.step()
                
                head.eval()
                with torch.no_grad():
                    preds = head(X_te.to(DEVICE)).argmax(1).cpu()
                    seed_accs.append(accuracy_score(y_te.numpy(), preds.numpy()))
                
                if shots is None: break # Run full only once
            
            acc_str = f"{np.mean(seed_accs)*100:.2f}% ± {np.std(seed_accs)*100:.2f}%" if shots else f"{seed_accs[0]*100:.2f}%"
            results.append({"Model": model_name, "Regime": regime, "Accuracy": acc_str})
            print(f"  -> {regime}: {acc_str}")

    pd.DataFrame(results).to_excel(os.path.join(OUTPUT_DIR, "classification_summary.xlsx"), index=False)

if __name__ == "__main__":
    run_classification()