
import os
import copy
import random
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ImageFile.LOAD_TRUNCATED_IMAGES = True


BASE_DIR = "PATH_TO_MOISTURE_BASE_DIR"
PARQUET_PATH = os.path.join(BASE_DIR, "metadata.parquet")
TARGET_COL = "moisture_content_percent"
IMAGE_PATH_COLS = ("jpeg_file", "image_path")
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

class MoistureDataset(Dataset):
    """
    A PyTorch Dataset for loading images and their corresponding moisture content targets.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing metadata, including image paths and target values.
    tf : callable
        The torchvision transform pipeline to apply to the images.
    """
    def __init__(self, df: pd.DataFrame, tf):
        self.df = df.reset_index(drop=True)
        self.tf = tf
        
    def __len__(self): 
        """
        Get the total number of samples in the dataset.

        Returns
        -------
        int
            The number of samples.
        """
        return len(self.df)
        
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
            A tuple containing the transformed image tensor, the regression target tensor,
            and the original image path.
        """
        r = self.df.iloc[i]
        img_path = str(r.get(IMAGE_PATH_COLS[0], r.get(IMAGE_PATH_COLS[1], "")))
        if not os.path.isabs(img_path): img_path = os.path.join(BASE_DIR, img_path)
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        return self.tf(img), torch.tensor(float(r[TARGET_COL]), dtype=torch.float32), img_path


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

def load_lejepa(backbone, ckpt_path):
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

    Returns
    -------
    torch.nn.Module
        The backbone loaded with the remapped LeJEPA weights. If the checkpoint 
        is not found, the original backbone is returned unchanged.
    """
    if not os.path.exists(ckpt_path): return backbone
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    
    target_keys = set(backbone.state_dict().keys())
    remapped = {}
    for k, v in sd.items():
        k_clean = k.replace("backbone.", "").replace("features.", "")
        if k_clean in target_keys: remapped[k_clean] = v
        elif f"features.{k_clean}" in target_keys: remapped[f"features.{k_clean}"] = v
    backbone.load_state_dict(remapped, strict=False)
    return backbone

# ==============================================================================
# EXECUTION
# ==============================================================================
class LinearRegHead(nn.Module):
    """
    A single-layer linear regression head for evaluating frozen features.

    Parameters
    ----------
    dim : int
        The dimensionality of the input features.

    Attributes
    ----------
    fc : torch.nn.Linear
        The fully connected linear regression layer mapping features to a single scalar.
    """
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        
    def forward(self, x): 
        """
        Perform a forward pass through the regression head.

        Parameters
        ----------
        x : torch.Tensor
            The input feature embeddings.

        Returns
        -------
        torch.Tensor
            The scalar regression predictions.
        """
        return self.fc(x).squeeze(-1)

def extract_embeddings(backbone, loader):
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
        A tuple containing (embeddings_tensor, targets_tensor), both mapped to CPU.
    """
    backbone.eval()
    backbone.to(DEVICE)
    all_z, all_y = [], []
    with torch.no_grad():
        for x, y, _ in tqdm.tqdm(loader, leave=False):
            all_z.append(backbone(x.to(DEVICE)).cpu())
            all_y.append(y.cpu())
    return torch.cat(all_z), torch.cat(all_y)

def run_regression():
    """
    Execute the full linear probing moisture regression pipeline.

    This function coordinates metadata loading, dataset splitting, feature 
    extraction across 4 backbones, and probe training. Evaluation metrics 
    (R2, MAE, RMSE) are calculated and exported to an Excel summary file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(BASE_SEED)
    
    df = pd.read_parquet(PARQUET_PATH).dropna(subset=[TARGET_COL])
    ds = MoistureDataset(df, eval_tf)
    
    tr_idx, te_idx = train_test_split(np.arange(len(ds)), test_size=0.20, random_state=BASE_SEED)
    
    s_tr = Subset(ds, tr_idx); s_tr.dataset = copy.copy(ds); s_tr.dataset.tf = train_tf
    s_te = Subset(ds, te_idx); s_te.dataset = copy.copy(ds); s_te.dataset.tf = eval_tf
    
    tr_ldr = DataLoader(s_tr, batch_size=BATCH_SIZE, shuffle=False)
    te_ldr = DataLoader(s_te, batch_size=BATCH_SIZE, shuffle=False)

    models_dict = {
        "lejepa_resnet18": load_lejepa(ResNet18Backbone(), CKPT_RESNET),
        "lejepa_efficientnetb0": load_lejepa(EfficientNetBackbone(), CKPT_EFFICIENTNET),
        "lejepa_mobilenetv2": load_lejepa(MobileNetBackbone(), CKPT_MOBILENET),
        "lejepa_customcnn": load_lejepa(CustomCNNBackbone(), CKPT_CUSTOMCNN),
    }

    results = []
    for model_name, backbone in models_dict.items():
        print(f"\nEvaluating: {model_name}")
        for p in backbone.parameters(): p.requires_grad = False
        
        X_tr, y_tr = extract_embeddings(backbone, tr_ldr)
        X_te, y_te = extract_embeddings(backbone, te_ldr)
        
        head = LinearRegHead(backbone.out_dim).to(DEVICE)
        opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.MSELoss()
        
        tds = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
        for _ in range(EPOCHS):
            head.train()
            for z, y in tds:
                opt.zero_grad()
                loss_fn(head(z.to(DEVICE)), y.to(DEVICE)).backward()
                opt.step()
                
        head.eval()
        with torch.no_grad():
            preds = head(X_te.to(DEVICE)).cpu().numpy()
            y_te_np = y_te.numpy()
            
        r2 = r2_score(y_te_np, preds)
        mae = mean_absolute_error(y_te_np, preds)
        rmse = np.sqrt(mean_squared_error(y_te_np, preds))
        
        results.append({"Model": model_name, "R2": round(r2,4), "MAE": round(mae,4), "RMSE": round(rmse,4)})
        print(f"  -> R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    pd.DataFrame(results).to_excel(os.path.join(OUTPUT_DIR, "moisture_summary.xlsx"), index=False)

if __name__ == "__main__":
    run_regression()