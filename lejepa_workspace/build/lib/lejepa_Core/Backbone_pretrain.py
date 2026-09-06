"""Code for loading and pretraining 4 backbones with a unified selector module."""
import os
import logging
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.ops import MLP
from torchvision.transforms import v2


def tile_image(img: Image.Image):
    """
    Split an image into four equal non-overlapping tiles (quadrants).

    Parameters
    ----------
    img : PIL.Image.Image
        The input image to be tiled.

    Returns
    -------
    list of PIL.Image.Image
        A list containing four image tiles (top-left, top-right, bottom-left, bottom-right).
    """
    w, h = img.size
    return [
        img.crop((0, 0, w // 2, h // 2)),
        img.crop((w // 2, 0, w, h // 2)),
        img.crop((0, h // 2, w // 2, h)),
        img.crop((w // 2, h // 2, w, h)),
    ]

class ForageTileDataset(Dataset):
    """
    A PyTorch Dataset that loads images, splits them into tiles, and applies augmentations
    for multi-view contrastive pretraining.

    Parameters
    ----------
    root : str or pathlib.Path
        The root directory containing the dataset images.
    image_size : int, optional
        The target size for the random resized crop augmentation. Default is 128.
    views : int, optional
        The number of augmented tile views to return per image. Default is 2.

    Attributes
    ----------
    root : pathlib.Path
        The path to the dataset directory.
    views : int
        The number of views to generate per sample.
    paths : list of str
        The file paths of all valid images found in the root directory.
    aug : torchvision.transforms.v2.Compose
        The composition of geometric and normalization augmentations applied to the tiles.

    Raises
    ------
    FileNotFoundError
        If no valid image files are found in the specified root directory.
    """
    def __init__(self, root, image_size=128, views=2):
        self.root = Path(root)
        self.views = views
        self.paths = [
            str(self.root / f) for f in os.listdir(self.root)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        
        if not self.paths:
            raise FileNotFoundError(f"No valid images found in {self.root}")

        self.aug = v2.Compose([
            v2.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.1),
            v2.RandomRotation(degrees=25),
            v2.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.85, 1.15)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        """
        Get the total number of images in the dataset.

        Returns
        -------
        int
            The number of images.
        """
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Retrieve and process a single sample from the dataset.

        Parameters
        ----------
        idx : int
            The index of the image to retrieve.

        Returns
        -------
        torch.Tensor
            A stacked tensor of the augmented tile views with shape (views, C, H, W).
        """
        img = Image.open(self.paths[idx]).convert("RGB")
        tiles = tile_image(img)
        selected = np.random.choice(4, size=self.views, replace=False)
        views = [self.aug(tiles[i]) for i in selected]
        return torch.stack(views, dim=0)


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization (SIGReg) module.
    
    Regularizes the feature space by forcing the empirical distribution 
    of projections to match an isotropic Gaussian.

    Parameters
    ----------
    knots : int, optional
        The number of interpolation knots used to estimate the density. Default is 17.
    """
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        Compute the SIGReg loss statistic.

        Parameters
        ----------
        proj : torch.Tensor
            The projected embeddings tensor of shape (V, B, D).

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the computed SIGReg loss.
        """
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)
        V, B, D = proj.shape
        A = torch.randn(D, 256, device=proj.device)
        A = A / (A.norm(dim=0, keepdim=True) + 1e-12)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * B
        return statistic.mean()

def lejepa_prediction_loss(proj: torch.Tensor) -> torch.Tensor:
    """
    Compute the L2 prediction loss across multiple views.

    Parameters
    ----------
    proj : torch.Tensor
        The projected embeddings tensor.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the mean squared error (variance) across views.
    """
    mu = proj.mean(dim=1, keepdim=True)
    dif = mu - proj
    return dif.square().mean()


class ConvNetEncoder(nn.Module):
    """
    A simple custom Convolutional Neural Network backbone.

    Parameters
    ----------
    proj_dim : int, optional
        The dimensionality of the output projection head. Default is 128.
    """
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        """
        Perform a forward pass through the ConvNet encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, V, C, H, W).

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing:
            - The unprojected base features of shape (B, V, 512).
            - The projected features of shape (B, V, proj_dim).
        """
        B, V = x.shape[:2]
        x = x.flatten(0, 1)
        z = self.backbone(x).flatten(1)
        p = self.proj(z)
        return z.view(B, V, -1), p.view(B, V, -1)

class EfficientNetB0Encoder(nn.Module):
    """
    An EfficientNet-B0 backbone adapted for multi-view processing.

    Parameters
    ----------
    proj_dim : int, optional
        The dimensionality of the output projection head. Default is 128.
    """
    def __init__(self, proj_dim=128):
        super().__init__()
        eff = models.efficientnet_b0(weights=None)
        self.backbone = eff.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = eff.classifier[1].in_features
        self.proj = MLP(feat_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        """
        Perform a forward pass through the EfficientNet-B0 encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, V, C, H, W).

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing:
            - The unprojected base features of shape (B, V, feat_dim).
            - The projected features of shape (B, V, proj_dim).
        """
        B, V = x.shape[:2]
        x = x.flatten(0, 1)
        z = self.pool(self.backbone(x)).flatten(1)
        p = self.proj(z)
        return z.view(B, V, -1), p.view(B, V, -1)

class MobileNetV2Encoder(nn.Module):
    """
    A MobileNetV2 backbone adapted for multi-view processing.

    Parameters
    ----------
    proj_dim : int, optional
        The dimensionality of the output projection head. Default is 128.
    """
    def __init__(self, proj_dim=128):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=None)
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = mobilenet.last_channel
        self.proj = MLP(feat_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        """
        Perform a forward pass through the MobileNetV2 encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, V, C, H, W).

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing:
            - The unprojected base features of shape (B, V, feat_dim).
            - The projected features of shape (B, V, proj_dim).
        """
        B, V = x.shape[:2]
        x = x.flatten(0, 1)
        z = self.pool(self.backbone(x)).flatten(1)
        p = self.proj(z)
        return z.view(B, V, -1), p.view(B, V, -1)

class ResNet18Encoder(nn.Module):
    """
    A ResNet18 backbone adapted for multi-view processing.

    Parameters
    ----------
    proj_dim : int, optional
        The dimensionality of the output projection head. Default is 128.
    """
    def __init__(self, proj_dim=128):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        """
        Perform a forward pass through the ResNet18 encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, V, C, H, W).

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing:
            - The unprojected base features of shape (B, V, 512).
            - The projected features of shape (B, V, proj_dim).
        """
        B, V = x.shape[:2]
        x = x.flatten(0, 1)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        z = self.pool(x).flatten(1)
        p = self.proj(z)
        return z.view(B, V, -1), p.view(B, V, -1)

# Registry to help the user select models easily
BACKBONE_REGISTRY = {
    "convnet": ConvNetEncoder,
    "efficientnetb0": EfficientNetB0Encoder,
    "mobilenetv2": MobileNetV2Encoder,
    "resnet18": ResNet18Encoder
}


def run_pretraining_loop(model, loader, epochs, lr, device, lambda_reg, save_path):
    """
    Execute the LeJEPA training loop on the selected model and dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network encoder model to train.
    loader : torch.utils.data.DataLoader
        The PyTorch DataLoader providing the batched views.
    epochs : int
        The total number of training epochs to complete.
    lr : float
        The learning rate for the AdamW optimizer.
    device : str or torch.device
        The device ('cpu' or 'cuda') to perform training on.
    lambda_reg : float
        The scalar weight balancing the prediction loss and the SIGReg loss.
    save_path : str
        The destination file path to save the pretrained model state dictionary.

    Returns
    -------
    torch.nn.Module
        The trained neural network model.
    """
    model = model.to(device)
    sigreg = SIGReg().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        for batch_idx, views in enumerate(loader):
            views = views.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            z, p = model(views) 
            
            # Compute losses
            pred_loss = lejepa_prediction_loss(p)
            sig_loss = sigreg(p.transpose(0, 1)) # Transpose for SIGReg expected shape
            
            # Combined LeJEPA Loss
            loss = (1.0 - lambda_reg) * pred_loss + lambda_reg * sig_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        logging.info(f"Epoch [{epoch}/{epochs}] | Loss: {avg_loss:.4f}")
        print(f"Epoch [{epoch}/{epochs}] | Avg Loss: {avg_loss:.4f}")
        
    torch.save(model.state_dict(), save_path)
    print(f"Pretraining complete! Weights saved to {save_path}")
    return model

def start_pretraining(
    data_dir: str, 
    model_name: str, 
    proj_dim: int = 128,
    epochs: int = 100, 
    batch_size: int = 16, 
    lr: float = 1e-3, 
    lambda_reg: float = 0.5,
    save_dir: str = "checkpoints"
):
    """
    High-level user function to initiate pretraining with a single command.
    
    Parameters
    ----------
    data_dir : str
        Path to the folder containing the JPG/PNG dataset images.
    model_name : str
        The string identifier for the chosen backbone. Must be one of 
        ['convnet', 'efficientnetb0', 'mobilenetv2', 'resnet18'].
    proj_dim : int, optional
        The dimensionality of the projection head. Default is 128.
    epochs : int, optional
        The number of training epochs. Default is 100.
    batch_size : int, optional
        The training batch size. Default is 16.
    lr : float, optional
        The learning rate for the optimizer. Default is 1e-3.
    lambda_reg : float, optional
        The trade-off parameter between prediction loss and SIGReg loss. Default is 0.5.
    save_dir : str, optional
        The directory where the final weights will be saved. Default is "checkpoints".

    Raises
    ------
    ValueError
        If an invalid `model_name` is provided that does not exist in `BACKBONE_REGISTRY`.
    """
    # 1. Validate the selected model
    model_name = model_name.lower()
    if model_name not in BACKBONE_REGISTRY:
        raise ValueError(f"Invalid model_name '{model_name}'. Choose from: {list(BACKBONE_REGISTRY.keys())}")
    
    # 2. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing pretraining for '{model_name}' on device '{device}'...")

    # 3. Setup Dataset and DataLoader
    print(f"Loading dataset from: {data_dir}")
    dataset = ForageTileDataset(root=data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Found {len(dataset)} images. Training in batches of {batch_size}.")

    # 4. Initialize Model
    model = BACKBONE_REGISTRY[model_name](proj_dim=proj_dim)
    
    # 5. Setup Output Directory
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"lejepa_{model_name}_pretrained.pth")
    
    # 6. Execute Training
    run_pretraining_loop(model, loader, epochs, lr, device, lambda_reg, save_path)


def load_lejepa(backbone, ckpt_path):
    """
    Safely load pretrained LeJEPA weights into a given backbone model.

    This function mitigates structure and mapping issues (such as double-prefixes 
    or unmapped keys) when loading custom state dictionaries into standard 
    architectures for downstream evaluation.

    Parameters
    ----------
    backbone : torch.nn.Module
        The target model architecture to populate with loaded weights.
    ckpt_path : str
        The file path to the saved `.pth` checkpoint.

    Returns
    -------
    torch.nn.Module
        The backbone model populated with the matching checkpoint weights.
    """
    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
    remapped = {}
    for k, v in sd.items():
        if not k.startswith("backbone."):
            continue
        k_clean = k.replace("backbone.", "")
        
        if not isinstance(backbone, ResNet18Encoder) and k_clean.startswith("features."):
            k_clean = k_clean.replace("features.", "", 1)
            remapped[f"features.{k_clean}"] = v
        else:
            remapped[k_clean] = v
            
    missing, unexpected = backbone.load_state_dict(remapped, strict=False)
    logging.info(f"Loaded {ckpt_path} | Missing: {len(missing)} | Unexpected: {len(unexpected)}")
    return backbone