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


def select_device():
    """
    Pick the best available torch device.

    Returns
    -------
    str or torch.device
        A DirectML device if ``torch_directml`` is installed (Windows), else
        ``"cuda"`` if available, else ``"mps"`` on Apple Silicon, else ``"cpu"``.
    """
    try:
        import torch_directml  # noqa: F401  (optional, Windows only)
        return torch_directml.device()
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    device = select_device()
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


# Mapping from the attribute names used by ``ResNet18Encoder`` (and the raw
# torchvision ResNet) to the positional indices used when the same layers are
# wrapped in an ``nn.Sequential`` (as the ``*Backbone`` classes in the
# evaluation modules do).
_RESNET_SEQUENTIAL_INDEX = {
    "conv1": "0",
    "bn1": "1",
    "layer1": "4",
    "layer2": "5",
    "layer3": "6",
    "layer4": "7",
}

_STATE_DICT_WRAPPERS = ("state_dict", "model_state_dict", "model", "backbone_state_dict")


def _unwrap_state_dict(obj):
    """Return the raw parameter dict from a checkpoint that may wrap it."""
    if isinstance(obj, dict):
        for wrapper in _STATE_DICT_WRAPPERS:
            if wrapper in obj and isinstance(obj[wrapper], dict):
                return obj[wrapper]
    return obj


def _candidate_keys(key):
    """
    Generate the target-module key names a checkpoint key could correspond to.

    Checkpoints written by :func:`run_pretraining_loop` use either bare
    attribute names (``ResNet18Encoder``: ``conv1.weight``, ``layer1.0...``,
    ``proj.0.weight``) or a ``backbone.`` prefix (the other encoders:
    ``backbone.0.0.weight``). Downstream feature extractors wrap the same
    layers under ``features.`` and, for ResNet18, as a positional
    ``nn.Sequential``. This yields every plausible spelling so the caller can
    pick whichever exists in the target module.
    """
    stripped = key
    for prefix in ("backbone.", "encoder.", "features."):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break

    candidates = [key, stripped, f"backbone.{stripped}", f"features.{stripped}"]

    head, sep, rest = stripped.partition(".")
    if sep and head in _RESNET_SEQUENTIAL_INDEX:
        idx = _RESNET_SEQUENTIAL_INDEX[head]
        candidates.append(f"features.{idx}.{rest}")
        candidates.append(f"backbone.{idx}.{rest}")

    seen, ordered = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def remap_lejepa_state_dict(sd, target_keys):
    """
    Rename checkpoint keys so they line up with ``target_keys``.

    Parameters
    ----------
    sd : dict
        A (possibly wrapped) state dict produced by LeJEPA pretraining.
    target_keys : iterable of str
        The keys of the module the weights will be loaded into.

    Returns
    -------
    tuple of (dict, list of str)
        The remapped state dict restricted to keys present in the target, and
        the list of checkpoint keys that could not be matched (typically the
        projection head when loading into a feature extractor).
    """
    sd = _unwrap_state_dict(sd)
    target_keys = set(target_keys)
    remapped, unmatched = {}, []
    for k, v in sd.items():
        for cand in _candidate_keys(k):
            if cand in target_keys:
                remapped[cand] = v
                break
        else:
            unmatched.append(k)
    return remapped, unmatched


def load_lejepa(backbone, ckpt_path, strict=True):
    """
    Load pretrained LeJEPA weights into a backbone, verifying they actually land.

    Works for the encoders defined in this module (``ResNet18Encoder``,
    ``EfficientNetB0Encoder``, ``MobileNetV2Encoder``, ``ConvNetEncoder``) as
    well as the ``*Backbone`` feature extractors in the evaluation modules,
    which wrap the same layers under ``features.``. Checkpoint keys are
    remapped with :func:`remap_lejepa_state_dict`.

    Parameters
    ----------
    backbone : torch.nn.Module
        The target model architecture to populate with loaded weights.
    ckpt_path : str
        The file path to the saved ``.pth`` checkpoint.
    strict : bool, optional
        If True (default), raise if any parameter of ``backbone`` is left
        uninitialised by the checkpoint. Keys present in the checkpoint but
        absent from ``backbone`` (e.g. the projection head when loading into a
        feature extractor) are always ignored.

    Returns
    -------
    torch.nn.Module
        The backbone model populated with the matching checkpoint weights.

    Raises
    ------
    FileNotFoundError
        If ``ckpt_path`` does not exist.
    RuntimeError
        If no checkpoint tensor could be matched to the backbone, or if
        ``strict`` is True and some backbone parameters were not covered.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    remapped, unmatched = remap_lejepa_state_dict(sd, backbone.state_dict().keys())
    if not remapped:
        raise RuntimeError(
            f"No tensor in {ckpt_path} matches {type(backbone).__name__}; "
            f"checkpoint keys look like {list(_unwrap_state_dict(sd))[:3]}"
        )

    missing, unexpected = backbone.load_state_dict(remapped, strict=False)
    logging.info(
        f"Loaded {ckpt_path} into {type(backbone).__name__} | "
        f"matched: {len(remapped)} | missing: {len(missing)} | ignored from ckpt: {len(unmatched)}"
    )
    if unexpected:  # cannot happen since we only pass target keys, but be explicit
        raise RuntimeError(f"Unexpected keys after remapping: {unexpected[:5]}")
    if missing and strict:
        raise RuntimeError(
            f"{len(missing)} parameters of {type(backbone).__name__} were not found in "
            f"{ckpt_path} (e.g. {missing[:3]}). Pass strict=False to allow a partial load."
        )
    return backbone
