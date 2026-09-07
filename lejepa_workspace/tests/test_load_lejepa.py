"""Round-trip tests: weights saved by pretraining must load into every consumer."""
import pytest
import torch

from lejepa_Core import Backbone_pretrain as bp
from lejepa_Core import Evaluate_classification as ec
from lejepa_Core import Evaluate_moisture as em

PAIRS = [
    ("resnet18", bp.ResNet18Encoder, ec.ResNet18Backbone, em.ResNet18Backbone),
    ("efficientnetb0", bp.EfficientNetB0Encoder, ec.EfficientNetBackbone, em.EfficientNetBackbone),
    ("mobilenetv2", bp.MobileNetV2Encoder, ec.MobileNetBackbone, em.MobileNetBackbone),
    ("convnet", bp.ConvNetEncoder, ec.CustomCNNBackbone, em.CustomCNNBackbone),
]


def _randomize(module):
    """Perturb every float tensor so a 'not loaded' outcome cannot hide behind default init."""
    with torch.no_grad():
        for t in module.state_dict().values():
            if t.dtype.is_floating_point:
                t.add_(torch.rand_like(t) * 0.1)  # non-negative keeps BN running_var valid
    return module


def _backbone_items(encoder_sd):
    """Checkpoint entries belonging to the feature extractor (i.e. not the projection head)."""
    return {k: v for k, v in encoder_sd.items() if not k.startswith("proj.")}


@pytest.mark.parametrize("name,Encoder,ClsBackbone,RegBackbone", PAIRS)
def test_roundtrip_into_pretrain_encoder(tmp_path, name, Encoder, ClsBackbone, RegBackbone):
    src = _randomize(Encoder(proj_dim=32))
    ckpt = tmp_path / f"{name}.pth"
    torch.save(src.state_dict(), ckpt)

    dst = bp.load_lejepa(Encoder(proj_dim=32), str(ckpt))
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v), k


@pytest.mark.parametrize("name,Encoder,ClsBackbone,RegBackbone", PAIRS)
@pytest.mark.parametrize("which", ["classification", "moisture"])
def test_roundtrip_into_eval_backbone(tmp_path, name, Encoder, ClsBackbone, RegBackbone, which):
    src = _randomize(Encoder(proj_dim=32))
    ckpt = tmp_path / f"{name}.pth"
    torch.save(src.state_dict(), ckpt)

    Backbone = ClsBackbone if which == "classification" else RegBackbone
    dst = bp.load_lejepa(Backbone(), str(ckpt))

    expected = _backbone_items(src.state_dict())
    remapped, unmatched = bp.remap_lejepa_state_dict(src.state_dict(), dst.state_dict().keys())
    # Every backbone tensor maps somewhere, only the projection head is left over.
    assert set(unmatched) == set(src.state_dict()) - set(expected)
    assert len(remapped) == len(expected)
    dst_sd = dst.state_dict()
    for target_key, v in remapped.items():
        assert torch.equal(dst_sd[target_key], v), target_key

    # And the two modules produce identical features for the same input.
    src.eval(); dst.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        z_src, _ = src(x.unsqueeze(1))
        z_dst = dst(x)
    assert torch.allclose(z_src[:, 0], z_dst, atol=1e-5)


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        bp.load_lejepa(bp.ResNet18Encoder(), str(tmp_path / "nope.pth"))


def test_wrong_architecture_raises(tmp_path):
    ckpt = tmp_path / "resnet.pth"
    torch.save(bp.ResNet18Encoder().state_dict(), ckpt)
    with pytest.raises(RuntimeError):
        bp.load_lejepa(ec.EfficientNetBackbone(), str(ckpt))


def test_shipped_resnet18_checkpoint_loads():
    import pathlib
    ckpt = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "checkpoints" / "lejepa_resnet18_pretrained.pth"
    if not ckpt.exists():
        pytest.skip("example checkpoint not present")
    enc = bp.load_lejepa(bp.ResNet18Encoder(), str(ckpt))
    cls = bp.load_lejepa(ec.ResNet18Backbone(), str(ckpt))
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert torch.equal(enc.state_dict()["layer4.1.conv2.weight"], sd["layer4.1.conv2.weight"])
    assert torch.equal(cls.state_dict()["features.7.1.conv2.weight"], sd["layer4.1.conv2.weight"])
