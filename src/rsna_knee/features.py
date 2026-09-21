"""Trich dac trung bang backbone dong bang (ngay 5).

Vi sao dong bang backbone truoc khi fine-tune:
  * 58 study co nhan la qua it de cap nhat hang chuc trieu tham so - fine-tune thang se
    khop thuoc du lieu ngay trong vai epoch.
  * Trich mot lan roi train head hang tram lan mat vai giay moi lan, nen vong thu nghiem
    (chon pooling, chon lr, chon nhan) quay rat nhanh.
  * No tao ra mot moc so sanh that su cho ngay 9: fine-tune phai thang duoc CAI NAY moi
    duoc goi la co tac dung.

Feature cua ca dataset la (so_lat, d) moi study - vai chuc MB, nam gon trong RAM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Chuan hoa cua ImageNet - moi backbone pretrained deu ky vong dau vao da tru mean/chia std
# theo dung bo so nay. Bo qua buoc nay khong bao loi, chi lam dac trung te di mot cach am tham.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_backbone(name: str = "resnet18") -> tuple[torch.nn.Module, int]:
    """Tra ve (model da eval + freeze, so chieu dac trung)."""
    if name.startswith("dinov2"):
        model = torch.hub.load("facebookresearch/dinov2", name)
        dim = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024}[name]
    elif name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        model, dim = net, 512
    else:
        raise ValueError(f"Backbone chua ho tro: {name}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)   # dong bang that su, khong chi dua vao torch.no_grad()
    return model, dim


def to_three_channel(slices: np.ndarray) -> torch.Tensor:
    """(N, H, W) trong [0,1] -> (N, 3, H, W) da chuan hoa theo ImageNet.

    MRI la anh mot kenh; backbone pretrained an 3 kenh. Lap lai kenh la cach re nhat va
    giu duoc trong so pretrained - doi lop conv dau tien se vut bo chinh phan da hoc.
    """
    x = torch.from_numpy(slices).float().unsqueeze(1).repeat(1, 3, 1, 1)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


@torch.no_grad()
def extract_study_features(
    slices: np.ndarray,
    model: torch.nn.Module,
    device: str = "cpu",
    batch_size: int = 16,
) -> np.ndarray:
    """(N, H, W) -> (N, d). Chay theo lo de khong do VRAM khi series co nhieu lat."""
    x = to_three_channel(slices)
    outs = []
    for i in range(0, len(x), batch_size):
        chunk = x[i : i + batch_size].to(device)
        outs.append(model(chunk).float().cpu())
    return torch.cat(outs).numpy()


def save_features(features: dict[str, np.ndarray], path: str | Path) -> None:
    """Luu dang .npz - mot file duy nhat, de chuyen tu Kaggle ve may."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **features)


def load_features(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def ensure_features(
    manifest,
    image_root,
    backbone: str = "resnet18",
    size: int = 224,
    device: str = "cpu",
    cache_path: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Trich dac trung cho nhung study CHUA co trong cache, roi ghi lai ca bo.

    Phai kiem tung study chu khong chi kiem file cache co ton tai hay khong: du lieu duoc
    tai dan tung dot, nen mot cache cu van "ton tai" nhung thieu nhung study moi tai ve -
    va ket qua la train tren it du lieu hon minh tuong ma khong co gi bao.
    """
    from .dataset import StudyDataset

    cache_path = Path(cache_path) if cache_path else None
    feats = load_features(cache_path) if (cache_path and cache_path.exists()) else {}

    from .config import STUDY_COL

    todo = manifest[~manifest[STUDY_COL].astype(str).isin(feats)]
    if len(todo) == 0:
        if verbose:
            print(f"  cache du: {len(feats)} study, khong can trich them")
        return feats

    ds = StudyDataset(todo, image_root, size=size)
    model, _ = build_backbone(backbone)
    model.to(device)
    if verbose:
        print(f"  trich them {len(todo)} study (da co {len(feats)})")

    for i in range(len(ds)):
        x, _, uid = ds[i]
        feats[uid] = extract_study_features(x.numpy(), model, device=device)
        if verbose and (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(ds)}")

    if cache_path:
        save_features(feats, cache_path)
        if verbose:
            print(f"  ghi {cache_path.name}: {len(feats)} study, "
                  f"{cache_path.stat().st_size/1024**2:.1f} MB")
    return feats
