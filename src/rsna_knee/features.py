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

    # Truong hop pho bien nhat sau khi trich tren Kaggle: cache da du, khong con gi de lam.
    # Phai thoat SOM - mot Series rong co dtype object, nen `~s` va `s.sum()` phia duoi se
    # nem loi kho hieu thay vi chay qua.
    if len(todo) == 0:
        if verbose:
            print(f"  cache du: {len(feats)} study, khong can trich them")
        return feats

    # Chi trich duoc cho study CO ANH tren dia. Tu khi dac trung duoc trich tren Kaggle,
    # phan lon study co dac trung ma khong co anh o may - do la binh thuong, khong phai
    # thieu sot. Khong loc o day thi StudyDataset nem FileNotFoundError ngay study dau.
    from .config import short_uid

    root = Path(image_root)
    co_anh = todo[STUDY_COL].astype(str).map(lambda u: (root / short_uid(u)).is_dir())
    bo_qua = int((~co_anh).sum())
    todo = todo[co_anh]

    if len(todo) == 0:
        if verbose:
            print(f"  cache du: {len(feats)} study; {bo_qua} study con lai chua co anh o may")
        return feats
    if verbose and bo_qua:
        print(f"  bo qua {bo_qua} study chua co anh o may (chi trich cho {len(todo)} study)")

    ds = StudyDataset(todo, image_root, size=size)
    model, _ = build_backbone(backbone)
    model.to(device)
    if verbose:
        print(f"  trich them {len(todo)} study (da co {len(feats)})")

    # Mot study hong KHONG duoc lam sap ca lan chay. Da gap that: MemoryError khi vua giu
    # dac trung cua 1,500 study trong RAM vua doc mot series 1024x1024. Bo qua study do va
    # bao ra con hon mat toan bo cong viec - nhung phai BAO, khong duoc im lang.
    loi: list[tuple[str, str]] = []
    for i in range(len(ds)):
        uid = str(ds.manifest.iloc[i][STUDY_COL])
        try:
            x, _, _ = ds[i]
            feats[uid] = extract_study_features(x.numpy(), model, device=device)
        except Exception as exc:
            loi.append((uid, f"{type(exc).__name__}: {exc}"))
        if verbose and (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(ds)}" + (f" (hong {len(loi)})" if loi else ""))

    if loi and verbose:
        print(f"  {len(loi)} study khong trich duoc, da bo qua:")
        for uid, err in loi[:5]:
            print(f"    ...{uid[-12:]}  {err[:90]}")

    if cache_path:
        save_features(feats, cache_path)
        if verbose:
            print(f"  ghi {cache_path.name}: {len(feats)} study, "
                  f"{cache_path.stat().st_size/1024**2:.1f} MB")
    return feats


def usable_studies(cache_path, image_root, extract_missing: bool = False) -> set[str]:
    """Study dung duoc cho mot thi nghiem.

    MAC DINH: chi lay study DA CO dac trung trong cache. Hai ly do:
      * Nhat quan. Dac trung trich tren Kaggle chay GPU fp16, trich o may chay CPU fp32.
        Do tuong dong cosin giua hai ben la 0.99999 nen tron lai van dung, nhung mot thi
        nghiem chi nen co MOT nguon dac trung - de sau nay con quy duoc nguyen nhan.
      * Tai nguyen. Trich them o may trong khi dang giu dac trung cua hang nghin study
        trong RAM da lam segfault vi het bo nho - va segfault thi khong `try` nao bat duoc.

    `extract_missing=True` mo lai che do cu: gop them study co anh o may de trich tai cho.
    Chi dung khi chua co cache, hoac khi that su can vai study le.
    """
    from .manifest import complete_local_studies

    cache_path = Path(cache_path)
    co_dac_trung: set[str] = set()
    if cache_path.exists():
        with np.load(cache_path) as z:
            co_dac_trung = set(z.files)

    if co_dac_trung and not extract_missing:
        return co_dac_trung
    return co_dac_trung | complete_local_studies()
